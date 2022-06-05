"""
Implement TRPO and HATRPO in Ray Rllib
__author__: minquan
__data__: May-15
"""

import logging
import re
from typing import Dict, List, Type, Union, Tuple
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.torch_ops import apply_grad_clipping, \
    explained_variance, sequence_mask
import numpy as np
from ray.rllib.evaluation.postprocessing import discount_cumsum, Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_tf, try_import_torch, get_variable
from ray.rllib.utils.typing import TrainerConfigDict, TensorType, \
    LocalOptimizer
from ray.rllib.agents.ppo.ppo import PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.agents.ppo.ppo_torch_policy import PPOTorchPolicy, ValueNetworkMixin, KLCoeffMixin
from ray.rllib.utils.torch_ops import apply_grad_clipping
from ray.rllib.policy.torch_policy import LearningRateSchedule, EntropyCoeffSchedule
from marl.algos.utils.setup_utils import setup_torch_mixins, get_agent_num
from marl.algos.utils.get_hetero_info import (
    get_global_name,
    contain_global_obs,
    hatrpo_post_process,
    value_normalizer,
    MODEL,
    global_state_name,
    STATE,
    TRAINING,
    state_name,
)

from marl.algos.utils.trust_regions import TrustRegionUpdator

from ray.rllib.examples.centralized_critic import CentralizedValueMixin
import ctypes

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()


logger = logging.getLogger(__name__)


def recovery_obj(_id):
    return ctypes.cast(_id, ctypes.py_object).value


def hatrpo_loss_fn(
        policy: Policy, model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch) -> Union[TensorType, List[TensorType]]:
    """Constructs the loss for TRPO
    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]: The action distr. class.
        train_batch (SampleBatch): The training data.
    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """

    CentralizedValueMixin.__init__(policy)

    logits, state = model(train_batch)
    curr_action_dist = dist_class(logits, model)
    # RNN case: Mask away 0-padded chunks at end of time axis.

    if state:
        B = len(train_batch[SampleBatch.SEQ_LENS])
        max_seq_len = logits.shape[0] // B
        mask = sequence_mask(
            train_batch[SampleBatch.SEQ_LENS],
            max_seq_len,
            time_major=model.is_time_major())
        mask = torch.reshape(mask, [-1])
        num_valid = torch.sum(mask)

        def reduce_mean_valid(t):
            return torch.sum(t[mask]) / num_valid

    # non-RNN case: No masking.
    else:
        mask = None
        reduce_mean_valid = torch.mean

    vf_saved = model.value_function

    if contain_global_obs(train_batch):
        opp_action_in_cc = policy.config["model"]["custom_model_config"]["opp_action_in_cc"]
        model.value_function = lambda: policy.model.central_value_function(
            train_batch[STATE],
            train_batch[get_global_name(SampleBatch.ACTIONS)]
            if opp_action_in_cc else None
        )

    agent_model_pat = get_global_name(MODEL, '(\d+)')
    matched_keys = [re.findall(agent_model_pat, key) for key in train_batch]

    collected_agent_ids = [int(m[0]) for m in matched_keys if m]

    contain_opponent_info = all(
        len(train_batch[get_global_name(MODEL, i)]) > 0 and train_batch[get_global_name(MODEL, i)][0] > 0
        for i in collected_agent_ids
    )

    if not contain_opponent_info:
        advantages = train_batch[Postprocessing.ADVANTAGES]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS]) -
            train_batch[SampleBatch.ACTION_LOGP]
        )

        policy_loss = reduce_mean_valid(logp_ratio * advantages)

        cur_model_trust_region_updator = TrustRegionUpdator(
            model=model, dist_class=dist_class, train_batch=train_batch, adv_targ=advantages
        )

        cur_model_trust_region_updator.update_actor(policy_loss=policy_loss)

    else:
        m_advantage = train_batch[Postprocessing.ADVANTAGES]

        agents_num = get_agent_num(policy)
        random_indices = np.random.permutation(range(agents_num))

        policy_losses = []
        kl_losses = []

        def is_current_agent(i): return i == (agents_num - 1)
        # the opponent indices is 0, 1, 2, 3, .. N - 2

        for agent_id in random_indices:
            if is_current_agent(agent_id):
                current_model = model
                logits, state = model(train_batch)
                current_action_dist = dist_class(logits, model)
                old_action_log_dist = train_batch[SampleBatch.ACTION_LOGP]
                actions = train_batch[SampleBatch.ACTIONS]
                train_batch_for_trpo_update = train_batch
                action_dist_input = train_batch[SampleBatch.ACTION_DIST_INPUTS]
            else:
                model_id = int(train_batch[get_global_name(MODEL, agent_id)][0])

                assert model_id > 0, 'model is must > 0, if set to 0 means no model at all'
                current_model = recovery_obj(model_id)

                current_action_logits = train_batch[
                    get_global_name(SampleBatch.ACTION_DIST_INPUTS, agent_id)
                ]

                current_action_dist = dist_class(current_action_logits, None)

                old_action_log_dist = train_batch[
                    get_global_name(SampleBatch.ACTION_LOGP, agent_id)
                ]

                actions = train_batch[get_global_name(SampleBatch.ACTIONS, agent_id)]

                obs = train_batch[get_global_name(SampleBatch.OBS, agent_id)]

                train_batch_for_trpo_update = SampleBatch(
                    obs=obs,
                    seq_lens=train_batch[SampleBatch.SEQ_LENS]
                )

                train_batch_for_trpo_update.is_training = bool(train_batch[get_global_name(TRAINING, agent_id)][0])

                i = 0

                while state_name(i) in train_batch:
                    agent_state_name = global_state_name(i, agent_id)
                    train_batch_for_trpo_update[state_name(i)] = train_batch[agent_state_name]
                    i += 1

            importance_sampling = torch.exp(current_action_dist.logp(actions) - old_action_log_dist)

            cur_updater = TrustRegionUpdator(
                model=current_model, dist_class=dist_class, train_batch=train_batch_for_trpo_update, adv_targ=m_advantage
            )

            loss = reduce_mean_valid(importance_sampling * m_advantage)
            cur_updater.update_actor(loss)

            policy_losses.append(loss)

            m_advantage = importance_sampling * m_advantage

        policy_loss = torch.mean(torch.stack(policy_losses, axis=1), axis=1)

    curr_entropy = curr_action_dist.entropy()

    # Compute a value function loss.
    # if policy.model.model_config['custom_model_config']['normal_value']:
    # value_normalizer.update(train_batch[Postprocessing.VALUE_TARGETS])
    # train_batch[Postprocessing.VALUE_TARGETS] = value_normalizer.normalize(train_batch[Postprocessing.VALUE_TARGETS])

    if policy.config["use_critic"]:
        prev_value_fn_out = train_batch[SampleBatch.VF_PREDS] #
        value_fn_out = model.value_function()  # same as values
        vf_loss1 = torch.pow(
            value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_clipped = prev_value_fn_out + torch.clamp(
            value_fn_out - prev_value_fn_out, -policy.config["vf_clip_param"],
            policy.config["vf_clip_param"])
        vf_loss2 = torch.pow(
            vf_clipped - train_batch[Postprocessing.VALUE_TARGETS], 2.0)
        vf_loss = torch.max(vf_loss1, vf_loss2)
        mean_vf_loss = reduce_mean_valid(vf_loss)
    # Ignore the value function.
    else:
        vf_loss = mean_vf_loss = 0.0

    prev_action_dist = dist_class(train_batch[SampleBatch.ACTION_DIST_INPUTS], model)
    action_kl = prev_action_dist.kl(curr_action_dist)

    model.value_function = vf_saved
    # recovery the value function.

    total_loss = -policy_loss + reduce_mean_valid(policy.kl_coeff * action_kl +
                                                  policy.config["vf_loss_coeff"] * vf_loss -
                                                  policy.entropy_coeff * curr_entropy
                                                  )

    # Store values for stats function in model (tower), such that for
    # multi-GPU, we do not override them during the parallel loss phase.
    mean_kl_loss = reduce_mean_valid(action_kl)
    mean_policy_loss = -policy_loss
    mean_entropy = reduce_mean_valid(curr_entropy)

    model.tower_stats["total_loss"] = total_loss
    model.tower_stats["mean_policy_loss"] = mean_policy_loss
    model.tower_stats["mean_vf_loss"] = mean_vf_loss
    model.tower_stats["vf_explained_var"] = explained_variance(
        train_batch[Postprocessing.VALUE_TARGETS], model.value_function())
    model.tower_stats["mean_entropy"] = mean_entropy
    model.tower_stats["mean_kl_loss"] = mean_kl_loss

    return total_loss


HAPTRPOTorchPolicy = PPOTorchPolicy.with_updates(
        name="HAPPOTorchPolicy",
        get_default_config=lambda: PPO_CONFIG,
        postprocess_fn=hatrpo_post_process,
        loss_fn=hatrpo_loss_fn,
        before_init=setup_torch_mixins,
        extra_grad_process_fn=apply_grad_clipping,
        mixins=[
            EntropyCoeffSchedule, KLCoeffMixin,
            CentralizedValueMixin, LearningRateSchedule,
        ])


def get_policy_class_hatrpo(config_):
    if config_["framework"] == "torch":
        return HAPTRPOTorchPolicy


HATRPOTrainer = PPOTrainer.with_updates(
    name="#hatrpo-trainer",
    default_policy=None,
    get_policy_class=get_policy_class_hatrpo,
)