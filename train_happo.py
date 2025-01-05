from marllib import marl

# prepare the environment
env = marl.make_env(environment_name="mamujoco", map_name="4AgentAnt")

# initialize algorithm and load hyperparameters
happo = marl.algos.happo(
    hyperparam_source="mamujoco",
    batch_mode="complete_episodes",
    num_sgd_iter=5,
    vf_clip_param=10.0,
)

# build agent model based on env + algorithms + user preference if checked available
model = marl.build_model(env, happo, {"core_arch": "gru"})

# start learning + extra experiment settings if needed. remember to check ray.yaml before use
happo.fit(
    env,
    model,
    stop={'episode_reward_mean': 2000, 'timesteps_total': 20000000},
    local_mode=False,
    num_gpus=1,
    num_workers=0,
    share_policy='individual',
    checkpoint_freq=500
)
