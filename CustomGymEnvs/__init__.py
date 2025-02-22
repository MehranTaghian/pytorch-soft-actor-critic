from gym.envs.registration import register

register(
    id="AntEnv-v0",
    entry_point="CustomGymEnvs.envs.ant.AntEnv_v0_Graph.AntEnv_v0_Normal:AntEnvV0",
    max_episode_steps=1000,
)

register(
    id="AntEnv-v1",
    entry_point="CustomGymEnvs.envs.ant.AntEnv_v0_Normal.AntEnv_v0_Normal:AntEnvV1",
    max_episode_steps=1000,
)

# FetchReach

register(
    id="FetchReachEnv-v0",
    entry_point="CustomGymEnvs.envs.fetchreach.CustomFetchReach.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)

register(
    id="FetchReachEnv-v1",
    entry_point="CustomGymEnvs.envs.fetchreach.FetchReachBroken.fetch.reach:FetchReachEnv",
    max_episode_steps=50,
)
