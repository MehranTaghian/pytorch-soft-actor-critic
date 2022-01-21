import gym
import CustomGymEnvs

env = gym.make("AntEnv-v1")

obs = env.reset()

print(obs.shape)