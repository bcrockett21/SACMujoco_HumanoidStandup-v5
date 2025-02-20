import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.spaces import Box
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.sac.policies import Actor
from gymnasium import ObservationWrapper


Actor.LOG_STD_MIN = -5
Actor.LOG_STD_MAX = 2



# custom observation space
class TruncatedObservationWrapper(ObservationWrapper):

    def __init__(self, env):
        super(TruncatedObservationWrapper, self).__init__(env)
        self.original_observation_space = env.observation_space

        # create new observation space
        low = self.original_observation_space.low[:45]
        high = self.original_observation_space.high[:45]
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        # truncate observation space to first 45 values
        return obs[:45]



def make_env():
    env = gym.make("HumanoidStandup-v5")
    env = TruncatedObservationWrapper(env)
    return env

env = DummyVecEnv([make_env])
env.training = False
env.norm_reward = False

# load the trained model
SAC_model = SAC.load("file_path", env=env)
SAC_model.set_env(env)
SAC_model.load_replay_buffer("file_path.pkl")


# initialize the environment
rewards = []
# run a test loop
for i in range(100):
    
    obs = env.reset()
    done = False
    total = 0
    env.render()
    while not done:

        action, _states = SAC_model.predict(obs)
        obs, reward, done, info = env.step(action)
        total += reward.item()

    print(i + 1, total)
    rewards.append((i+1, total))

average = sum([reward for _, reward in rewards]) / len(rewards)
std_dev = np.std([reward for _, reward in rewards])
print(f"Average Reward: {average} +- {std_dev}")

rewards_df = pd.DataFrame(rewards, columns=["Episode", "Rewards"])
rewards_df.to_csv("file_path.csv", index=False)
env.close()
    
