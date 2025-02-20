import gymnasium as gym
import numpy as np
import cv2
from gymnasium.spaces import Box
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium import ObservationWrapper



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
    env = gym.make("HumanoidStandup-v5", render_mode="rgb_array")
    env = TruncatedObservationWrapper(env)
    return env

env = DummyVecEnv([make_env])
env.training = False
env.norm_reward = False
SAC_model = SAC.load("file_path", env=env)
SAC_model.set_env(env)
SAC_model.load_replay_buffer("file_path.pkl")



output_file = "file_path.mp4"
fps = 60
width, height = 4000, 3000
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

base_env = env.envs[0]


obs = base_env.reset()
obs = obs[0]
done = False


for i in range(1000):

    print(f"RENDERING FRAME {i+1}")
    action, _ = SAC_model.predict(obs)
    action = action.squeeze()
    obs = base_env.step(action)
    obs = obs[0]
    frame = base_env.render()
    frame_resized = cv2.resize(frame, (width, height))
    frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
    video_writer.write(frame_bgr)

    if done:
        obs = base_env.reset()

video_writer.release()
env.close()
print("Video saved as:", output_file)
