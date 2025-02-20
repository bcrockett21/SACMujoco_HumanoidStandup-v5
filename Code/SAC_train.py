import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.sac.policies import Actor
from gymnasium import ObservationWrapper
import torch.nn as nn
import csv


Actor.LOG_STD_MIN = -5
Actor.LOG_STD_MAX = 2



# custom callback to add rewards to csv as it trains
class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_file, verbose=1):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.log_file = log_file
        self.episode_rewards = None
        self.episode_counts = None

        # initialize CSV file
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Reward"])

    def _on_training_start(self) -> None:
        num_envs = self.training_env.num_envs
        self.episode_rewards = [0] * num_envs
        self.episode_counts = [0] * num_envs

    def _on_step(self) -> bool:
        rewards = self.locals['rewards']
        dones = self.locals['dones']
        
        for i, done in enumerate(dones):
            self.episode_rewards[i] += rewards[i]

            if done:
                self.episode_counts[i] += 1
                print(f"Env {i}, Episode: {self.episode_counts[i]}, Reward: {self.episode_rewards[i]}")
                
                with open(self.log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.episode_counts[i], self.episode_rewards[i]])
                self.episode_rewards[i] = 0
        return True


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



# function to create environment
def make_env():
    env = gym.make("HumanoidStandup-v5")
    env = TruncatedObservationWrapper(env)
    return env


# custom weights and biases
def custom_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=5000000,
    save_path='/mnt/c//Users/Blake/Documents/checkpoints/',
    name_prefix="sac_humanoid_sac",
    save_replay_buffer=True,
    save_vecnormalize=True
)

# initialize the environment
env = DummyVecEnv([make_env])


# SAC model definition
SAC_model = SAC(
    "MlpPolicy", 
    env,
    learning_rate=3e-4,
    buffer_size=1000000,
    learning_starts=10000,
    batch_size=256,
    train_freq=1,
    gamma=0.99,
    tau=0.005,
    ent_coef='auto',
    gradient_steps=1,
    target_update_interval=1,
    sde_sample_freq=-1,
    target_entropy=-17,
    use_sde_at_warmup=False,
    verbose=1,
    tensorboard_log="./tensorboard_logs/sac_humanoid/",
    device='cuda'
)

"""""""""
new_SAC_model is used for loading a previously trained model from
a checkpoint
"""""""""

# new_SAC_model = SAC.load("/mnt/c//Users/Blake/Documents/checkpoints/sac_humanoid_sac_10000000_steps")
# new_SAC_model.set_env(env)
# new_SAC_model.load_replay_buffer("/mnt/c//Users/Blake/Documents/checkpoints/sac_humanoid_sac_replay_buffer_10000000_steps.pkl")


SAC_model.policy.apply(custom_init)


# Confirm the model is on the GPU
#print(f"Policy device: {next(model.policy.parameters())}")

# Callbacks
log_file_path = "file_path.csv"
reward_logger_callback = RewardLoggerCallback(log_file=log_file_path)
callback_list = CallbackList([checkpoint_callback, reward_logger_callback])

# Train the model
SAC_model.learn(total_timesteps=20000000, callback=callback_list)
# new_SAC_model.learn(total_timesteps=20000000, callback=callback_list, reset_num_timesteps=False)

# Save the model
SAC_model.save("file_path")
# new_SAC_model.save("/mnt/c//Users/Blake/Documents/sac_humanoid_sac")
print(f"Training complete! Rewards saved to {log_file_path}")
