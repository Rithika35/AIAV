import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

import torch
import torch.nn as nn
import os
import time
import argparse
import csv

# ------------------------------------------------------------------------
# 1) Always create a 1200-step speed dataset
# ------------------------------------------------------------------------
DATA_LEN = 1200
CSV_FILE = "speed_profile.csv"

# Force-generate a 1200-step sinusoidal + noise speed profile
speeds = 10 + 5 * np.sin(0.02 * np.arange(DATA_LEN)) + 2 * np.random.randn(DATA_LEN)
df_fake = pd.DataFrame({"speed": speeds})
df_fake.to_csv(CSV_FILE, index=False)
print(f"Created {CSV_FILE} with {DATA_LEN} steps.")

df = pd.read_csv(CSV_FILE)
full_speed_data = df["speed"].values
assert len(full_speed_data) == DATA_LEN, "Dataset must be 1200 steps after generation."

# ------------------------------------------------------------------------
# 2) Utility: chunk the dataset, possibly with leftover
# ------------------------------------------------------------------------
def chunk_into_episodes(data, chunk_size):
    """
    Splits `data` into chunks of length `chunk_size`.
    If leftover < chunk_size remains, it becomes a smaller final chunk.
    """
    episodes = []
    start = 0
    while start < len(data):
        end = start + chunk_size
        chunk = data[start:end]
        episodes.append(chunk)
        start = end
    return episodes

# ------------------------------------------------------------------------
# 3A) Training Environment: picks a random chunk each reset
# ------------------------------------------------------------------------
class TrainEnv(gym.Env):
    """
    Speed-following training environment:
      - The dataset is split into episodes of length `chunk_size`.
      - Each reset(), we pick one chunk at random.
      - action: acceleration in [-3,3]
      - observation: [current_speed, reference_speed]
      - reward: -|current_speed - reference_speed|
    """

    def __init__(self, episodes_list, delta_t=1.0):
        super().__init__() 
        self.episodes_list = episodes_list
        self.num_episodes = len(episodes_list)
        self.delta_t = delta_t

        # Actions, Observations
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=50.0, shape=(2,), dtype=np.float32)

        # Episode-specific
        self.current_episode = None
        self.episode_len = 0
        self.step_idx = 0
        self.current_speed = 0.0
        self.ref_speed = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Pick random chunk from episodes_list
        ep_idx = np.random.randint(0, self.num_episodes)
        self.current_episode = self.episodes_list[ep_idx]
        self.episode_len = len(self.current_episode)
        self.step_idx = 0

        # Initialize
        self.current_speed = 0.0
        self.ref_speed = self.current_episode[self.step_idx]

        obs = np.array([self.current_speed, self.ref_speed], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        accel = np.clip(action[0], -3.0, 3.0)
        self.current_speed += accel * self.delta_t
        if self.current_speed < 0:
            self.current_speed = 0.0

        self.ref_speed = self.current_episode[self.step_idx]
        error = abs(self.current_speed - self.ref_speed)
        reward = -error

        self.step_idx += 1
        terminated = (self.step_idx >= self.episode_len)
        truncated = False

        obs = np.array([self.current_speed, self.ref_speed], dtype=np.float32)
        info = {"speed_error": error}
        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------------------
# 3B) Testing Environment: run entire 1200-step data in one episode
# ------------------------------------------------------------------------
class TestEnv(gym.Env):
    """
    Speed-following testing environment:
      - We run through the entire 1200-step dataset in one go.
      - observation: [current_speed, reference_speed]
      - reward: -|current_speed - reference_speed|
    """

    def __init__(self, full_data, delta_t=1.0):
        super().__init__()
        self.full_data = full_data
        self.n_steps = len(full_data)
        self.delta_t = delta_t

        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=50.0, shape=(2,), dtype=np.float32)

        self.idx = 0
        self.current_speed = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        self.current_speed = 0.0
        ref_speed = self.full_data[self.idx]
        obs = np.array([self.current_speed, ref_speed], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        accel = np.clip(action[0], -3.0, 3.0)
        self.current_speed += accel * self.delta_t
        if self.current_speed < 0:
            self.current_speed = 0.0

        ref_speed = self.full_data[self.idx]
        error = abs(self.current_speed - ref_speed)
        reward = -error

        self.idx += 1
        terminated = (self.idx >= self.n_steps)
        truncated = False

        obs = np.array([self.current_speed, ref_speed], dtype=np.float32)
        info = {"speed_error": error}
        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------------------
# 4) CustomLoggingCallback (optional)
# ------------------------------------------------------------------------
from stable_baselines3.common.callbacks import BaseCallback

class CustomLoggingCallback(BaseCallback):
    def __init__(self, log_dir, log_name="training_log.csv", verbose=1):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.log_name = log_name
        self.log_path = os.path.join(log_dir, log_name)
        self.episode_rewards = []
        os.makedirs(log_dir, exist_ok=True)
        with open(self.log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestep', 'average_reward'])

    def _on_step(self):
        t = self.num_timesteps
        reward = self.locals.get('rewards', [0])[-1]
        self.episode_rewards.append(reward)

        if self.locals.get('dones', [False])[-1]:
            avg_reward = np.mean(self.episode_rewards)
            with open(self.log_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([t, avg_reward])
            self.logger.record("reward/average_reward", avg_reward)
            self.episode_rewards.clear()

        return True


# ------------------------------------------------------------------------
# 5) Main: user sets chunk_size from command line, train, then test
# ------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./logs_chunk_training",
        help="Directory to store logs and trained model."
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100,
        help="Episode length for training (e.g. 50, 100, 200)."
    )
    args = parser.parse_args()

    log_dir = args.output_dir
    os.makedirs(log_dir, exist_ok=True)
    logger = configure(log_dir, ["stdout", "tensorboard"])

    chunk_size = args.chunk_size
    print(f"[INFO] Using chunk_size = {chunk_size}")

    # 5A) Split the 1200-step dataset into chunk_size episodes
    episodes_list = chunk_into_episodes(full_speed_data, chunk_size)
    print(f"Number of episodes: {len(episodes_list)} (some leftover if 1200 not divisible by {chunk_size})")

    # 5B) Create the TRAIN environment
    def make_train_env():
        return TrainEnv(episodes_list, delta_t=1.0)
    n_envs = 4
    train_env = SubprocVecEnv([make_train_env for _ in range(n_envs)])

    # 5C) Build the model (SAC with MlpPolicy)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    policy_kwargs = dict(net_arch=[256, 256], activation_fn=nn.ReLU)
    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=200000,
        tau=0.005,
        gamma=0.99,
        ent_coef='auto',
        device=device
    )

    model.set_logger(logger)

    total_timesteps = 100_000
    callback = CustomLoggingCallback(log_dir)

    print(f"[INFO] Start training for {total_timesteps} timesteps...")
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=100,
        callback=callback
    )
    end_time = time.time()
    print(f"[INFO] Training finished in {end_time - start_time:.2f}s")

    # 5D) Save the model
    save_path = os.path.join(log_dir, f"sac_speed_follow_chunk{chunk_size}")
    model.save(save_path)
    print(f"[INFO] Model saved to: {save_path}.zip")

    # ------------------------------------------------------------------------
    # 5E) Test the model on the FULL 1200-step dataset in one go
    # ------------------------------------------------------------------------
    test_env = TestEnv(full_speed_data, delta_t=1.0)

    obs, _ = test_env.reset()
    predicted_speeds = []
    reference_speeds = []
    rewards = []

    for _ in range(DATA_LEN):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        predicted_speeds.append(obs[0])  # current_speed
        reference_speeds.append(obs[1])  # reference_speed
        rewards.append(reward)
        if terminated or truncated:
            break

    avg_test_reward = np.mean(rewards)
    print(f"[TEST] Average reward over 1200-step test: {avg_test_reward:.3f}")

    # Plot the entire test
    plt.figure(figsize=(10, 5))
    plt.plot(reference_speeds, label="Reference Speed", linestyle="--")
    plt.plot(predicted_speeds, label="Predicted Speed", linestyle="-")
    plt.xlabel("Timestep")
    plt.ylabel("Speed (m/s)")
    plt.title(f"Test on full 1200-step dataset (chunk_size={chunk_size})")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
