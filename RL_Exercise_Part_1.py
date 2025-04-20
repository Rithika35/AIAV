import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
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
# matplotlib.use('Qt5Agg')
# os.environ["QT_QPA_PLATFORM"] = "xcb"
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
        reward = -0.8*error - 0.2*abs(accel)
        # reward = -error

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
        reward = -0.8*error - 0.2*abs(accel)
        # reward = -error

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

        self.episode_errors = []  # Track errors for MAE/MSE
        self.timesteps = []
        self.avg_rewards = []
        os.makedirs(log_dir, exist_ok=True)
        with open(self.log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestep', 'average_reward', 'mae', 'mse'])

    def _on_step(self):
        t = self.num_timesteps
        reward = self.locals.get('rewards', [0])[-1]
        error = self.locals.get('infos', [{}])[-1].get('speed_error', 0)
        self.episode_rewards.append(reward)
        self.episode_errors.append(error)

        if self.locals.get('dones', [False])[-1]:
            avg_reward = np.mean(self.episode_rewards)
            mae = np.mean(self.episode_errors)
            mse = np.mean(np.square(self.episode_errors))
            with open(self.log_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([t, avg_reward, mae, mse])
            self.timesteps.append(t)
            self.avg_rewards.append(avg_reward)
            self.logger.record("reward/average_reward", avg_reward)
            self.logger.record("error/mae", mae)
            self.logger.record("error/mse", mse)
            self.episode_rewards.clear()
            self.episode_errors.clear()

        return True


    def plot_convergence(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.timesteps, self.avg_rewards, label="Average Reward")
        plt.xlabel("Timestep")
        plt.ylabel("Average Reward")
        plt.title("Training Convergence Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "convergence_curve.png"))
        plt.show()


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
    #train_env = SubprocVecEnv([make_train_env for _ in range(n_envs)])
    train_env = DummyVecEnv([make_train_env for _ in range(n_envs)])

    # 5C) Build the model (SAC with MlpPolicy)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    policy_kwargs = dict(net_arch=[256, 256], activation_fn=nn.ReLU)

    # Action noise for exploration (required for DDPG)
    n_actions = train_env.action_space.shape[-1]  # 1 for your accel action
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # # DDPG model
    # model = DDPG(
    #     policy="MlpPolicy",
    #     env=train_env,
    #     verbose=1,
    #     policy_kwargs=policy_kwargs,
    #     learning_rate=3e-4,
    #     batch_size=256,
    #     buffer_size=100000,
    #     tau=0.005,
    #     gamma=0.99,
    #     action_noise=action_noise,  # Exploration noise
    #     learning_starts=10000,  # Delay learning until buffer has some data
    #     train_freq=1,  # Train every step
    #     gradient_steps=1,  # One gradient step per training
    #     device=device
    # )

    # # TD3 model
    model = TD3(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        batch_size=128,
        buffer_size=100000,
        tau=0.005,
        gamma=0.99,
        action_noise=action_noise,
        learning_starts=10000,  # Delay learning until buffer has data
        train_freq=1,  # Train every step
        gradient_steps=1,  # One gradient step per training
        policy_delay=2,  # TD3-specific: Delay policy updates (default 2)
        target_policy_noise=0.2,  # TD3-specific: Noise added to target policy
        target_noise_clip=0.5,  # TD3-specific: Clip target policy noise
        device=device
    )
    # model = SAC(
    #     policy="MlpPolicy",
    #     env=train_env,
    #     policy_kwargs=dict(net_arch=[256, 256]),  # Keep as is
    #     learning_rate=3e-4,
    #     batch_size=256,
    #     buffer_size=500000,
    #     tau=0.005,
    #     gamma=0.99,
    #     ent_coef='auto',
    #     device=device
    # )
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
    errors = []

    for _ in range(DATA_LEN):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        predicted_speeds.append(obs[0])  # current_speed
        reference_speeds.append(obs[1])  # reference_speed
        rewards.append(reward)
        errors.append(info["speed_error"])
        if terminated or truncated:
            break

    # Quantitative Metrics
    avg_test_reward = np.mean(rewards)
    mae_test = np.mean(errors)
    mse_test = np.mean(np.square(errors))
    print(f"[TEST] Average Reward over 1200-step dataset: {avg_test_reward:.3f}")
    print(f"[TEST] Mean Absolute Error (MAE): {mae_test:.3f}")
    print(f"[TEST] Mean Squared Error (MSE): {mse_test:.3f}")

    # Visualizations
    # 1. Speed Comparison
    plt.figure(figsize=(12, 10))
    plt.subplot(3, 1, 1)
    plt.plot(reference_speeds, label="Reference Speed", linestyle="--")
    plt.plot(predicted_speeds, label="Predicted Speed", linestyle="-")
    plt.xlabel("Timestep")
    plt.ylabel("Speed (m/s)")
    plt.title(f"Test on Full 1200-step Dataset (chunk_size={chunk_size})")
    plt.legend()

    # 2. Convergence Curve (from training)
    convergence_rate = np.diff(callback.avg_rewards) / np.diff(callback.timesteps)
    plt.subplot(3, 1, 2)
    plt.plot(callback.timesteps[1:], convergence_rate, label="Convergence Rate")
    plt.xlabel("Timestep")
    plt.ylabel("Reward Change Rate")
    plt.title("Convergence Rate Over Time")

    # 3. Cumulative Error Metrics
    cumulative_mae = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    cumulative_mse = np.cumsum(np.square(errors)) / np.arange(1, len(errors) + 1)
    plt.subplot(3, 1, 3)
    plt.plot(cumulative_mae, label="Cumulative MAE")
    plt.plot(cumulative_mse, label="Cumulative MSE")
    plt.xlabel("Timestep")
    plt.ylabel("Cumulative Error")
    plt.title("Cumulative Error Metrics Over Time")
    plt.legend()

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "td3_batch.jpeg"))
    plt.show()


if __name__ == "__main__":
    main()
