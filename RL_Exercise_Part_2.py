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
      - A lead vehicle moves ahead with a dynamic speed profile.
      - action: acceleration in [-3,3]
      - observation: [current_speed, reference_speed]
      - reward: -0.8 * |speed_error| - 0.2 * |accel| + distance penalties
    """

    def __init__(self, episodes_list, delta_t=1.0):
        super().__init__() 
        self.episodes_list = episodes_list
        self.num_episodes = len(episodes_list)
        self.delta_t = delta_t

        # Actions, Observations
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -np.inf]),  # current_speed, ref_speed, lead_speed, relative_distance
            high=np.array([50.0, 50.0, 50.0, np.inf]),
            shape=(4,),
            dtype=np.float32
        )
        # Acceleration constraints
        self.previous_acceleration = 0.0  # For tracking jerk

        # Episode-specific
        self.current_episode = None
        self.episode_len = 0
        self.step_idx = 0
        self.current_speed = 0.0 # Ego vehicle speed (m/s)
        self.position = 0.0 # Ego vehicle position(m)
        self.ref_speed = 0.0

        # Lead vehicle state
        self.lead_speed = 0.0  # Lead vehicle speed (m/s)
        self.lead_position = 0.0  # Lead vehicle position (m)
        self.lead_base_speed = 10.0  # Base speed for lead vehicle (m/s)
        self.lead_speed_noise = np.random.normal(0, 1, len(episodes_list[0])) # Noise for lead speed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Pick random chunk from episodes_list
        ep_idx = np.random.randint(0, self.num_episodes)
        self.current_episode = self.episodes_list[ep_idx]
        self.episode_len = len(self.current_episode)
        self.step_idx = 0

        # Initialize ego vehicle
        self.current_speed = 0.0
        self.position = 0.0
        self.previous_acceleration = 0.0

        # Initialize lead vehicle
        self.lead_position = 20.0  # Start 20m ahead
        self.lead_speed = self.lead_base_speed + 5 * np.sin(0.02 * self.step_idx) + self.lead_speed_noise[
            self.step_idx % len(self.lead_speed_noise)]
        self.lead_speed = max(0.0, self.lead_speed)

        self.ref_speed = self.current_episode[self.step_idx]
        relative_distance = self.lead_position - self.position
        obs = np.array([self.current_speed, self.ref_speed, self.lead_speed, relative_distance], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        accel = np.clip(action[0], -3.0, 3.0)
        # Compute jerk for smoothness
        jerk = (accel - self.previous_acceleration) / self.delta_t
        self.previous_acceleration = accel

        # Update ego vehicle
        self.current_speed += accel * self.delta_t
        if self.current_speed < 0:  # Reinstated check
            self.current_speed = 0.0
        self.position += self.current_speed * self.delta_t

        # Update lead vehicle
        self.lead_speed = self.lead_base_speed + 5 * np.sin(0.02 * self.step_idx) + self.lead_speed_noise[
            self.step_idx % len(self.lead_speed_noise)]
        self.lead_speed = max(0.0, self.lead_speed)
        self.lead_position += self.lead_speed * self.delta_t

        # Compute relative distance
        relative_distance = self.lead_position - self.position

        # Update state
        self.ref_speed = self.current_episode[self.step_idx]
        speed_error = abs(self.current_speed - self.ref_speed)  # Track reference speed
        speed_diff = abs(self.current_speed - self.lead_speed)  # Speed similarity with lead
        accel_penalty = abs(accel)
        # Speed tracking (reference speed)
        speed_tracking_penalty = -0.7 * speed_error

        # Speed similarity with lead vehicle
        speed_similarity_reward = 0.1 * np.exp(-speed_diff)

        # Distance penalties
        distance_penalty = 0.0
        safe_distance_bonus = 0.0
        if relative_distance < 5.0:
            distance_penalty = -10.0 * (5.0 - relative_distance)  # Heavy penalty for too close
        elif relative_distance > 30.0:
            distance_penalty = -10.0 * (relative_distance - 30.0)  # Moderate penalty for too far
        else:
            safe_distance_bonus = 2.0  # Bonus for maintaining safe distance


        reward = speed_tracking_penalty + speed_similarity_reward + distance_penalty + safe_distance_bonus -0.2 * accel_penalty -2* abs(jerk)

        self.step_idx += 1
        terminated = (self.step_idx >= self.episode_len)
        truncated = False

        obs = np.array([self.current_speed, self.ref_speed, self.lead_speed, relative_distance], dtype=np.float32)
        info = {
            "speed_error": speed_error,
            "speed_difference": speed_diff,
            "relative_distance": relative_distance,
            "acceleration": accel,
            "jerk": jerk
        }
        return obs, reward, terminated, truncated, info


# ------------------------------------------------------------------------
# 3B) Testing Environment: run entire 1200-step data in one episode
# ------------------------------------------------------------------------
class TestEnv(gym.Env):
    """
    Speed-following testing environment:
      - We run through the entire 1200-step dataset in one go.
      - observation: [current_speed, reference_speed]
      - reward: track ref_speed + speed similarity with lead + distance + smoothness
    """

    def __init__(self, full_data, delta_t=1.0):
        super().__init__()
        self.full_data = full_data
        self.n_steps = len(full_data)
        self.delta_t = delta_t

        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -np.inf]),
            high=np.array([50.0, 50.0, 50.0, np.inf]),
            shape=(4,),
            dtype=np.float32
        )

        self.idx = 0
        self.current_speed = 0.0
        self.position = 0.0
        self.lead_position = 0.0
        self.lead_speed = 0.0
        self.lead_base_speed = 10.0
        self.lead_speed_noise = np.random.normal(0, 1, self.n_steps)
        self.previous_acceleration = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        self.current_speed = 0.0
        self.position = 0.0
        self.lead_position = 20.0  # Start 20m ahead
        self.previous_acceleration = 0.0

        self.lead_speed = self.lead_base_speed + 5 * np.sin(0.02 * self.idx) + self.lead_speed_noise[self.idx]
        self.lead_speed = max(0.0, self.lead_speed)

        ref_speed = self.full_data[self.idx]
        relative_distance = self.lead_position - self.position
        obs = np.array([self.current_speed, ref_speed, self.lead_speed, relative_distance], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        accel = np.clip(action[0], -3.0, 3.0)
        jerk = (accel - self.previous_acceleration) / self.delta_t
        self.previous_acceleration = accel
        self.current_speed += accel * self.delta_t
        if self.current_speed < 0:  # Reinstated check
            self.current_speed = 0.0
        self.position += self.current_speed * self.delta_t

        self.lead_speed = self.lead_base_speed + 5 * np.sin(0.02 * self.idx) + self.lead_speed_noise[self.idx]
        self.lead_speed = max(0.0, self.lead_speed)
        self.lead_position += self.lead_speed * self.delta_t

        relative_distance = self.lead_position - self.position

        ref_speed = self.full_data[self.idx]
        speed_error = abs(self.current_speed - ref_speed)
        speed_diff = abs(self.current_speed - self.lead_speed)
        accel_penalty = abs(accel)

        speed_tracking_penalty = -0.8 * speed_error
        speed_similarity_reward = 0.1 * np.exp(-speed_diff)

        distance_penalty = 0.0
        safe_distance_bonus = 0.0
        if relative_distance < 5.0:
            distance_penalty = -10.0 * (5.0 - relative_distance)
        elif relative_distance > 30.0:
            distance_penalty = -6.0 * (relative_distance - 30.0)
        else:
            safe_distance_bonus = 2.0

        reward = speed_tracking_penalty +speed_similarity_reward + distance_penalty + safe_distance_bonus - 0.2*accel_penalty - 2 * abs(jerk)

        self.idx += 1
        terminated = (self.idx >= self.n_steps)
        truncated = False

        obs = np.array([self.current_speed, ref_speed, self.lead_speed, relative_distance], dtype=np.float32)
        info = {
            "speed_error": speed_error,
            "speed_difference": speed_diff,
            "relative_distance": relative_distance,
            "acceleration": accel,
            "jerk": jerk
        }
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

    DDPG model
    model = DDPG(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=5e-5,
        batch_size=128,
        buffer_size=100000,
        tau=0.005,
        gamma=0.95,
        action_noise=action_noise,  # Exploration noise
        learning_starts=10000,  # Delay learning until buffer has some data
        train_freq=1,  # Train every step
        gradient_steps=1,  # One gradient step per training
        device=device
    )

    #  TD3 model
    # model = TD3(
    #     policy="MlpPolicy",
    #     env=train_env,
    #     verbose=1,
    #     policy_kwargs=policy_kwargs,
    #     learning_rate=3e-5,
    #     batch_size=256,
    #     buffer_size=200000,
    #     tau=0.003,
    #     gamma=0.985,
    #     action_noise=action_noise,
    #     learning_starts=10000,  # Delay learning until buffer has data
    #     train_freq=1,  # Train every step
    #     gradient_steps=1,  # One gradient step per training
    #     policy_delay=2,  # TD3-specific: Delay policy updates (default 2)
    #     target_policy_noise=0.2,  # TD3-specific: Noise added to target policy
    #     target_noise_clip=0.5,  # TD3-specific: Clip target policy noise
    #     device=device
    # )
    # model = SAC(
    #     policy="MlpPolicy",
    #     env=train_env,
    #     policy_kwargs=dict(net_arch=[256, 256]),  # Keep as is
    #     learning_rate=3e-4,
    #     batch_size=256,
    #     buffer_size=200000,
    #     tau=0.003,
    #     gamma=0.99,
    #     ent_coef='auto',
    #     device=device
    # )
    model.set_logger(logger)

    total_timesteps = 150_000
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
    save_path = os.path.join(log_dir, f"ddpg_speed_follow_chunk{chunk_size}")
    model.save(save_path)
    print(f"[INFO] Model saved to: {save_path}.zip")

    # ------------------------------------------------------------------------
    # 5E) Test the model on the FULL 1200-step dataset in one go
    # ------------------------------------------------------------------------
    test_env = TestEnv(full_speed_data, delta_t=1.0)

    obs, _ = test_env.reset()
    predicted_speeds = []
    ref_speeds = []
    lead_speeds=[]
    distances = []
    rewards = []
    errors = []
    jerks = []
    speed_differences = []

    for _ in range(DATA_LEN):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        predicted_speeds.append(obs[0])  # current_speed
        ref_speeds.append(obs[1])  # reference_speed
        lead_speeds.append(obs[2])  # lead_speed
        distances.append(obs[3])  # relative_distance
        rewards.append(reward)
        errors.append(info["speed_error"])
        jerks.append(info["jerk"])
        speed_differences.append(info["speed_difference"])
        if terminated or truncated:
            break

    # Quantitative Metrics
    avg_test_reward = np.mean(rewards)
    mae_ref = np.mean(np.abs(np.array(predicted_speeds) - np.array(ref_speeds)))
    mse_ref = np.mean((np.array(predicted_speeds) - np.array(ref_speeds)) ** 2)
    avg_speed_diff = np.mean(speed_differences)
    avg_jerk = np.mean(np.abs(jerks))
    avg_distance = np.mean(distances)
    distance_violations = np.sum((np.array(distances) < 5.0) | (np.array(distances) > 30.0))

    # Compute mean and variance of jerk for comfort
    jerks_array = np.array(jerks)
    mean_jerk = np.mean(jerks_array)  # Mean of raw jerk (not absolute)
    variance_jerk = np.var(jerks_array)  # Variance of raw jerk

    print(f"[TEST] Average Reward over 1200-step dataset: {avg_test_reward:.3f}")
    print(f"[TEST] Mean Absolute Error (MAE) vs. Ref Speed: {mae_ref:.3f}")
    print(f"[TEST] Mean Squared Error (MSE) vs. Ref Speed: {mse_ref:.3f}")
    print(f"[TEST] Average Speed Difference (Ego vs. Lead): {avg_speed_diff:.3f}")
    print(f"[TEST] Average Absolute Jerk: {avg_jerk:.3f}")
    print(f"[TEST] Average Distance to Lead Vehicle: {avg_distance:.3f}")
    print(f"[TEST] Number of Distance Violations (<5m or >30m): {distance_violations}")
    print(f"[TEST] Mean Jerk (m/s³): {mean_jerk:.3f}")
    print(f"[TEST] Variance of Jerk (m/s³)²: {variance_jerk:.3f}")

    # Visualizations

    # Smooth jerk and speed difference using moving averages
    window_size_jerk = 50
    window_size_speed_diff = 20
    jerks_array = np.array(jerks)
    speed_diff_array = np.array(speed_differences)
    smoothed_jerks = np.convolve(jerks_array, np.ones(window_size_jerk) / window_size_jerk, mode='valid')
    smoothed_speed_diff = np.convolve(speed_diff_array, np.ones(window_size_speed_diff) / window_size_speed_diff,
                                      mode='valid')

    # Compute mean jerk over a sliding window (e.g., every 50 timesteps)
    window_avg_jerk = 50
    avg_jerks = []
    avg_timesteps = []
    for i in range(0, len(jerks) - window_avg_jerk + 1, window_avg_jerk):
        window = jerks[i:i + window_avg_jerk]
        avg_jerks.append(np.mean(window))
        avg_timesteps.append(i + window_avg_jerk // 2)  # Center of the window

    # Adjust timesteps for smoothed data
    smoothed_length_jerk = len(smoothed_jerks)
    smoothed_length_speed_diff = len(smoothed_speed_diff)
    smoothed_timesteps_jerk = np.linspace(window_size_jerk // 2, len(jerks) - window_size_jerk // 2,
                                          smoothed_length_jerk)
    smoothed_timesteps_speed_diff = np.linspace(window_size_speed_diff // 2,
                                                len(speed_differences) - window_size_speed_diff // 2,
                                                smoothed_length_speed_diff)

    # Compute cumulative reward
    cumulative_rewards = np.cumsum(rewards)

    # Generate plots (back to 5 subplots, replacing normalized jerk with mean jerk)
    plt.figure(figsize=(12, 12))

    # Plot 1: Speed comparison
    plt.subplot(4, 1, 1)
    plt.plot(ref_speeds, label="Reference Speed", linestyle="--", color="blue")
    plt.plot(lead_speeds, label="Lead Vehicle Speed", linestyle="-.", color="orange")
    plt.plot(predicted_speeds, label="Ego Vehicle Speed", linestyle="-", color="green")
    plt.xlabel("Timestep")
    plt.ylabel("Speed (m/s)")
    plt.title(f"Speed Comparison (chunk_size={chunk_size})")
    plt.legend()

    # Plot 2: Relative distance
    plt.subplot(4, 1, 2)
    plt.plot(distances, label="Relative Distance", color="green")
    plt.axhline(y=5, color='r', linestyle='--', label="Min Distance (5m)")
    plt.axhline(y=30, color='b', linestyle='--', label="Max Distance (30m)")
    plt.xlabel("Timestep")
    plt.ylabel("Distance (m)")
    plt.title("Relative Distance to Lead Vehicle")
    plt.legend()

    # Plot 3: Smoothed Jerk with Mean Jerk Over Time
    plt.subplot(4, 1, 3)
    plt.plot(smoothed_timesteps_jerk, smoothed_jerks, label="Smoothed Jerk", color="purple")
    plt.step(avg_timesteps, avg_jerks, label=f"Mean Jerk (Window={window_avg_jerk})", color="black", where='mid')
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel("Timestep")
    plt.ylabel("Jerk (m/s³)")
    plt.title(f"Smoothed and Mean Jerk Over Time (Window Size={window_size_jerk})")
    plt.legend()

    # Plot 4: Smoothed Speed Difference (Ego vs. Lead)
    plt.subplot(4, 1, 4)
    plt.plot(smoothed_timesteps_speed_diff, smoothed_speed_diff, label="Smoothed Speed Difference", color="red")
    plt.xlabel("Timestep")
    plt.ylabel("Speed Diff (m/s)")
    plt.title(f"Smoothed Speed Difference (Ego vs. Lead, Window Size={window_size_speed_diff})")
    plt.legend()


    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "part 2/ddpg_batch.jpeg"))
    plt.show()

if __name__ == "__main__":
    main()
