import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import os
import sys
import time
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

import matplotlib.pyplot as plt
from city_traffic_env import CityTrafficEnv

# --- Training script for CityTrafficEnv using Stable Baselines3 PPO ---
# This script uses single-agent mode (multi_agent=False) for simplicity.
# For multi-agent, see comments below.

def make_env():
    # Use single-agent mode: agent 0 is learning, others are scripted
    env = CityTrafficEnv(grid_size=(10, 10), max_vehicles=5, multi_agent=False)
    return env

def plot_metrics(metrics_log):
    """
    Plot average reward and number of collisions per episode over time.
    metrics_log: path to the log file (CSV with columns: step, avg_reward, avg_collisions, ...)
    """
    steps, rewards, collisions = [], [], []
    with open(metrics_log, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 3:
                continue
            step, avg_reward, avg_collisions = int(parts[0]), float(parts[1]), float(parts[2])
            steps.append(step)
            rewards.append(avg_reward)
            collisions.append(avg_collisions)
    plt.figure(figsize=(10,5))
    plt.plot(steps, rewards, label='Avg Reward per Episode', marker='o')
    plt.plot(steps, collisions, label='Avg Collisions per Episode', marker='x')
    plt.xlabel('Training Step')
    plt.ylabel('Value')
    plt.title('Training Progress: Reward and Collisions')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    env = make_env()
    check_env(env, warn=True)

    # Define PPO model
    model = PPO('MlpPolicy', env, verbose=1)

    # Training loop
    total_timesteps = 100_000
    eval_freq = 5_000
    n_eval_episodes = 5
    log_file = "training_log.txt"
    if os.path.exists(log_file):
        os.remove(log_file)

    # TensorBoard writer
    writer = SummaryWriter() if TENSORBOARD_AVAILABLE else None

    for step in range(0, total_timesteps, eval_freq):
        model.learn(total_timesteps=eval_freq, reset_num_timesteps=False)
        # Evaluate
        rewards, collisions, travel_times, throughputs, waiting_times, near_misses = [], [], [], [], [], []
        for _ in range(n_eval_episodes):
            obs = env.reset()
            done = False
            ep_reward = 0
            ep_collisions = 0
            ep_travel_time = 0
            ep_throughput = 0
            ep_waiting_time = 0
            ep_near_miss = 0
            # For travel time and waiting time tracking
            vehicle_steps = [0 for _ in range(env.num_vehicles)]
            vehicle_waiting = [0 for _ in range(env.num_vehicles)]
            vehicle_reached = [0 for _ in range(env.num_vehicles)]
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                ep_reward += reward
                ep_collisions += len(info['collisions']) if 'collisions' in info else 0
                # Track per-vehicle travel time and waiting
                for i in range(env.num_vehicles):
                    vehicle_steps[i] += 1
                    # If vehicle is stopped (vx==0 and vy==0), count as waiting
                    vx, vy = env.vehicles[i, 2], env.vehicles[i, 3]
                    if vx == 0 and vy == 0:
                        vehicle_waiting[i] += 1
                # Throughput: count vehicles reaching destination (reward +1)
                if reward >= 1:
                    ep_throughput += 1
                # Near-miss: if two vehicles are within 1 cell but not colliding
                positions = [(int(env.vehicles[i, 0]), int(env.vehicles[i, 1])) for i in range(env.num_vehicles) if env.vehicles[i, 6] == 1]
                for i in range(len(positions)):
                    for j in range(i+1, len(positions)):
                        if abs(positions[i][0] - positions[j][0]) + abs(positions[i][1] - positions[j][1]) == 1:
                            ep_near_miss += 1
            # Average travel time for vehicles still active
            avg_travel_time = np.mean([vehicle_steps[i] for i in range(env.num_vehicles) if env.vehicles[i, 6] == 1]) if any(env.vehicles[:,6]==1) else 0
            avg_waiting_time = np.mean(vehicle_waiting)
            rewards.append(ep_reward)
            collisions.append(ep_collisions)
            travel_times.append(avg_travel_time)
            throughputs.append(ep_throughput)
            waiting_times.append(avg_waiting_time)
            near_misses.append(ep_near_miss)
        avg_reward = np.mean(rewards)
        avg_collisions = np.mean(collisions)
        avg_travel_time = np.mean(travel_times)
        avg_throughput = np.mean(throughputs)
        avg_waiting_time = np.mean(waiting_times)
        avg_near_miss = np.mean(near_misses)
        print(f"Step {step+eval_freq}: Avg Reward {avg_reward:.2f}, Avg Collisions {avg_collisions:.2f}, "
              f"Avg Travel Time {avg_travel_time:.2f}, Avg Throughput {avg_throughput:.2f}, "
              f"Avg Waiting {avg_waiting_time:.2f}, Avg Near-Miss {avg_near_miss:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step+eval_freq},{avg_reward},{avg_collisions},{avg_travel_time},{avg_throughput},{avg_waiting_time},{avg_near_miss}\n")
        if writer:
            writer.add_scalar("Reward/Avg", avg_reward, step+eval_freq)
            writer.add_scalar("Collisions/Avg", avg_collisions, step+eval_freq)
            writer.add_scalar("TravelTime/Avg", avg_travel_time, step+eval_freq)
            writer.add_scalar("Throughput/Avg", avg_throughput, step+eval_freq)
            writer.add_scalar("WaitingTime/Avg", avg_waiting_time, step+eval_freq)
            writer.add_scalar("NearMiss/Avg", avg_near_miss, step+eval_freq)
    if writer:
        writer.close()

    # Save the trained model
    model.save("ppo_city_traffic")
    print("Training complete. Model saved as 'ppo_city_traffic'.")

    # Final evaluation
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Final evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

    # Plot metrics after training
    plot_metrics(log_file)

if __name__ == "__main__":
    main()

# --- Notes for Multi-Agent Training ---
# - For multi-agent mode (multi_agent=True), you can use parameter sharing:
#   - Wrap the environment so that all agents use the same policy (e.g., flatten obs/actions).
#   - Or, train one agent at a time (others scripted), rotating through agents.
#   - For true multi-agent RL, consider using PettingZoo + SB3 wrappers or MARL libraries.
# - To extend: replace 'multi_agent=False' with 'multi_agent=True',
#   and adapt the training loop to handle dict obs/rewards (e.g., sum or average rewards). 