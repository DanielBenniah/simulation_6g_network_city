import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
import os
import sys
import time

# Import the environment
from city_traffic_env import CityTrafficEnv

# --- Training script for CityTrafficEnv using Stable Baselines3 PPO ---
# This script uses single-agent mode (multi_agent=False) for simplicity.
# For multi-agent, see comments below.

def make_env():
    # Use single-agent mode: agent 0 is learning, others are scripted
    env = CityTrafficEnv(grid_size=(10, 10), max_vehicles=5, multi_agent=False)
    return env

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

    for step in range(0, total_timesteps, eval_freq):
        model.learn(total_timesteps=eval_freq, reset_num_timesteps=False)
        # Evaluate
        rewards, collisions = [], []
        for _ in range(n_eval_episodes):
            obs = env.reset()
            done = False
            ep_reward = 0
            ep_collisions = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                ep_reward += reward
                if 'collisions' in info and info['collisions']:
                    ep_collisions += len(info['collisions'])
            rewards.append(ep_reward)
            collisions.append(ep_collisions)
        avg_reward = np.mean(rewards)
        avg_collisions = np.mean(collisions)
        print(f"Step {step+eval_freq}: Avg Reward {avg_reward:.2f}, Avg Collisions {avg_collisions:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step+eval_freq},{avg_reward},{avg_collisions}\n")

    # Save the trained model
    model.save("ppo_city_traffic")
    print("Training complete. Model saved as 'ppo_city_traffic'.")

    # Final evaluation
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Final evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == "__main__":
    main()

# --- Notes for Multi-Agent Training ---
# - For multi-agent mode (multi_agent=True), you can use parameter sharing:
#   - Wrap the environment so that all agents use the same policy (e.g., flatten obs/actions).
#   - Or, train one agent at a time (others scripted), rotating through agents.
#   - For true multi-agent RL, consider using PettingZoo + SB3 wrappers or MARL libraries.
# - To extend: replace 'multi_agent=False' with 'multi_agent=True',
#   and adapt the training loop to handle dict obs/rewards (e.g., sum or average rewards). 