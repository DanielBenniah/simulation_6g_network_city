import gymnasium as gym
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
# Set matplotlib backend for compatibility
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend by default
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
    try:
        if not os.path.exists(metrics_log):
            print(f"Warning: Log file {metrics_log} does not exist. Skipping plot.")
            return
        
        steps, rewards, collisions = [], [], []
        with open(metrics_log, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 3:
                    continue
                try:
                    step, avg_reward, avg_collisions = int(parts[0]), float(parts[1]), float(parts[2])
                    steps.append(step)
                    rewards.append(avg_reward)
                    collisions.append(avg_collisions)
                except (ValueError, IndexError):
                    continue
        
        if len(steps) == 0:
            print(f"Warning: No valid data found in {metrics_log}. Skipping plot.")
            return
        
        plt.figure(figsize=(10,5))
        plt.plot(steps, rewards, label='Avg Reward per Episode', marker='o')
        plt.plot(steps, collisions, label='Avg Collisions per Episode', marker='x')
        plt.xlabel('Training Step')
        plt.ylabel('Value')
        plt.title('Training Progress: Reward and Collisions')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Save plot instead of showing (for compatibility)
        plt.savefig('training_metrics.png', dpi=150, bbox_inches='tight')
        print("Training metrics plot saved as 'training_metrics.png'")
        plt.close()  # Close to free memory
        
    except Exception as e:
        print(f"Error plotting metrics: {e}")

def main():
    try:
        env = make_env()
        print("Environment created successfully.")
        
        # Check environment
        try:
            check_env(env, warn=True)
            print("Environment validation passed.")
        except Exception as e:
            print(f"Environment validation failed: {e}")
            return

        # Define PPO model
        model = PPO('MlpPolicy', env, verbose=1)
        print("PPO model created successfully.")

        # Training loop
        total_timesteps = 100_000
        eval_freq = 5_000
        n_eval_episodes = 5
        log_file = "training_log.txt"
        
        # Clear existing log file
        try:
            if os.path.exists(log_file):
                os.remove(log_file)
        except Exception as e:
            print(f"Warning: Could not remove existing log file: {e}")

        # TensorBoard writer
        writer = None
        if TENSORBOARD_AVAILABLE:
            try:
                writer = SummaryWriter()
                print("TensorBoard logging enabled.")
            except Exception as e:
                print(f"Warning: TensorBoard initialization failed: {e}")

        print(f"Starting training for {total_timesteps} timesteps...")
        
        for step in range(0, total_timesteps, eval_freq):
            try:
                # Training
                model.learn(total_timesteps=eval_freq, reset_num_timesteps=False)
                
                # Evaluation
                rewards, collisions, travel_times, throughputs, waiting_times, near_misses = [], [], [], [], [], []
                
                for ep in range(n_eval_episodes):
                    try:
                        obs, info = env.reset()
                        terminated = False
                        truncated = False
                        ep_reward = 0
                        ep_collisions = 0
                        ep_travel_time = 0
                        ep_throughput = 0
                        ep_waiting_time = 0
                        ep_near_miss = 0
                        
                        # Initialize tracking arrays safely
                        vehicle_steps = [0] * max(env.num_vehicles, 1)
                        vehicle_waiting = [0] * max(env.num_vehicles, 1)
                        vehicle_reached = [0] * max(env.num_vehicles, 1)
                        
                        step_count = 0
                        max_steps = 1000  # Prevent infinite episodes
                        
                        while not (terminated or truncated) and step_count < max_steps:
                            action, _ = model.predict(obs, deterministic=True)
                            obs, reward, terminated, truncated, info = env.step(action)
                            
                            ep_reward += reward
                            ep_collisions += len(info.get('collisions', []))
                            
                            # Track per-vehicle metrics safely
                            for i in range(min(env.num_vehicles, len(vehicle_steps))):
                                if i < len(env.vehicles) and env.vehicles[i, 6] == 1:  # Active vehicle
                                    vehicle_steps[i] += 1
                                    vx, vy = env.vehicles[i, 2], env.vehicles[i, 3]
                                    if vx == 0 and vy == 0:
                                        vehicle_waiting[i] += 1
                            
                            # Throughput: count vehicles reaching destination
                            if reward >= 1:
                                ep_throughput += 1
                            
                            # Near-miss calculation with bounds checking
                            try:
                                positions = [(int(env.vehicles[i, 0]), int(env.vehicles[i, 1])) 
                                           for i in range(env.num_vehicles) 
                                           if i < len(env.vehicles) and env.vehicles[i, 6] == 1]
                                for i in range(len(positions)):
                                    for j in range(i+1, len(positions)):
                                        if abs(positions[i][0] - positions[j][0]) + abs(positions[i][1] - positions[j][1]) == 1:
                                            ep_near_miss += 1
                            except (IndexError, ValueError):
                                pass  # Skip near-miss calculation if error occurs
                            
                            step_count += 1
                        
                        # Calculate metrics safely
                        active_vehicles = [i for i in range(min(env.num_vehicles, len(vehicle_steps))) 
                                         if i < len(env.vehicles) and env.vehicles[i, 6] == 1]
                        
                        if active_vehicles:
                            avg_travel_time = np.mean([vehicle_steps[i] for i in active_vehicles])
                        else:
                            avg_travel_time = 0
                        
                        avg_waiting_time = np.mean(vehicle_waiting) if vehicle_waiting else 0
                        
                        rewards.append(ep_reward)
                        collisions.append(ep_collisions)
                        travel_times.append(avg_travel_time)
                        throughputs.append(ep_throughput)
                        waiting_times.append(avg_waiting_time)
                        near_misses.append(ep_near_miss)
                        
                    except Exception as e:
                        print(f"Warning: Episode {ep} failed: {e}")
                        # Add default values to prevent empty lists
                        rewards.append(0)
                        collisions.append(0)
                        travel_times.append(0)
                        throughputs.append(0)
                        waiting_times.append(0)
                        near_misses.append(0)
                
                # Calculate averages safely
                avg_reward = np.mean(rewards) if rewards else 0
                avg_collisions = np.mean(collisions) if collisions else 0
                avg_travel_time = np.mean(travel_times) if travel_times else 0
                avg_throughput = np.mean(throughputs) if throughputs else 0
                avg_waiting_time = np.mean(waiting_times) if waiting_times else 0
                avg_near_miss = np.mean(near_misses) if near_misses else 0
                
                print(f"Step {step+eval_freq}: Avg Reward {avg_reward:.2f}, Avg Collisions {avg_collisions:.2f}, "
                      f"Avg Travel Time {avg_travel_time:.2f}, Avg Throughput {avg_throughput:.2f}, "
                      f"Avg Waiting {avg_waiting_time:.2f}, Avg Near-Miss {avg_near_miss:.2f}")
                
                # Log to file
                try:
                    with open(log_file, "a") as f:
                        f.write(f"{step+eval_freq},{avg_reward},{avg_collisions},{avg_travel_time},{avg_throughput},{avg_waiting_time},{avg_near_miss}\n")
                except Exception as e:
                    print(f"Warning: Could not write to log file: {e}")
                
                # TensorBoard logging
                if writer:
                    try:
                        writer.add_scalar("Reward/Avg", avg_reward, step+eval_freq)
                        writer.add_scalar("Collisions/Avg", avg_collisions, step+eval_freq)
                        writer.add_scalar("TravelTime/Avg", avg_travel_time, step+eval_freq)
                        writer.add_scalar("Throughput/Avg", avg_throughput, step+eval_freq)
                        writer.add_scalar("WaitingTime/Avg", avg_waiting_time, step+eval_freq)
                        writer.add_scalar("NearMiss/Avg", avg_near_miss, step+eval_freq)
                    except Exception as e:
                        print(f"Warning: TensorBoard logging failed: {e}")
                        
            except Exception as e:
                print(f"Error in training step {step}: {e}")
                continue
        
        if writer:
            try:
                writer.close()
            except Exception as e:
                print(f"Warning: Could not close TensorBoard writer: {e}")

        # Save the trained model
        try:
            model.save("ppo_city_traffic")
            print("Training complete. Model saved as 'ppo_city_traffic'.")
        except Exception as e:
            print(f"Error saving model: {e}")

        # Final evaluation
        try:
            def eval_policy_gymnasium(model, env, n_eval_episodes=10):
                rewards = []
                for _ in range(n_eval_episodes):
                    try:
                        obs, info = env.reset()
                        terminated = False
                        truncated = False
                        ep_reward = 0
                        step_count = 0
                        max_steps = 1000
                        
                        while not (terminated or truncated) and step_count < max_steps:
                            action, _ = model.predict(obs, deterministic=True)
                            obs, reward, terminated, truncated, info = env.step(action)
                            ep_reward += reward
                            step_count += 1
                        rewards.append(ep_reward)
                    except Exception as e:
                        print(f"Warning: Evaluation episode failed: {e}")
                        rewards.append(0)
                
                return np.mean(rewards) if rewards else 0, np.std(rewards) if rewards else 0

            mean_reward, std_reward = eval_policy_gymnasium(model, env, n_eval_episodes=10)
            print(f"Final evaluation: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
        except Exception as e:
            print(f"Error in final evaluation: {e}")

        # Plot metrics after training
        plot_metrics(log_file)
        
    except Exception as e:
        print(f"Fatal error in main: {e}")
        return

if __name__ == "__main__":
    main()

# --- Notes for Multi-Agent Training ---
# - For multi-agent mode (multi_agent=True), you can use parameter sharing:
#   - Wrap the environment so that all agents use the same policy (e.g., flatten obs/actions).
#   - Or, train one agent at a time (others scripted), rotating through agents.
#   - For true multi-agent RL, consider using PettingZoo + SB3 wrappers or MARL libraries.
# - To extend: replace 'multi_agent=False' with 'multi_agent=True',
#   and adapt the training loop to handle dict obs/rewards (e.g., sum or average rewards). 