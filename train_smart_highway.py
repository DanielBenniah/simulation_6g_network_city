#!/usr/bin/env python3
"""
Smart Highway Training Script
============================

Train an agent to control vehicles in the Smart Highway environment with 6G communication.
This script will create a model that learns to navigate efficiently while coordinating
with other vehicles through 6G V2V/V2I communication.
"""

import numpy as np
import sys
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import time

# Import from local environments folder
sys.path.insert(0, os.path.dirname(__file__))
from environments.smart_highway_env import SmartHighwayEnv

class SmartHighwayGymEnv(gym.Env):
    """Gymnasium wrapper for SmartHighwayEnv."""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.env = SmartHighwayEnv(**kwargs)
        
        # Action space: 0=maintain, 1=accelerate, 2=brake
        self.action_space = spaces.Discrete(3)
        
        # Observation space: agent state + nearby vehicles + destination
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(20,), dtype=np.float32
        )
        
        self.episode_rewards = []
        self.episode_steps = []
        self.collision_stats = []
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
        
    def render(self, mode='human'):
        pass  # Rendering handled by visualizer
        
    def close(self):
        pass

def create_learning_environment():
    """Create environment optimized for learning."""
    env_kwargs = {
        'grid_size': (10, 10),
        'max_vehicles': 16,   # Restored to original complexity
        'spawn_rate': 0.3,    # Restored to original spawn rate
        'debug': False
    }
    
    def _init():
        env = SmartHighwayGymEnv(**env_kwargs)
        env = Monitor(env)
        return env
    
    return _init

def train_smart_highway_agent():
    """Train the smart highway agent using PPO."""
    print("🚗" * 25)
    print("🎓 SMART HIGHWAY AGENT TRAINING 🎓")
    print("🚗" * 25)
    print("🎯 Training Objectives:")
    print("  • Learn efficient navigation in dedicated lanes")
    print("  • Coordinate with 6G communication system")
    print("  • Minimize collisions through intelligent actions")
    print("  • Optimize journey times and traffic flow")
    print("🚗" * 25)
    
    # Create training environment
    print("\n📚 Setting up training environment...")
    env = make_vec_env(create_learning_environment(), n_envs=4)
    
    # Create evaluation environment
    eval_env = SmartHighwayGymEnv(
        grid_size=(10, 10),
        max_vehicles=16,    # Restored to original complexity
        spawn_rate=0.3,     # Restored to original spawn rate
        debug=False
    )
    eval_env = Monitor(eval_env)
    
    # Training parameters (properly scaled for real learning)
    total_timesteps = 100000  # Full training for proper learning
    eval_freq = 5000          # Standard evaluation frequency
    target_reward = 50        # Higher target based on actual reward scale observed
    
    print(f"📊 Training Configuration:")
    print(f"  Total Timesteps: {total_timesteps:,}")
    print(f"  Evaluation Frequency: {eval_freq:,}")
    print(f"  Target Reward: {target_reward}")
    print(f"  Parallel Environments: 4")
    
    # Create PPO model
    print("\n🧠 Creating PPO Agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./logs/smart_highway_training/"
    )
    
    # Setup callbacks
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=target_reward, 
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./trained_models/",
        log_path="./logs/smart_highway_eval/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
        callback_on_new_best=stop_callback
    )
    
    # Train the model
    print("\n🎓 Starting Training...")
    print("📈 Monitor progress in TensorBoard: tensorboard --logdir ./logs/smart_highway_training/")
    
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            tb_log_name="smart_highway_ppo",
            progress_bar=True
        )
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time:.1f} seconds!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user after {(time.time() - start_time):.1f} seconds")
    
    # Save the final model
    model_path = "trained_models/ppo_smart_highway"
    model.save(model_path)
    print(f"💾 Model saved to: {model_path}")
    
    # Test the trained model
    print("\n🧪 Testing trained model...")
    test_trained_model(model, eval_env)
    
    return model

def test_trained_model(model, env, episodes=5):
    """Test the trained model performance."""
    print(f"\n🎯 Testing model for {episodes} episodes...")
    
    total_rewards = []
    total_steps = []
    collision_counts = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        episode_collisions = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            steps += 1
            
            # Count actual collisions (not 6G preventions)
            episode_collisions += len(info.get('actual_collisions', []))
            
            if terminated or truncated or steps >= 500:
                break
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        collision_counts.append(episode_collisions)
        
        print(f"  Episode {episode + 1}: Reward={episode_reward:.2f}, Steps={steps}, Collisions={episode_collisions}")
    
    # Summary statistics
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    avg_collisions = np.mean(collision_counts)
    
    print(f"\n📊 TRAINED MODEL PERFORMANCE:")
    print(f"  Average Reward: {avg_reward:.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Average Episode Length: {avg_steps:.1f} ± {np.std(total_steps):.1f}")
    print(f"  Average Collisions per Episode: {avg_collisions:.2f} ± {np.std(collision_counts):.2f}")
    
    if avg_collisions < 1.0:
        print("  ✅ Excellent collision avoidance!")
    elif avg_collisions < 3.0:
        print("  ✅ Good collision avoidance")
    else:
        print("  ⚠️  High collision rate - more training needed")

def compare_random_vs_trained():
    """Compare random agent vs trained agent performance."""
    print("\n🏁 RANDOM vs TRAINED COMPARISON")
    print("=" * 50)
    
    # Test environment
    env = SmartHighwayGymEnv(
        grid_size=(10, 10),
        max_vehicles=16,
        spawn_rate=0.3,
        debug=False
    )
    env = Monitor(env)
    
    # Test random agent
    print("\n🎲 Testing Random Agent...")
    random_rewards = []
    random_collisions = []
    
    for episode in range(3):
        obs, info = env.reset()
        episode_reward = 0
        episode_collisions = 0
        steps = 0
        
        while steps < 200:
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_collisions += len(info.get('actual_collisions', []))
            steps += 1
            
            if terminated or truncated:
                break
        
        random_rewards.append(episode_reward)
        random_collisions.append(episode_collisions)
    
    # Test trained agent
    print("\n🧠 Testing Trained Agent...")
    try:
        model = PPO.load("trained_models/ppo_smart_highway")
        trained_rewards = []
        trained_collisions = []
        
        for episode in range(3):
            obs, info = env.reset()
            episode_reward = 0
            episode_collisions = 0
            steps = 0
            
            while steps < 200:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_collisions += len(info.get('actual_collisions', []))
                steps += 1
                
                if terminated or truncated:
                    break
            
            trained_rewards.append(episode_reward)
            trained_collisions.append(episode_collisions)
        
        # Comparison
        print(f"\n📈 PERFORMANCE COMPARISON:")
        print(f"Random Agent:")
        print(f"  Average Reward: {np.mean(random_rewards):.2f}")
        print(f"  Average Collisions: {np.mean(random_collisions):.2f}")
        print(f"Trained Agent:")
        print(f"  Average Reward: {np.mean(trained_rewards):.2f}")
        print(f"  Average Collisions: {np.mean(trained_collisions):.2f}")
        print(f"Improvement:")
        print(f"  Reward: {np.mean(trained_rewards) - np.mean(random_rewards):.2f} (+{((np.mean(trained_rewards) - np.mean(random_rewards)) / abs(np.mean(random_rewards)) * 100):.1f}%)")
        print(f"  Collision Reduction: {np.mean(random_collisions) - np.mean(trained_collisions):.2f}")
        
    except FileNotFoundError:
        print("❌ No trained model found. Run training first!")

def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Smart Highway Agent')
    parser.add_argument('--train', action='store_true',
                       help='Train a new model')
    parser.add_argument('--test', action='store_true', 
                       help='Test existing model')
    parser.add_argument('--compare', action='store_true',
                       help='Compare random vs trained agent')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Total training timesteps')
    
    args = parser.parse_args()
    
    if args.train:
        model = train_smart_highway_agent()
    
    if args.test:
        try:
            model = PPO.load("trained_models/ppo_smart_highway")
            env = SmartHighwayGymEnv(grid_size=(10, 10), max_vehicles=16, spawn_rate=0.3)
            env = Monitor(env)
            test_trained_model(model, env)
        except FileNotFoundError:
            print("❌ No trained model found. Run training first with --train")
    
    if args.compare:
        compare_random_vs_trained()
    
    if not any([args.train, args.test, args.compare]):
        print("🎓 Smart Highway Training Options:")
        print("  --train     : Train a new model")
        print("  --test      : Test existing model")
        print("  --compare   : Compare random vs trained")
        print("\nExample: python train_smart_highway.py --train")

if __name__ == "__main__":
    main() 