#!/usr/bin/env python3
"""
Training Results Analysis Tool
=============================

Analyze your trained smart highway models and extract detailed statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
import sys
import os

# Import your environment
sys.path.insert(0, os.path.dirname(__file__))
from train_smart_highway import SmartHighwayGymEnv

def analyze_tensorboard_logs(log_dir="./logs/smart_highway_training/"):
    """Analyze TensorBoard logs to extract training progression."""
    import glob
    
    print("📈 TENSORBOARD LOG ANALYSIS")
    print("=" * 50)
    
    # Find all training runs
    runs = glob.glob(os.path.join(log_dir, "smart_highway_ppo_*"))
    runs.sort()
    
    print(f"Found {len(runs)} training runs:")
    for i, run in enumerate(runs):
        run_name = os.path.basename(run)
        print(f"  {i+1}. {run_name}")
    
    return runs

def test_trained_model_detailed(model_path="trained_models/ppo_smart_highway", episodes=10):
    """Test trained model and collect detailed statistics."""
    print("\n🧪 DETAILED MODEL TESTING")
    print("=" * 50)
    
    try:
        # Load model
        model = PPO.load(model_path)
        print(f"✅ Model loaded from: {model_path}")
        
        # Create test environment
        env = SmartHighwayGymEnv(
            grid_size=(10, 10),
            max_vehicles=16,
            spawn_rate=0.3,
            debug=False
        )
        env = Monitor(env)
        
        # Collect statistics
        episode_rewards = []
        episode_lengths = []
        collision_counts = []
        journey_stats_per_episode = []
        
        print(f"\n🎯 Testing model for {episodes} episodes...")
        
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
                
                # Count collisions
                episode_collisions += len(info.get('actual_collisions', []))
                
                if terminated or truncated or steps >= 500:
                    break
            
            # Collect episode statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            collision_counts.append(episode_collisions)
            
            # Get journey statistics from environment
            stats = env.env.env.get_statistics()  # env.env.env because of Gym wrapper + Monitor wrapper
            journey_stats_per_episode.append(stats.copy())
            
            print(f"  Episode {episode + 1:2d}: Reward={episode_reward:6.2f}, "
                  f"Steps={steps:3d}, Collisions={episode_collisions}, "
                  f"Completed={stats['total_completed']:2d}/{stats['total_spawned']:2d}")
        
        # Calculate summary statistics
        print(f"\n📊 TRAINED MODEL PERFORMANCE SUMMARY:")
        print(f"  Episodes tested: {episodes}")
        print(f"  Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"  Reward range: {np.min(episode_rewards):.2f} - {np.max(episode_rewards):.2f}")
        print(f"  Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"  Average collisions per episode: {np.mean(collision_counts):.2f} ± {np.std(collision_counts):.2f}")
        
        # Journey time analysis
        all_journey_times = []
        total_spawned = 0
        total_completed = 0
        
        for stats in journey_stats_per_episode:
            total_spawned += stats['total_spawned']
            total_completed += stats['total_completed']
            if stats['avg_journey_time'] > 0:
                all_journey_times.append(stats['avg_journey_time'])
        
        print(f"\n🚗 JOURNEY TIME ANALYSIS:")
        print(f"  Total vehicles spawned: {total_spawned}")
        print(f"  Total journeys completed: {total_completed}")
        print(f"  Overall completion rate: {(total_completed/max(total_spawned,1)*100):.1f}%")
        
        if all_journey_times:
            print(f"  Average journey time: {np.mean(all_journey_times):.1f} ± {np.std(all_journey_times):.1f} steps")
            print(f"  Journey time range: {np.min(all_journey_times):.1f} - {np.max(all_journey_times):.1f} steps")
            print(f"  Episodes with completed journeys: {len(all_journey_times)}/{episodes}")
        
        # Performance categorization
        if np.mean(collision_counts) < 1.0:
            print(f"  ✅ Excellent collision avoidance!")
        elif np.mean(collision_counts) < 3.0:
            print(f"  ✅ Good collision avoidance")
        else:
            print(f"  ⚠️  High collision rate - more training needed")
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'collision_counts': collision_counts,
            'journey_stats': journey_stats_per_episode,
            'summary': {
                'avg_reward': np.mean(episode_rewards),
                'avg_length': np.mean(episode_lengths),
                'avg_collisions': np.mean(collision_counts),
                'total_spawned': total_spawned,
                'total_completed': total_completed,
                'completion_rate': total_completed/max(total_spawned,1)*100,
                'avg_journey_time': np.mean(all_journey_times) if all_journey_times else 0
            }
        }
        
    except FileNotFoundError:
        print("❌ No trained model found. Run training first!")
        return None

def compare_random_vs_trained_detailed():
    """Detailed comparison between random and trained agents."""
    print("\n🏁 DETAILED RANDOM vs TRAINED COMPARISON")
    print("=" * 60)
    
    env = SmartHighwayGymEnv(
        grid_size=(10, 10),
        max_vehicles=16,
        spawn_rate=0.3,
        debug=False
    )
    env = Monitor(env)
    
    def test_agent(agent_type, episodes=5):
        rewards = []
        lengths = []
        collisions = []
        journey_times = []
        completion_rates = []
        
        for episode in range(episodes):
            obs, info = env.reset()
            episode_reward = 0
            steps = 0
            episode_collisions = 0
            
            while steps < 200:
                if agent_type == "random":
                    action = env.action_space.sample()
                else:  # trained
                    action, _ = model.predict(obs, deterministic=True)
                
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_collisions += len(info.get('actual_collisions', []))
                steps += 1
                
                if terminated or truncated:
                    break
            
                         # Collect statistics
             stats = env.env.env.get_statistics()
            rewards.append(episode_reward)
            lengths.append(steps)
            collisions.append(episode_collisions)
            if stats['avg_journey_time'] > 0:
                journey_times.append(stats['avg_journey_time'])
            completion_rates.append(stats['efficiency'])
        
        return {
            'rewards': rewards,
            'lengths': lengths,
            'collisions': collisions,
            'journey_times': journey_times,
            'completion_rates': completion_rates
        }
    
    # Test random agent
    print("🎲 Testing Random Agent...")
    random_results = test_agent("random")
    
    # Test trained agent
    print("🧠 Testing Trained Agent...")
    try:
        model = PPO.load("trained_models/ppo_smart_highway")
        trained_results = test_agent("trained")
        
        # Print comparison
        print(f"\n📈 DETAILED PERFORMANCE COMPARISON:")
        print(f"{'Metric':<20} {'Random':<15} {'Trained':<15} {'Improvement':<15}")
        print("-" * 65)
        
        # Rewards
        random_reward = np.mean(random_results['rewards'])
        trained_reward = np.mean(trained_results['rewards'])
        reward_improvement = trained_reward - random_reward
        print(f"{'Avg Reward':<20} {random_reward:<15.2f} {trained_reward:<15.2f} {reward_improvement:+.2f}")
        
        # Collisions
        random_collisions = np.mean(random_results['collisions'])
        trained_collisions = np.mean(trained_results['collisions'])
        collision_reduction = random_collisions - trained_collisions
        print(f"{'Avg Collisions':<20} {random_collisions:<15.2f} {trained_collisions:<15.2f} {collision_reduction:+.2f}")
        
        # Journey times
        if random_results['journey_times'] and trained_results['journey_times']:
            random_journey = np.mean(random_results['journey_times'])
            trained_journey = np.mean(trained_results['journey_times'])
            journey_improvement = random_journey - trained_journey
            print(f"{'Avg Journey Time':<20} {random_journey:<15.1f} {trained_journey:<15.1f} {journey_improvement:+.1f}")
        
        # Completion rates
        random_completion = np.mean(random_results['completion_rates'])
        trained_completion = np.mean(trained_results['completion_rates'])
        completion_improvement = trained_completion - random_completion
        print(f"{'Completion Rate %':<20} {random_completion:<15.1f} {trained_completion:<15.1f} {completion_improvement:+.1f}")
        
    except FileNotFoundError:
        print("❌ No trained model found for comparison!")

def plot_training_progress():
    """Create plots showing training progress."""
    print("\n📊 CREATING TRAINING PROGRESS PLOTS")
    print("=" * 50)
    
    # This would require parsing TensorBoard logs or training logs
    # For now, provide instructions on using TensorBoard
    print("📈 To see detailed training progress:")
    print("  1. Open terminal and run: python -m tensorboard.main --logdir ./logs/smart_highway_training/")
    print("  2. Open browser to: http://localhost:6006")
    print("  3. Look at these key metrics:")
    print("     - rollout/ep_rew_mean (episode rewards)")
    print("     - rollout/ep_len_mean (episode lengths)")
    print("     - train/loss (training loss)")
    print("     - train/explained_variance (learning progress)")

def main():
    """Main analysis function."""
    print("🎯 SMART HIGHWAY TRAINING RESULTS ANALYZER")
    print("=" * 60)
    
    # Analyze TensorBoard logs
    runs = analyze_tensorboard_logs()
    
    # Test trained model in detail
    results = test_trained_model_detailed(episodes=10)
    
    # Compare with random agent
    compare_random_vs_trained_detailed()
    
    # Show how to view training progress
    plot_training_progress()
    
    print(f"\n✅ Analysis complete! Your model shows:")
    if results:
        summary = results['summary']
        print(f"   - Average reward: {summary['avg_reward']:.1f}")
        print(f"   - Completion rate: {summary['completion_rate']:.1f}%")
        print(f"   - Average journey time: {summary['avg_journey_time']:.1f} steps")
        print(f"   - Collision avoidance: {summary['avg_collisions']:.2f} collisions/episode")

if __name__ == "__main__":
    main() 