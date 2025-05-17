#!/usr/bin/env python3
"""
Training Log Analysis Script
===========================

Analyze training logs and generate performance visualizations for the Smart Highway model.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from pathlib import Path
import seaborn as sns

def analyze_evaluation_logs():
    """Analyze evaluation logs and create visualizations."""
    eval_path = Path("logs/smart_highway_eval")
    
    if not eval_path.exists():
        print("❌ No evaluation logs found. Run training first!")
        return
    
    print("📊 ANALYZING TRAINING LOGS")
    print("=" * 50)
    
    # Load evaluation data
    eval_file = eval_path / "evaluations.npz"
    monitor_file = eval_path / "monitor.csv"
    
    if eval_file.exists():
        print("📈 Loading evaluation results...")
        eval_data = np.load(eval_file)
        rewards = eval_data['results']
        timesteps = eval_data['timesteps']
        
        # Create learning curve
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Learning Curve
        axes[0, 0].plot(timesteps, rewards.mean(axis=1), 'b-', linewidth=2, label='Mean Reward')
        axes[0, 0].fill_between(timesteps, 
                                rewards.mean(axis=1) - rewards.std(axis=1),
                                rewards.mean(axis=1) + rewards.std(axis=1), 
                                alpha=0.3, color='blue')
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('Episode Reward')
        axes[0, 0].set_title('🎯 Learning Progress')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Plot 2: Reward Distribution
        final_rewards = rewards[-1] if len(rewards) > 0 else [0]
        axes[0, 1].hist(final_rewards, bins=10, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_xlabel('Final Episode Rewards')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('🎯 Final Performance Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Print statistics
        print(f"📊 Training Statistics:")
        print(f"   Total Training Steps: {timesteps[-1]:,}")
        print(f"   Initial Average Reward: {rewards[0].mean():.2f} ± {rewards[0].std():.2f}")
        print(f"   Final Average Reward: {rewards[-1].mean():.2f} ± {rewards[-1].std():.2f}")
        print(f"   Improvement: {rewards[-1].mean() - rewards[0].mean():.2f} ({((rewards[-1].mean() - rewards[0].mean()) / abs(rewards[0].mean()) * 100):.1f}%)")
        print(f"   Best Single Episode: {rewards.max():.2f}")
        
    else:
        print("⚠️  No evaluation data found")
    
    # Load monitor data
    if monitor_file.exists():
        print("\n📈 Loading episode monitor data...")
        monitor_data = pd.read_csv(monitor_file)
        
        if len(monitor_data) > 0:
            # Plot 3: Episode Length over Time
            axes[1, 0].plot(monitor_data.index, monitor_data['l'], 'r-', alpha=0.7)
            axes[1, 0].set_xlabel('Episode Number')
            axes[1, 0].set_ylabel('Episode Length (Steps)')
            axes[1, 0].set_title('📏 Episode Length Progress')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Reward vs Episode Length
            axes[1, 1].scatter(monitor_data['l'], monitor_data['r'], alpha=0.6, color='purple')
            axes[1, 1].set_xlabel('Episode Length (Steps)')
            axes[1, 1].set_ylabel('Episode Reward')
            axes[1, 1].set_title('🎯 Reward vs Episode Length')
            axes[1, 1].grid(True, alpha=0.3)
            
            print(f"📊 Episode Statistics:")
            print(f"   Total Episodes: {len(monitor_data)}")
            print(f"   Average Episode Length: {monitor_data['l'].mean():.1f} ± {monitor_data['l'].std():.1f}")
            print(f"   Longest Episode: {monitor_data['l'].max()}")
            print(f"   Shortest Episode: {monitor_data['l'].min()}")
            print(f"   Average Episode Reward: {monitor_data['r'].mean():.2f} ± {monitor_data['r'].std():.2f}")
    else:
        print("⚠️  No monitor data found")
    
    plt.tight_layout()
    plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Analysis plots saved to: training_analysis.png")
    plt.show()

def analyze_tensorboard_logs():
    """Provide instructions for TensorBoard analysis."""
    tb_path = Path("logs/smart_highway_training")
    
    print("\n📊 TENSORBOARD ANALYSIS")
    print("=" * 50)
    
    if tb_path.exists() and any(tb_path.iterdir()):
        print("✅ TensorBoard logs found!")
        print("\n🚀 To view detailed training metrics:")
        print("   1. Install TensorBoard: pip install tensorboard")
        print("   2. Run: tensorboard --logdir logs/smart_highway_training/")
        print("   3. Open: http://localhost:6006")
        print("\n📈 Key metrics to examine:")
        print("   • Reward progression")
        print("   • Policy/Value loss")
        print("   • Episode length")
        print("   • Learning rate")
        print("   • Entropy (exploration)")
    else:
        print("❌ No TensorBoard logs found. Run training first!")

def compare_training_runs():
    """Compare multiple training runs if available."""
    tb_path = Path("logs/smart_highway_training")
    
    if not tb_path.exists():
        print("❌ No training logs to compare")
        return
    
    runs = [d for d in tb_path.iterdir() if d.is_dir()]
    
    if len(runs) < 2:
        print(f"⚠️  Only {len(runs)} training run(s) found. Need at least 2 to compare.")
        return
    
    print(f"\n🏁 COMPARING {len(runs)} TRAINING RUNS")
    print("=" * 50)
    
    for i, run in enumerate(runs):
        print(f"   Run {i+1}: {run.name}")
    
    print("\n📊 To compare runs in TensorBoard:")
    print("   tensorboard --logdir logs/smart_highway_training/")
    print("   Each run will appear as a separate line in the plots")

def generate_performance_report():
    """Generate a comprehensive performance report."""
    print("\n📋 PERFORMANCE REPORT")
    print("=" * 50)
    
    # Check for trained model
    model_path = Path("trained_models/ppo_smart_highway.zip")
    if model_path.exists():
        print("✅ Trained model found")
        model_size = model_path.stat().st_size / (1024 * 1024)  # MB
        print(f"   Model size: {model_size:.2f} MB")
        print(f"   Last modified: {pd.Timestamp.fromtimestamp(model_path.stat().st_mtime)}")
    else:
        print("❌ No trained model found")
    
    # Check evaluation data
    eval_file = Path("logs/smart_highway_eval/evaluations.npz")
    if eval_file.exists():
        eval_data = np.load(eval_file)
        rewards = eval_data['results']
        final_performance = rewards[-1].mean() if len(rewards) > 0 else 0
        
        print(f"\n🎯 Performance Summary:")
        print(f"   Final Average Reward: {final_performance:.2f}")
        
        if final_performance > 150:
            print("   ✅ Excellent performance!")
        elif final_performance > 100:
            print("   ✅ Good performance")
        elif final_performance > 50:
            print("   ⚠️  Moderate performance - consider more training")
        else:
            print("   ❌ Poor performance - training needed")
    
    # Training recommendations
    print(f"\n💡 Recommendations:")
    if not model_path.exists():
        print("   • Run training: python train_smart_highway.py --train")
    else:
        print("   • Test performance: python train_smart_highway.py --test")
        print("   • Compare with random: python train_smart_highway.py --compare")
        print("   • Visualize behavior: python visualizers/smart_highway_visualizer.py --use-trained-model")

def main():
    parser = argparse.ArgumentParser(description='Analyze Smart Highway training logs')
    parser.add_argument('--eval', action='store_true',
                       help='Analyze evaluation logs')
    parser.add_argument('--tensorboard', action='store_true',
                       help='Show TensorBoard instructions')
    parser.add_argument('--compare', action='store_true',
                       help='Compare training runs')
    parser.add_argument('--report', action='store_true',
                       help='Generate performance report')
    parser.add_argument('--all', action='store_true',
                       help='Run all analyses')
    
    args = parser.parse_args()
    
    if args.all or not any([args.eval, args.tensorboard, args.compare, args.report]):
        # Run everything if no specific option or --all
        analyze_evaluation_logs()
        analyze_tensorboard_logs()
        compare_training_runs()
        generate_performance_report()
    else:
        if args.eval:
            analyze_evaluation_logs()
        if args.tensorboard:
            analyze_tensorboard_logs()
        if args.compare:
            compare_training_runs()
        if args.report:
            generate_performance_report()

if __name__ == "__main__":
    main() 