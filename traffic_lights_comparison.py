#!/usr/bin/env python3
"""
Traffic Lights Impact Analysis
=============================

Compare journey times and traffic efficiency with and without traffic lights.
This script demonstrates the impact of traffic light control on vehicle flow and safety.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import argparse
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from environments.smart_highway_env import SmartHighwayEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

class TrafficLightsVisualizer:
    """Visualizer for traffic lights impact analysis."""
    
    def __init__(self, grid_size=(10, 10), max_vehicles=16, spawn_rate=0.4, enable_6g=True):
        self.grid_size = grid_size
        self.max_vehicles = max_vehicles
        self.spawn_rate = spawn_rate
        self.enable_6g = enable_6g
        
        # Create environments with and without traffic lights
        self.env_no_lights = SmartHighwayEnv(
            grid_size=grid_size,
            max_vehicles=max_vehicles,
            spawn_rate=spawn_rate,
            enable_6g=enable_6g,
            enable_traffic_lights=False,
            debug=False
        )
        
        self.env_with_lights = SmartHighwayEnv(
            grid_size=grid_size,
            max_vehicles=max_vehicles,
            spawn_rate=spawn_rate,
            enable_6g=enable_6g,
            enable_traffic_lights=True,
            debug=False
        )
        
        # Load trained model if available
        self.model = None
        try:
            self.model = PPO.load("trained_models/ppo_smart_highway")
            print("✅ Using trained model for intelligent behavior")
        except:
            print("ℹ️ No trained model found - using random actions")
    
    def run_comparison_study(self, episode_duration=300):
        """Run comparison between traffic light scenarios."""
        print("🚦" * 30)
        print("🚗 TRAFFIC LIGHTS IMPACT ANALYSIS 🚗")
        print("🚦" * 30)
        print("📋 Comparing two scenarios:")
        print("  🔹 Scenario A: Smart Highway WITHOUT Traffic Lights")
        print("  🔹 Scenario B: Smart Highway WITH Traffic Lights")
        print("🚦" * 30)
        
        # Run simulation without traffic lights
        print("\n📊 Running Scenario A: No Traffic Lights...")
        results_no_lights = self.run_episode(self.env_no_lights, episode_duration, "No Traffic Lights")
        
        print("\n📊 Running Scenario B: With Traffic Lights...")
        results_with_lights = self.run_episode(self.env_with_lights, episode_duration, "With Traffic Lights")
        
        # Analyze and compare results
        self.analyze_comparison(results_no_lights, results_with_lights)
        
        return results_no_lights, results_with_lights
    
    def run_episode(self, env, max_steps, scenario_name):
        """Run a single episode and collect detailed metrics."""
        print(f"  🎯 Running {scenario_name} for {max_steps} steps...")
        
        obs, info = env.reset()
        episode_data = {
            'journey_times': [],
            'wait_times': [],
            'collision_count': 0,
            'total_spawned': 0,
            'total_completed': 0,
            'traffic_efficiency': [],
            'queued_vehicles_over_time': [],
            'scenario': scenario_name
        }
        
        for step in range(max_steps):
            # Get action
            if self.model:
                action, _ = self.model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            
            # Take step
            obs, reward, terminated, truncated, step_info = env.step(action)
            
            # Collect statistics
            stats = env.get_statistics()
            episode_data['total_spawned'] = stats['total_spawned']
            episode_data['total_completed'] = stats['total_completed']
            episode_data['collision_count'] = stats['collision_count']
            
            if env.enable_traffic_lights:
                episode_data['queued_vehicles_over_time'].append(stats['queued_vehicles'])
                episode_data['traffic_efficiency'].append(stats['traffic_efficiency'])
            else:
                episode_data['queued_vehicles_over_time'].append(0)
                episode_data['traffic_efficiency'].append(100)
            
            # Progress indicator
            if step % 50 == 0:
                completion_rate = (stats['total_completed'] / max(stats['total_spawned'], 1)) * 100
                print(f"    Step {step}: Spawned={stats['total_spawned']}, Completed={stats['total_completed']}, "
                      f"Rate={completion_rate:.1f}%, Collisions={stats['collision_count']}")
        
        # Final statistics
        final_stats = env.get_statistics()
        episode_data['journey_times'] = final_stats['journey_times']
        episode_data['avg_journey_time'] = final_stats['avg_journey_time']
        episode_data['collision_rate'] = final_stats['collision_rate']
        episode_data['completion_rate'] = final_stats['completion_rate']
        
        if env.enable_traffic_lights:
            episode_data['total_wait_time'] = final_stats['total_wait_time']
            episode_data['avg_wait_time_per_vehicle'] = final_stats['avg_wait_time_per_vehicle']
        else:
            episode_data['total_wait_time'] = 0
            episode_data['avg_wait_time_per_vehicle'] = 0
        
        print(f"  ✅ {scenario_name} completed!")
        print(f"    📈 Vehicles: {final_stats['total_spawned']} spawned, {final_stats['total_completed']} completed")
        print(f"    ⏱️ Avg Journey Time: {final_stats['avg_journey_time']:.2f} steps")
        print(f"    💥 Collisions: {final_stats['collision_count']}")
        if env.enable_traffic_lights:
            print(f"    🚦 Avg Wait Time: {final_stats['avg_wait_time_per_vehicle']:.2f} steps per vehicle")
        
        return episode_data
    
    def analyze_comparison(self, no_lights, with_lights):
        """Analyze and compare the two scenarios."""
        print("\n📊 COMPREHENSIVE TRAFFIC LIGHTS IMPACT ANALYSIS")
        print("=" * 70)
        
        # Journey time comparison
        print(f"\n🏁 JOURNEY TIME ANALYSIS:")
        print(f"{'Metric':<25} {'No Lights':<15} {'With Lights':<15} {'Difference':<15}")
        print("-" * 70)
        
        # Average journey time
        no_lights_avg = no_lights['avg_journey_time']
        with_lights_avg = with_lights['avg_journey_time']
        journey_diff = with_lights_avg - no_lights_avg
        journey_pct = (journey_diff / max(no_lights_avg, 0.1)) * 100
        
        print(f"{'Avg Journey Time':<25} {no_lights_avg:<15.2f} {with_lights_avg:<15.2f} {journey_diff:+.2f} ({journey_pct:+.1f}%)")
        
        # Wait time analysis
        print(f"{'Wait Time/Vehicle':<25} {no_lights['avg_wait_time_per_vehicle']:<15.2f} {with_lights['avg_wait_time_per_vehicle']:<15.2f} {with_lights['avg_wait_time_per_vehicle']:+.2f}")
        
        # Safety analysis
        no_lights_rate = no_lights['collision_rate']
        with_lights_rate = with_lights['collision_rate']
        safety_diff = no_lights_rate - with_lights_rate
        safety_pct = (safety_diff / max(no_lights_rate, 0.1)) * 100
        
        print(f"{'Collision Rate':<25} {no_lights_rate:<15.2f} {with_lights_rate:<15.2f} {safety_diff:+.2f} ({safety_pct:+.1f}%)")
        
        # Completion analysis
        no_lights_comp = no_lights['completion_rate']
        with_lights_comp = with_lights['completion_rate']
        comp_diff = with_lights_comp - no_lights_comp
        
        print(f"{'Completion Rate %':<25} {no_lights_comp:<15.2f} {with_lights_comp:<15.2f} {comp_diff:+.2f}")
        
        print("-" * 70)
        
        # Key findings
        print(f"\n🎯 KEY FINDINGS:")
        print("-" * 30)
        
        if journey_diff > 0:
            print(f"🔸 Traffic lights INCREASE journey time by {journey_diff:.2f} steps ({journey_pct:+.1f}%)")
        else:
            print(f"🔸 Traffic lights DECREASE journey time by {abs(journey_diff):.2f} steps ({abs(journey_pct):.1f}%)")
        
        if safety_diff > 0:
            print(f"🔸 Traffic lights IMPROVE safety by {safety_diff:.2f} fewer collisions ({safety_pct:.1f}% reduction)")
        else:
            print(f"🔸 Traffic lights WORSEN safety by {abs(safety_diff):.2f} more collisions")
        
        print(f"🔸 Traffic lights add {with_lights['avg_wait_time_per_vehicle']:.2f} steps wait time per vehicle")
        
        if comp_diff > 0:
            print(f"🔸 Traffic lights IMPROVE completion rate by {comp_diff:.1f}%")
        else:
            print(f"🔸 Traffic lights REDUCE completion rate by {abs(comp_diff):.1f}%")
        
        # Generate visualizations
        self.create_comparison_charts(no_lights, with_lights)
        
        # Save detailed report
        self.save_comparison_report(no_lights, with_lights)
    
    def create_comparison_charts(self, no_lights, with_lights):
        """Create comprehensive comparison charts."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # Journey time distribution
            if no_lights['journey_times'] and with_lights['journey_times']:
                ax1.hist(no_lights['journey_times'], bins=20, alpha=0.7, label='No Traffic Lights', color='#FF6B6B')
                ax1.hist(with_lights['journey_times'], bins=20, alpha=0.7, label='With Traffic Lights', color='#4ECDC4')
                ax1.set_title('Journey Time Distribution', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Journey Time (steps)')
                ax1.set_ylabel('Frequency')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Queue length over time
            steps = range(len(with_lights['queued_vehicles_over_time']))
            ax2.plot(steps, no_lights['queued_vehicles_over_time'], label='No Traffic Lights', color='#FF6B6B')
            ax2.plot(steps, with_lights['queued_vehicles_over_time'], label='With Traffic Lights', color='#4ECDC4')
            ax2.set_title('Queued Vehicles Over Time', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Simulation Step')
            ax2.set_ylabel('Queued Vehicles')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Performance metrics comparison
            metrics = ['Journey Time', 'Wait Time', 'Collision Rate', 'Completion Rate']
            no_lights_values = [
                no_lights['avg_journey_time'],
                no_lights['avg_wait_time_per_vehicle'],
                no_lights['collision_rate'],
                no_lights['completion_rate']
            ]
            with_lights_values = [
                with_lights['avg_journey_time'],
                with_lights['avg_wait_time_per_vehicle'],
                with_lights['collision_rate'],
                with_lights['completion_rate']
            ]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, no_lights_values, width, label='No Traffic Lights', color='#FF6B6B')
            bars2 = ax3.bar(x + width/2, with_lights_values, width, label='With Traffic Lights', color='#4ECDC4')
            
            ax3.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Metrics')
            ax3.set_ylabel('Value')
            ax3.set_xticks(x)
            ax3.set_xticklabels(metrics, rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom')
            for bar in bars2:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}', ha='center', va='bottom')
            
            # Traffic efficiency over time
            steps = range(len(with_lights['traffic_efficiency']))
            ax4.plot(steps, no_lights['traffic_efficiency'], label='No Traffic Lights', color='#FF6B6B')
            ax4.plot(steps, with_lights['traffic_efficiency'], label='With Traffic Lights', color='#4ECDC4')
            ax4.set_title('Traffic Efficiency Over Time', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Simulation Step')
            ax4.set_ylabel('Efficiency (%)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(0, 100)
            
            plt.tight_layout()
            
            # Save chart
            os.makedirs("traffic_light_results", exist_ok=True)
            plt.savefig('traffic_light_results/traffic_lights_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("📊 Comparison charts saved: traffic_light_results/traffic_lights_comparison.png")
            
        except Exception as e:
            print(f"⚠️ Could not create charts: {e}")
    
    def save_comparison_report(self, no_lights, with_lights):
        """Save detailed comparison report."""
        try:
            os.makedirs("traffic_light_results", exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open('traffic_light_results/traffic_lights_report.txt', 'w') as f:
                f.write("TRAFFIC LIGHTS IMPACT ANALYSIS REPORT\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {timestamp}\n\n")
                
                f.write("STUDY OVERVIEW:\n")
                f.write("This study compares vehicle journey times and traffic efficiency\n")
                f.write("with and without traffic light control in smart highway scenarios.\n\n")
                
                f.write("SCENARIOS COMPARED:\n")
                f.write("A. Smart Highway WITHOUT Traffic Lights\n")
                f.write("B. Smart Highway WITH Traffic Lights\n\n")
                
                f.write("DETAILED RESULTS:\n")
                f.write("-" * 30 + "\n")
                
                scenarios = [
                    ("NO TRAFFIC LIGHTS", no_lights),
                    ("WITH TRAFFIC LIGHTS", with_lights)
                ]
                
                for name, results in scenarios:
                    f.write(f"\n{name}:\n")
                    f.write(f"  Average Journey Time: {results['avg_journey_time']:.2f} steps\n")
                    f.write(f"  Average Wait Time: {results['avg_wait_time_per_vehicle']:.2f} steps per vehicle\n")
                    f.write(f"  Collision Rate: {results['collision_rate']:.2f} collisions per 100 steps\n")
                    f.write(f"  Completion Rate: {results['completion_rate']:.2f}%\n")
                    f.write(f"  Total Spawned: {results['total_spawned']} vehicles\n")
                    f.write(f"  Total Completed: {results['total_completed']} vehicles\n")
                
                f.write("\nCOMPARATIVE ANALYSIS:\n")
                f.write("-" * 30 + "\n")
                
                journey_diff = with_lights['avg_journey_time'] - no_lights['avg_journey_time']
                journey_pct = (journey_diff / max(no_lights['avg_journey_time'], 0.1)) * 100
                
                f.write(f"Journey Time Impact: {journey_diff:+.2f} steps ({journey_pct:+.1f}%)\n")
                f.write(f"Wait Time Added: {with_lights['avg_wait_time_per_vehicle']:.2f} steps per vehicle\n")
                
                safety_diff = no_lights['collision_rate'] - with_lights['collision_rate']
                f.write(f"Safety Improvement: {safety_diff:+.2f} fewer collisions per 100 steps\n")
                
                comp_diff = with_lights['completion_rate'] - no_lights['completion_rate']
                f.write(f"Completion Rate Change: {comp_diff:+.2f}%\n")
                
                f.write("\nKEY FINDINGS:\n")
                f.write("-" * 20 + "\n")
                if journey_diff > 0:
                    f.write("• Traffic lights increase journey time but may improve safety\n")
                else:
                    f.write("• Traffic lights improve both journey time and traffic flow\n")
                f.write("• Traffic light timing optimization can balance efficiency and safety\n")
                f.write("• Queue management is critical for traffic light effectiveness\n")
                
            print("📄 Detailed report saved: traffic_light_results/traffic_lights_report.txt")
            
        except Exception as e:
            print(f"⚠️ Could not save report: {e}")

def main():
    parser = argparse.ArgumentParser(description='Traffic Lights Impact Analysis')
    parser.add_argument('--duration', type=int, default=300,
                       help='Episode duration in steps')
    parser.add_argument('--vehicles', type=int, default=16,
                       help='Maximum vehicles in simulation')
    parser.add_argument('--spawn-rate', type=float, default=0.4,
                       help='Vehicle spawn rate')
    parser.add_argument('--no-6g', action='store_true',
                       help='Disable 6G communication')
    
    args = parser.parse_args()
    
    visualizer = TrafficLightsVisualizer(
        max_vehicles=args.vehicles,
        spawn_rate=args.spawn_rate,
        enable_6g=not args.no_6g
    )
    
    # Run the comparison study
    visualizer.run_comparison_study(episode_duration=args.duration)

if __name__ == "__main__":
    main() 