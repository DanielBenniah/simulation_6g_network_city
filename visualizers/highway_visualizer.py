#!/usr/bin/env python3
"""
Highway traffic visualizer showing continuous traffic flow with spawn zones,
dedicated lanes, and realistic speed control behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import argparse
from stable_baselines3 import PPO
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environments.highway_traffic_env import HighwayTrafficEnv

class HighwayTrafficVisualizer:
    """Visualizer for highway-style traffic simulation."""
    
    def __init__(self, env, model=None):
        self.env = env
        self.model = model
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        grid_width, grid_height = env.grid_size
        self.ax.set_xlim(-1, grid_width + 1)
        self.ax.set_ylim(-1, grid_height + 1)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('🛣️ Highway Traffic - Continuous Flow with Spawn Zones 🚗', fontsize=16, fontweight='bold')
        
        # Vehicle patches
        self.vehicle_patches = []
        
        # Info displays
        self.info_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                     verticalalignment='top', fontsize=11,
                                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
        
        self.flow_text = self.ax.text(0.98, 0.98, '', transform=self.ax.transAxes, 
                                     verticalalignment='top', horizontalalignment='right', fontsize=10,
                                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
        
        # Draw highway infrastructure
        self._draw_highway_grid()
    
    def _draw_highway_grid(self):
        """Draw highway grid with lanes and spawn zones."""
        grid_width, grid_height = self.env.grid_size
        
        # Draw horizontal lanes (left-right and right-left traffic)
        for y in range(grid_height):
            # Left-to-right lane (upper part)
            lane_y = y + 0.3
            self.ax.plot([0, grid_width], [lane_y, lane_y], color='darkblue', linewidth=4, alpha=0.7)
            # Right-to-left lane (lower part)
            lane_y = y - 0.3
            self.ax.plot([0, grid_width], [lane_y, lane_y], color='darkgreen', linewidth=4, alpha=0.7)
        
        # Draw vertical lanes (top-bottom and bottom-top traffic)
        for x in range(grid_width):
            # Top-to-bottom lane (right part)
            lane_x = x + 0.3
            self.ax.plot([lane_x, lane_x], [0, grid_height], color='darkorange', linewidth=4, alpha=0.7)
            # Bottom-to-top lane (left part)
            lane_x = x - 0.3
            self.ax.plot([lane_x, lane_x], [0, grid_height], color='darkred', linewidth=4, alpha=0.7)
        
        # Draw spawn zones
        spawn_zones = [
            # Left spawn (→)
            patches.Rectangle((-0.8, -0.5), 0.6, grid_height + 1, color='blue', alpha=0.3),
            # Right spawn (←)
            patches.Rectangle((grid_width + 0.2, -0.5), 0.6, grid_height + 1, color='green', alpha=0.3),
            # Top spawn (↓)
            patches.Rectangle((-0.5, -0.8), grid_width + 1, 0.6, color='orange', alpha=0.3),
            # Bottom spawn (↑)
            patches.Rectangle((-0.5, grid_height + 0.2), grid_width + 1, 0.6, color='red', alpha=0.3)
        ]
        
        for zone in spawn_zones:
            self.ax.add_patch(zone)
        
        # Add spawn zone labels
        self.ax.text(-0.5, grid_height/2, '→\nSPAWN', ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='blue')
        self.ax.text(grid_width + 0.5, grid_height/2, '←\nSPAWN', ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='green')
        self.ax.text(grid_width/2, -0.5, '↓ SPAWN', ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='orange')
        self.ax.text(grid_width/2, grid_height + 0.5, '↑ SPAWN', ha='center', va='center', 
                    fontsize=10, fontweight='bold', color='red')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='darkblue', linewidth=4, label='→ Left-to-Right'),
            plt.Line2D([0], [0], color='darkgreen', linewidth=4, label='← Right-to-Left'),
            plt.Line2D([0], [0], color='darkorange', linewidth=4, label='↓ Top-to-Bottom'),
            plt.Line2D([0], [0], color='darkred', linewidth=4, label='↑ Bottom-to-Top')
        ]
        self.ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.85))
    
    def _draw_vehicles(self, active_vehicles):
        """Draw vehicles with lane-specific colors and speed indicators."""
        # Clear previous patches
        for patch in self.vehicle_patches:
            patch.remove()
        self.vehicle_patches.clear()
        
        # Direction colors
        direction_colors = {
            0: '#4169E1',  # Left-to-right: Royal Blue
            1: '#228B22',  # Right-to-left: Forest Green  
            2: '#FF8C00',  # Top-to-bottom: Dark Orange
            3: '#DC143C'   # Bottom-to-top: Crimson
        }
        
        direction_symbols = {0: '→', 1: '←', 2: '↓', 3: '↑'}
        
        total_speed = 0
        speed_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Count by direction
        
        for vehicle in active_vehicles:
            x, y = vehicle['position']
            vx, vy = vehicle['velocity']
            direction_code = vehicle['direction_code']
            speed = vehicle['speed']
            vehicle_id = vehicle['id']
            
            total_speed += speed
            speed_counts[direction_code] += 1
            
            color = direction_colors[direction_code]
            symbol = direction_symbols[direction_code]
            
            # Vehicle size based on speed
            base_size = 0.15
            size_factor = 1.0 + speed * 0.3
            width = base_size * size_factor
            height = base_size * 0.8
            
            # Color intensity based on speed
            alpha = 0.7 + min(speed * 0.2, 0.3)
            
            # Draw vehicle
            vehicle_rect = patches.Rectangle(
                (x - width/2, y - height/2), width, height,
                color=color, alpha=alpha, linewidth=1, edgecolor='black'
            )
            self.ax.add_patch(vehicle_rect)
            self.vehicle_patches.append(vehicle_rect)
            
            # Speed and direction indicator with journey time
            speed_category = "🐌" if speed < 0.6 else "🚗" if speed < 1.2 else "🏎️"
            journey_time = vehicle.get('journey_time', 0)
            vehicle_text = f'{symbol}\n{speed_category}\n{speed:.1f}\n{journey_time:.0f}s'
            text = self.ax.text(x, y + 0.35, vehicle_text, ha='center', va='center', 
                               fontsize=6, fontweight='bold', color='white',
                               bbox=dict(boxstyle='round,pad=0.1', facecolor=color, alpha=0.8))
            self.vehicle_patches.append(text)
            
            # Velocity arrow
            if speed > 0.2:
                arrow_length = 0.3
                arrow = patches.FancyArrowPatch(
                    (x, y), (x + vx/speed * arrow_length, y + vy/speed * arrow_length),
                    arrowstyle='->', mutation_scale=12, color='white', linewidth=2
                )
                self.ax.add_patch(arrow)
                self.vehicle_patches.append(arrow)
        
        return len(active_vehicles), total_speed, speed_counts
    
    def _update_displays(self, step, total_reward, active_count, total_speed, speed_counts, 
                        collisions, info, episode, journey_stats):
        """Update information displays."""
        avg_speed = total_speed / max(active_count, 1)
        
        # Main info
        main_info = f"Episode: {episode} | Step: {step}\n"
        main_info += f"Agent Reward: {total_reward:.2f}\n"
        main_info += f"Active Vehicles: {active_count}\n"
        main_info += f"Avg Speed: {avg_speed:.2f}\n"
        main_info += f"Collisions: {collisions}\n"
        main_info += f"Vehicles Spawned: {info.get('vehicles_spawned', 0)}\n"
        main_info += f"Vehicles Exited: {info.get('vehicles_exited', 0)}\n"
        main_info += f"Throughput: {info.get('throughput', 0)}\n"
        main_info += f"Journey Stats:\n"
        main_info += f"  Completed: {journey_stats['total_completed']}\n"
        if journey_stats['total_completed'] > 0:
            main_info += f"  Avg Time: {journey_stats['average_time']:.1f}s\n"
            main_info += f"  Range: {journey_stats['min_time']:.1f}-{journey_stats['max_time']:.1f}s"
        
        self.info_text.set_text(main_info)
        
        # Traffic flow details
        flow_details = "Traffic Flow:\n"
        directions = ['→ L2R', '← R2L', '↓ T2B', '↑ B2T']
        for i, direction in enumerate(directions):
            count = speed_counts[i]
            if count > 0:
                avg_dir_speed = sum(v['speed'] for v in self.env.get_active_vehicles() 
                                  if v['direction_code'] == i) / count
                flow_details += f"{direction}: {count} ({avg_dir_speed:.1f})\n"
            else:
                flow_details += f"{direction}: 0\n"
        
        flow_details += f"\nSpawn Rate: {self.env.spawn_rate:.2f}"
        
        self.flow_text.set_text(flow_details)
    
    def run_highway_episode(self, episode_num=1, max_steps=400, delay=0.15):
        """Run a highway traffic episode."""
        print(f"\n🛣️ === HIGHWAY TRAFFIC EPISODE {episode_num} ===")
        print("🎯 Features:")
        print("   • Continuous vehicle spawning from edges")
        print("   • Dedicated lanes (no head-on collisions)")
        print("   • Vehicles only accelerate/brake (no direction changes)")
        print("   • Collisions only at intersections (perpendicular traffic)")
        print("   • Realistic highway-style following behavior")
        
        obs, info = self.env.reset()
        total_reward = 0
        step = 0
        total_collisions = 0
        total_spawned = info.get('vehicles_spawned', 0)
        total_exited = 0
        
        plt.ion()
        
        while step < max_steps:
            # Get action (random for demo, or from trained model)
            if self.model:
                action, _ = self.model.predict(obs, deterministic=True)
            else:
                action = self.env.action_space.sample()
            
            # Take step
            obs, reward, terminated, truncated, step_info = self.env.step(action)
            total_reward += reward
            step += 1
            
            # Track events
            if step_info.get('collisions'):
                total_collisions += len(step_info['collisions'])
                print(f"   💥 Step {step}: Collision at intersection!")
            
            if step_info.get('vehicles_spawned', 0) > 0:
                total_spawned += step_info['vehicles_spawned']
                
            if step_info.get('vehicles_exited', 0) > 0:
                total_exited += step_info['vehicles_exited']
                if step % 10 == 0:  # Don't spam
                    print(f"   ✅ Step {step}: {step_info['vehicles_exited']} vehicle(s) completed journey")
            
            # Update visualization
            active_vehicles = self.env.get_active_vehicles()
            active_count, total_speed, speed_counts = self._draw_vehicles(active_vehicles)
            journey_stats = self.env.get_journey_statistics()
            self._update_displays(step, total_reward, active_count, total_speed, speed_counts,
                                total_collisions, step_info, episode_num, journey_stats)
            
            plt.draw()
            plt.pause(delay)
            
            # Progress updates
            if step % 50 == 0:
                avg_speed = total_speed / max(active_count, 1)
                print(f"   📊 Step {step}: {active_count} vehicles, avg speed {avg_speed:.2f}, "
                     f"throughput {total_exited}")
            
            if terminated or truncated:
                break
        
        # Get final journey statistics
        final_journey_stats = self.env.get_journey_statistics()
        
        print(f"\n📈 HIGHWAY EPISODE {episode_num} SUMMARY:")
        print(f"   Duration: {step} steps")
        print(f"   Agent Total Reward: {total_reward:.2f}")
        print(f"   Total Collisions: {total_collisions}")
        print(f"   Vehicles Spawned: {total_spawned}")
        print(f"   Vehicles Completed Journey: {total_exited}")
        print(f"   Final Active Vehicles: {active_count}")
        print(f"   Throughput Rate: {total_exited/step:.3f} vehicles/step")
        if final_journey_stats['total_completed'] > 0:
            print(f"   Journey Time Stats:")
            print(f"     Average: {final_journey_stats['average_time']:.1f} steps")
            print(f"     Range: {final_journey_stats['min_time']:.1f} - {final_journey_stats['max_time']:.1f} steps")
            print(f"     Efficiency: {final_journey_stats['total_completed']/total_spawned*100:.1f}% completion rate")
        
        return total_reward, step, total_collisions, total_exited

def main():
    parser = argparse.ArgumentParser(description='Highway traffic simulation with spawn zones')
    parser.add_argument('--episodes', type=int, default=2,
                       help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=300,
                       help='Maximum steps per episode')
    parser.add_argument('--delay', type=float, default=0.15,
                       help='Animation delay (seconds)')
    parser.add_argument('--grid-size', type=int, default=10,
                       help='Grid size')
    parser.add_argument('--max-vehicles', type=int, default=24,
                       help='Maximum vehicles')
    parser.add_argument('--spawn-rate', type=float, default=0.4,
                       help='Vehicle spawn rate (0.0-1.0)')
    parser.add_argument('--use-trained-model', action='store_true',
                       help='Use trained model instead of random actions')
    
    args = parser.parse_args()
    
    print("🛣️" * 30)
    print("🚗 HIGHWAY TRAFFIC SIMULATION 🚗")
    print("🛣️" * 30)
    print(f"Grid: {args.grid_size}x{args.grid_size}")
    print(f"Max Vehicles: {args.max_vehicles}")
    print(f"Spawn Rate: {args.spawn_rate}")
    print(f"Episodes: {args.episodes}")
    print("Key Features:")
    print("  🛣️  Continuous traffic flow from spawn zones")
    print("  🚗 Dedicated lanes (no head-on collisions)")
    print("  ⚡ Speed control only (accelerate/brake)")
    print("  💥 Intersection collisions only")
    print("  📊 Realistic following behavior")
    print("🛣️" * 30)
    
    # Create highway environment
    env = HighwayTrafficEnv(
        grid_size=(args.grid_size, args.grid_size),
        max_vehicles=args.max_vehicles,
        spawn_rate=args.spawn_rate,
        debug=False
    )
    
    # Load model if requested
    model = None
    if args.use_trained_model:
        try:
            model = PPO.load('ppo_city_traffic.zip')
            print("✅ Using trained PPO model for agent vehicle")
        except Exception as e:
            print(f"❌ Could not load model: {e}")
            print("🎲 Using random actions for agent vehicle")
    else:
        print("🎲 Using random actions for agent vehicle")
    
    # Create visualizer
    visualizer = HighwayTrafficVisualizer(env, model)
    
    # Run episodes
    total_throughput = 0
    total_collisions = 0
    try:
        for episode in range(1, args.episodes + 1):
            reward, steps, collisions, throughput = visualizer.run_highway_episode(
                episode_num=episode,
                max_steps=args.max_steps,
                delay=args.delay
            )
            total_throughput += throughput
            total_collisions += collisions
            
            if episode < args.episodes:
                print(f"\n⏳ Waiting 3 seconds before next episode...")
                time.sleep(3)
    
    except KeyboardInterrupt:
        print("\n⚠️  Highway simulation interrupted by user")
    
    print(f"\n🏁 FINAL HIGHWAY TRAFFIC SUMMARY:")
    print(f"Total vehicle throughput: {total_throughput}")
    print(f"Total collisions: {total_collisions}")
    print(f"This demonstrates realistic highway-style traffic flow!")
    
    input("Press Enter to close...")
    plt.close()

if __name__ == "__main__":
    main() 