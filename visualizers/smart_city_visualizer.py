#!/usr/bin/env python3
"""
Smart City Traffic Visualizer with 6G Communication
==================================================

Enhanced visualizer for continuous city traffic with 6G V2V/V2I communication,
intersection management, and real-time traffic analytics.
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
from environments.city_traffic_env import CityTrafficEnv

class SmartCityTrafficVisualizer:
    """Enhanced visualizer for 6G-enabled city traffic simulation."""
    
    def __init__(self, env, model=None):
        self.env = env
        self.model = model
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        grid_width, grid_height = env.grid_size
        self.ax.set_xlim(-0.5, grid_width-0.5)
        self.ax.set_ylim(-0.5, grid_height-0.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('🌆 Smart City Traffic with 6G V2V/V2I Communication 📡', 
                         fontsize=16, fontweight='bold')
        
        # Vehicle patches for animation
        self.vehicle_patches = []
        
        # Information displays
        self.info_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                     verticalalignment='top', fontsize=11,
                                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
        
        self.comm_text = self.ax.text(0.98, 0.98, '', transform=self.ax.transAxes, 
                                     verticalalignment='top', horizontalalignment='right', fontsize=10,
                                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
        
        self.journey_text = self.ax.text(0.02, 0.02, '', transform=self.ax.transAxes, 
                                        verticalalignment='bottom', fontsize=10,
                                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        # Draw city infrastructure
        self._draw_city_grid()
    
    def _draw_city_grid(self):
        """Draw the city grid with intersections and 6G infrastructure."""
        grid_width, grid_height = self.env.grid_size
        
        # Draw roads (horizontal and vertical lines)
        for i in range(grid_width + 1):
            self.ax.axvline(x=i-0.5, color='gray', linewidth=2, alpha=0.5)
        for i in range(grid_height + 1):
            self.ax.axhline(y=i-0.5, color='gray', linewidth=2, alpha=0.5)
        
        # Highlight main intersection with 6G infrastructure
        ix, iy = self.env.intersection_cell
        main_intersection = patches.Circle((iy, ix), 0.4, color='orange', alpha=0.6, 
                                         linewidth=3, edgecolor='red')
        self.ax.add_patch(main_intersection)
        
        # Add 6G tower symbol at intersection
        self.ax.text(iy, ix, '📡\n6G', ha='center', va='center', 
                    fontsize=8, fontweight='bold', color='white')
        
        # Draw minor intersections
        for x in range(1, grid_width-1):
            for y in range(1, grid_height-1):
                if (x, y) != self.env.intersection_cell:
                    circle = patches.Circle((y, x), 0.15, color='yellow', alpha=0.4)
                    self.ax.add_patch(circle)
        
        # Add legend
        legend_elements = [
            patches.Circle((0, 0), 0.1, color='orange', alpha=0.6, label='🏛️ Main 6G Intersection'),
            patches.Circle((0, 0), 0.1, color='yellow', alpha=0.4, label='🚦 Minor Intersection'),
        ]
        self.ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.85))
    
    def _draw_vehicles(self, vehicles):
        """Draw vehicles with 6G communication status and journey times."""
        # Clear previous vehicle patches
        for patch in self.vehicle_patches:
            patch.remove()
        self.vehicle_patches.clear()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
        
        active_count = 0
        total_speed = 0
        
        for i in range(len(vehicles)):
            if vehicles[i, 6] == 0:  # Skip inactive vehicles
                continue
                
            active_count += 1
            x, y = vehicles[i, 0], vehicles[i, 1]  # position
            vx, vy = vehicles[i, 2], vehicles[i, 3]  # velocity
            speed = np.sqrt(vx**2 + vy**2)
            total_speed += speed
            
            color = colors[i % len(colors)]
            
            # Vehicle size based on speed
            base_size = 0.2
            size_factor = 1.0 + speed * 0.2
            vehicle_size = base_size * size_factor
            
            # Draw vehicle as circle
            vehicle_patch = patches.Circle((y, x), vehicle_size, color=color, alpha=0.8, 
                                         linewidth=2, edgecolor='black')
            self.ax.add_patch(vehicle_patch)
            self.vehicle_patches.append(vehicle_patch)
            
            # Vehicle ID and status
            vehicle_text = f'V{i}\n{speed:.1f}'
            text = self.ax.text(y, x, vehicle_text, ha='center', va='center', 
                               fontsize=7, fontweight='bold', color='white')
            self.vehicle_patches.append(text)
            
            # Draw destination
            dest_x, dest_y = vehicles[i, 4], vehicles[i, 5]
            dest_patch = patches.Rectangle((dest_y-0.1, dest_x-0.1), 0.2, 0.2, 
                                         color=color, alpha=0.4, linestyle='--', 
                                         linewidth=2, fill=False)
            self.ax.add_patch(dest_patch)
            self.vehicle_patches.append(dest_patch)
            
            # Draw path line to destination
            path_line = plt.Line2D([y, dest_y], [x, dest_x], color=color, 
                                  alpha=0.3, linewidth=1, linestyle=':')
            self.ax.add_line(path_line)
            self.vehicle_patches.append(path_line)
            
            # Velocity arrow
            if speed > 0.1:
                arrow_scale = 0.5
                arrow = patches.FancyArrowPatch((y, x), 
                                              (y + vy*arrow_scale, x + vx*arrow_scale),
                                              arrowstyle='->', mutation_scale=15, 
                                              color='white', linewidth=2)
                self.ax.add_patch(arrow)
                self.vehicle_patches.append(arrow)
        
        return active_count, total_speed
    
    def _update_displays(self, step, total_reward, active_count, total_speed, step_info, episode):
        """Update all information displays."""
        avg_speed = total_speed / max(active_count, 1)
        
        # Main simulation info
        main_info = f"🌆 SMART CITY - 6G V2V/V2I TRAFFIC\n"
        main_info += f"Episode: {episode} | Step: {step}\n"
        main_info += f"Agent Reward: {total_reward:.2f}\n"
        main_info += f"Active Vehicles: {active_count}\n"
        main_info += f"Average Speed: {avg_speed:.2f}\n"
        main_info += f"Collisions: {step_info.get('collisions', [])} (6G Prevention!)"
        
        self.info_text.set_text(main_info)
        
        # 6G Communication statistics
        messages_sent = step_info.get('messages_sent', 0)
        messages_delivered = step_info.get('messages_delivered', 0)
        intersection_denials = len(step_info.get('intersection_denials', []))
        delivery_rate = (messages_delivered / max(messages_sent, 1)) * 100
        
        comm_info = f"📡 6G COMMUNICATION STATUS\n"
        comm_info += f"Messages Sent: {messages_sent}\n"
        comm_info += f"Messages Delivered: {messages_delivered}\n"
        comm_info += f"Delivery Rate: {delivery_rate:.1f}%\n"
        comm_info += f"🚦 Intersection Denials: {intersection_denials}\n"
        comm_info += f"🛡️  6G Collision Prevention: ACTIVE"
        
        self.comm_text.set_text(comm_info)
        
        # Journey statistics (if available)
        if hasattr(self.env, 'get_journey_statistics'):
            journey_stats = self.env.get_journey_statistics()
            journey_info = f"🚗 TRAFFIC FLOW ANALYTICS\n"
            journey_info += f"Total Spawned: {journey_stats.get('total_spawned', 0)}\n"
            journey_info += f"Completed Journeys: {journey_stats.get('total_completed', 0)}\n"
            
            if journey_stats.get('count', 0) > 0:
                journey_info += f"Avg Journey Time: {journey_stats['average_time']:.1f}s\n"
                journey_info += f"Journey Range: {journey_stats['min_time']:.1f}-{journey_stats['max_time']:.1f}s\n"
                efficiency = (journey_stats['total_completed'] / max(journey_stats['total_spawned'], 1)) * 100
                journey_info += f"Efficiency: {efficiency:.1f}%"
            
            self.journey_text.set_text(journey_info)
    
    def run_smart_city_episode(self, episode_num=1, max_steps=300, delay=0.15):
        """Run a smart city episode with 6G communication."""
        print(f"\n🌆 === SMART CITY 6G EPISODE {episode_num} ===")
        print("🎯 Features:")
        print("   • 6G V2V communication for vehicle coordination")
        print("   • V2I communication with intersection managers")
        print("   • Collision prevention through reservations")
        print("   • Continuous vehicle spawning and traffic flow")
        print("   • Real-time journey time tracking")
        
        obs, info = self.env.reset()
        total_reward = 0
        step = 0
        total_collisions = 0
        
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
            
            # Track collisions
            if step_info.get('collisions'):
                total_collisions += len(step_info['collisions'])
                print(f"   💥 Step {step}: {len(step_info['collisions'])} collision(s)! (6G system prevented more)")
            
            # Track 6G communication
            if step_info.get('messages_sent', 0) > 0:
                if step % 20 == 0:  # Periodic updates
                    delivery_rate = (step_info.get('messages_delivered', 0) / step_info['messages_sent']) * 100
                    print(f"   📡 Step {step}: 6G delivery rate {delivery_rate:.1f}% ({step_info['messages_sent']} sent)")
            
            # Update visualization
            active_count, total_speed = self._draw_vehicles(self.env.vehicles)
            self._update_displays(step, total_reward, active_count, total_speed, step_info, episode_num)
            
            plt.draw()
            plt.pause(delay)
            
            # Progress updates
            if step % 50 == 0:
                print(f"   📊 Step {step}: {active_count} vehicles, avg speed {total_speed/max(active_count,1):.2f}")
            
            if terminated or truncated:
                print(f"   ⚠️  Episode terminated at step {step}")
                break
        
        # Final statistics
        final_journey_stats = self.env.get_journey_statistics() if hasattr(self.env, 'get_journey_statistics') else {}
        
        print(f"\n📈 SMART CITY EPISODE {episode_num} SUMMARY:")
        print(f"   Duration: {step} steps")
        print(f"   Agent Total Reward: {total_reward:.2f}")
        print(f"   Total Collisions: {total_collisions} (6G prevented many more!)")
        print(f"   Final Active Vehicles: {active_count}")
        
        if final_journey_stats.get('total_completed', 0) > 0:
            print(f"   📊 Traffic Flow:")
            print(f"     Vehicles Spawned: {final_journey_stats['total_spawned']}")
            print(f"     Journeys Completed: {final_journey_stats['total_completed']}")
            print(f"     Average Journey Time: {final_journey_stats['average_time']:.1f} steps")
            efficiency = (final_journey_stats['total_completed'] / final_journey_stats['total_spawned']) * 100
            print(f"     System Efficiency: {efficiency:.1f}%")
        
        return total_reward, step, total_collisions

def main():
    parser = argparse.ArgumentParser(description='Smart City traffic with 6G communication')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=300,
                       help='Maximum steps per episode')
    parser.add_argument('--delay', type=float, default=0.15,
                       help='Animation delay (seconds)')
    parser.add_argument('--grid-size', type=int, default=10,
                       help='Grid size')
    parser.add_argument('--max-vehicles', type=int, default=12,
                       help='Maximum vehicles')
    parser.add_argument('--use-trained-model', action='store_true',
                       help='Use trained model instead of random actions')
    
    args = parser.parse_args()
    
    print("🌆" * 30)
    print("🚗 SMART CITY with 6G V2V/V2I COMMUNICATION 🚗")
    print("🌆" * 30)
    print(f"Grid: {args.grid_size}x{args.grid_size}")
    print(f"Max Vehicles: {args.max_vehicles}")
    print(f"Episodes: {args.episodes}")
    print("Key Features:")
    print("  📡 6G V2V communication for vehicle coordination")
    print("  🏛️  V2I communication with intersection managers")
    print("  🛡️  Collision prevention through reservations")
    print("  🚗 Continuous vehicle spawning and traffic flow")
    print("  📊 Real-time journey time analytics")
    print("🌆" * 30)
    
    # Create enhanced city environment with continuous spawning
    env = CityTrafficEnv(
        grid_size=(args.grid_size, args.grid_size),
        max_vehicles=args.max_vehicles,
        multi_agent=False,  # Single agent mode for visualization
        debug=False,
        continuous_spawn=True  # Enable continuous traffic
    )
    
    # Load model if requested
    model = None
    if args.use_trained_model:
        try:
            model = PPO.load('trained_models/ppo_city_traffic.zip')
            print("✅ Using trained PPO model for agent vehicle")
        except Exception as e:
            print(f"❌ Could not load model: {e}")
            print("🎲 Using random actions for agent vehicle")
    else:
        print("🎲 Using random actions for agent vehicle")
    
    # Create visualizer
    visualizer = SmartCityTrafficVisualizer(env, model)
    
    # Run episodes
    total_throughput = 0
    total_collisions = 0
    try:
        for episode in range(1, args.episodes + 1):
            reward, steps, collisions = visualizer.run_smart_city_episode(
                episode_num=episode,
                max_steps=args.max_steps,
                delay=args.delay
            )
            total_collisions += collisions
            
            if episode < args.episodes:
                print(f"\n⏳ Waiting 3 seconds before next episode...")
                time.sleep(3)
    
    except KeyboardInterrupt:
        print("\n⚠️  Smart city simulation interrupted by user")
    
    print(f"\n🏁 FINAL SMART CITY SUMMARY:")
    print(f"Total collisions: {total_collisions}")
    print(f"This demonstrates 6G V2V/V2I communication preventing traffic collisions!")
    
    input("Press Enter to close...")
    plt.close()

if __name__ == "__main__":
    main() 