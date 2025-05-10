#!/usr/bin/env python3
"""
Demo visualizer for traffic simulation showing vehicles speeding up and slowing down.
Uses the DemoTrafficEnv for better visualization with longer episodes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import argparse
from stable_baselines3 import PPO
from demo_traffic_env import DemoTrafficEnv

class DemoTrafficVisualizer:
    """Demo visualizer optimized for showing speed variations and traffic behavior."""
    
    def __init__(self, env, model=None):
        self.env = env
        self.model = model
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        grid_width, grid_height = env.grid_size
        self.ax.set_xlim(-0.5, grid_width - 0.5)
        self.ax.set_ylim(-0.5, grid_height - 0.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('🚗 Traffic Demo - Watch Vehicles Speed Up & Slow Down! 🚗', fontsize=16, fontweight='bold')
        
        # Vehicle patches
        self.vehicle_patches = []
        self.speed_history = {}  # Track speed history for each vehicle
        
        # Info display
        self.info_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                     verticalalignment='top', fontsize=11,
                                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
        
        self.speed_text = self.ax.text(0.98, 0.98, '', transform=self.ax.transAxes, 
                                      verticalalignment='top', horizontalalignment='right', fontsize=10,
                                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
        
        # Draw enhanced grid
        self._draw_demo_grid()
    
    def _draw_demo_grid(self):
        """Draw demo-friendly grid with clear roads and intersections."""
        grid_width, grid_height = self.env.grid_size
        
        # Draw roads as wide gray lines
        for i in range(grid_width):
            self.ax.axvline(x=i, color='gray', linewidth=6, alpha=0.6)
        for i in range(grid_height):
            self.ax.axhline(y=i, color='gray', linewidth=6, alpha=0.6)
        
        # Main intersection
        intersection_x, intersection_y = self.env.intersection_cell
        main_intersection = patches.Circle((intersection_x, intersection_y), 0.4, 
                                         color='red', alpha=0.5, linewidth=3, 
                                         edgecolor='darkred', linestyle='--')
        self.ax.add_patch(main_intersection)
        
        # Label the main intersection
        self.ax.text(intersection_x, intersection_y - 0.8, '🚦 Main Intersection', 
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        
        # Other intersections
        for i in range(grid_width):
            for j in range(grid_height):
                if (i, j) != self.env.intersection_cell:
                    circle = patches.Circle((i, j), 0.12, color='orange', alpha=0.4)
                    self.ax.add_patch(circle)
    
    def _draw_vehicles(self, vehicles):
        """Draw vehicles with clear speed visualization."""
        # Clear previous patches
        for patch in self.vehicle_patches:
            patch.remove()
        self.vehicle_patches.clear()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                 '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
        
        active_count = 0
        total_speed = 0
        
        for i in range(len(vehicles)):
            if vehicles[i, 6] == 0:  # Skip inactive vehicles
                continue
                
            active_count += 1
            x, y = vehicles[i, 0], vehicles[i, 1]
            vx, vy = vehicles[i, 2], vehicles[i, 3]
            speed = np.sqrt(vx**2 + vy**2)
            total_speed += speed
            
            # Track speed history
            if i not in self.speed_history:
                self.speed_history[i] = []
            self.speed_history[i].append(speed)
            if len(self.speed_history[i]) > 10:  # Keep last 10 speeds
                self.speed_history[i].pop(0)
            
            color = colors[i % len(colors)]
            
            # Vehicle size based on speed (bigger = faster)
            base_size = 0.25
            size_factor = 1.0 + speed * 0.3
            width = base_size * size_factor
            height = base_size * 0.7
            
            # Color intensity based on speed
            alpha = 0.7 + min(speed * 0.15, 0.3)
            
            # Draw vehicle body
            vehicle_rect = patches.Rectangle(
                (x - width/2, y - height/2), width, height,
                color=color, alpha=alpha, linewidth=2, edgecolor='black'
            )
            self.ax.add_patch(vehicle_rect)
            self.vehicle_patches.append(vehicle_rect)
            
            # Speed indicator text
            speed_category = "🐌" if speed < 0.5 else "🚶" if speed < 1.0 else "🏃" if speed < 1.5 else "🚀"
            speed_text = f'{speed_category}\nV{i}\n{speed:.1f}'
            text = self.ax.text(x, y + 0.4, speed_text, ha='center', va='center', 
                               fontsize=8, fontweight='bold', color='white',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.9))
            self.vehicle_patches.append(text)
            
            # Velocity arrow (direction and magnitude)
            if speed > 0.1:
                arrow_length = min(speed * 0.4, 1.0)
                arrow = patches.FancyArrowPatch(
                    (x, y), (x + vx/speed * arrow_length, y + vy/speed * arrow_length),
                    arrowstyle='->', mutation_scale=20, color='white', linewidth=3,
                    alpha=0.8
                )
                self.ax.add_patch(arrow)
                self.vehicle_patches.append(arrow)
            
            # Destination
            dest_x, dest_y = vehicles[i, 4], vehicles[i, 5]
            dest_circle = patches.Circle((dest_x, dest_y), 0.1, 
                                       color=color, alpha=0.5, linestyle=':', 
                                       linewidth=2, edgecolor=color)
            self.ax.add_patch(dest_circle)
            self.vehicle_patches.append(dest_circle)
            
            # Line to destination
            line = plt.Line2D([x, dest_x], [y, dest_y], color=color, alpha=0.4, 
                            linestyle=':', linewidth=1)
            self.ax.add_line(line)
            self.vehicle_patches.append(line)
        
        return active_count, total_speed
    
    def _update_displays(self, step, total_reward, vehicles, collisions, episode, info, active_count, total_speed):
        """Update information displays."""
        avg_speed = total_speed / max(active_count, 1)
        
        # Main info
        main_info = f"🎮 Episode: {episode} | Step: {step}\n"
        main_info += f"💰 Reward: {total_reward:.2f}\n"
        main_info += f"🚗 Active Vehicles: {active_count}\n"
        main_info += f"💥 Collisions: {collisions}\n"
        main_info += f"⚡ Avg Speed: {avg_speed:.2f}\n"
        main_info += f"📡 Messages: {info.get('messages_sent', 0)}"
        
        self.info_text.set_text(main_info)
        
        # Speed details
        speed_details = "🏁 Speed Details:\n"
        for i in range(min(len(vehicles), 6)):
            if vehicles[i, 6] == 0:
                speed_details += f"V{i}: INACTIVE\n"
            else:
                speed = np.sqrt(vehicles[i, 2]**2 + vehicles[i, 3]**2)
                if i in self.speed_history and len(self.speed_history[i]) > 1:
                    trend = "📈" if speed > self.speed_history[i][-2] else "📉" if speed < self.speed_history[i][-2] else "➡️"
                else:
                    trend = "➡️"
                    
                status = "STOPPED" if speed < 0.1 else "SLOW" if speed < 0.8 else "FAST"
                speed_details += f"V{i}: {trend} {status} ({speed:.1f})\n"
        
        self.speed_text.set_text(speed_details)
    
    def run_demo_episode(self, episode_num=1, max_steps=300, delay=0.2):
        """Run a demo episode with enhanced feedback."""
        print(f"\n🚀 === DEMO EPISODE {episode_num} ===")
        print("🎯 Watch for:")
        print("   • Vehicles getting BIGGER and BRIGHTER when speeding up")
        print("   • Vehicles getting SMALLER and DIMMER when slowing down")
        print("   • Speed indicators: 🐌 (slow) 🚶 (medium) 🏃 (fast) 🚀 (very fast)")
        print("   • White arrows showing direction and speed")
        
        obs, info = self.env.reset()
        total_reward = 0
        step = 0
        collisions = 0
        speed_changes = 0
        
        plt.ion()
        
        while step < max_steps:
            # Get action
            if self.model:
                action, _ = self.model.predict(obs, deterministic=True)
            else:
                action = self.env.action_space.sample()
            
            # Take step
            obs, reward, terminated, truncated, step_info = self.env.step(action)
            total_reward += reward
            step += 1
            
            # Count events
            if 'collisions' in step_info and step_info['collisions']:
                collisions += len(step_info['collisions'])
                print(f"   💥 Step {step}: Collision! Vehicles slowed down")
            
            # Update visualization
            active_count, total_speed = self._draw_vehicles(self.env.vehicles)
            self._update_displays(step, total_reward, self.env.vehicles, collisions, 
                                episode_num, step_info, active_count, total_speed)
            
            plt.draw()
            plt.pause(delay)
            
            # Print interesting speed changes
            if step % 20 == 0:
                avg_speed = total_speed / max(active_count, 1)
                print(f"   📊 Step {step}: {active_count} vehicles, avg speed {avg_speed:.2f}")
            
            if terminated or truncated:
                print(f"   🏁 Episode ended at step {step}")
                break
        
        print(f"\n📈 DEMO EPISODE {episode_num} SUMMARY:")
        print(f"   Duration: {step} steps")
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Collisions: {collisions}")
        print(f"   Final Active Vehicles: {active_count}")
        
        return total_reward, step, collisions

def main():
    parser = argparse.ArgumentParser(description='Demo traffic visualization with speed variations')
    parser.add_argument('--episodes', type=int, default=2,
                       help='Number of demo episodes')
    parser.add_argument('--max-steps', type=int, default=250,
                       help='Maximum steps per episode')
    parser.add_argument('--delay', type=float, default=0.2,
                       help='Animation delay (seconds)')
    parser.add_argument('--grid-size', type=int, default=10,
                       help='Grid size')
    parser.add_argument('--num-vehicles', type=int, default=5,
                       help='Number of vehicles')
    parser.add_argument('--use-trained-model', action='store_true',
                       help='Use trained model instead of random actions')
    
    args = parser.parse_args()
    
    print("🚗" * 20)
    print("🎬 TRAFFIC SPEED VARIATION DEMO")
    print("🚗" * 20)
    print(f"🏁 Grid: {args.grid_size}x{args.grid_size}")
    print(f"🚙 Vehicles: {args.num_vehicles}")
    print(f"🎮 Episodes: {args.episodes}")
    print(f"⏱️  Animation Speed: {args.delay}s per step")
    print("🚗" * 20)
    
    # Create demo environment
    env = DemoTrafficEnv(
        grid_size=(args.grid_size, args.grid_size),
        max_vehicles=args.num_vehicles,
        multi_agent=False,
        debug=False
    )
    
    # Load model if requested
    model = None
    if args.use_trained_model:
        try:
            model = PPO.load('ppo_city_traffic.zip')
            print("✅ Using trained PPO model")
        except Exception as e:
            print(f"❌ Could not load model: {e}")
            print("🎲 Using random actions instead")
    else:
        print("🎲 Using random actions for demo")
    
    # Create visualizer
    visualizer = DemoTrafficVisualizer(env, model)
    
    # Run demo episodes
    try:
        for episode in range(1, args.episodes + 1):
            reward, steps, collisions = visualizer.run_demo_episode(
                episode_num=episode,
                max_steps=args.max_steps,
                delay=args.delay
            )
            
            if episode < args.episodes:
                print(f"\n⏳ Waiting 3 seconds before next episode...")
                time.sleep(3)
    
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    
    print("\n🎉 Demo completed! Thanks for watching!")
    input("Press Enter to close...")
    plt.close()

if __name__ == "__main__":
    main() 