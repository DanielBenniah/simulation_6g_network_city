#!/usr/bin/env python3
"""
Enhanced visualization script for the traffic simulation with better settings
to show interesting traffic behavior including vehicles slowing down, speeding up,
and complex interactions at intersections.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time
import argparse
from stable_baselines3 import PPO
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from environments.city_traffic_env import CityTrafficEnv

class EnhancedTrafficVisualizer:
    """Enhanced visualizer for traffic environment with better visual feedback."""
    
    def __init__(self, env, model=None):
        self.env = env
        self.model = model
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        grid_width, grid_height = env.grid_size
        self.ax.set_xlim(-0.5, grid_width - 0.5)
        self.ax.set_ylim(-0.5, grid_height - 0.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Enhanced Traffic Simulation - Watch Vehicles Speed Up & Slow Down', fontsize=14)
        
        # Vehicle patches for animation
        self.vehicle_patches = []
        self.velocity_arrows = []
        
        # Info panels
        self.info_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                     verticalalignment='top', fontsize=10,
                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        self.vehicle_info_text = self.ax.text(0.98, 0.98, '', transform=self.ax.transAxes, 
                                            verticalalignment='top', horizontalalignment='right', fontsize=9,
                                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Draw grid
        self._draw_enhanced_grid()
    
    def _draw_enhanced_grid(self):
        """Draw an enhanced city grid with roads and intersections."""
        grid_width, grid_height = self.env.grid_size
        
        # Draw roads as thick lines
        for i in range(grid_width):
            # Vertical roads
            self.ax.axvline(x=i, color='darkgray', linewidth=4, alpha=0.7)
        for i in range(grid_height):
            # Horizontal roads  
            self.ax.axhline(y=i, color='darkgray', linewidth=4, alpha=0.7)
        
        # Highlight major intersections
        intersection_x, intersection_y = self.env.intersection_cell
        major_intersection = patches.Circle((intersection_x, intersection_y), 0.3, 
                                          color='red', alpha=0.4, linewidth=2, edgecolor='darkred')
        self.ax.add_patch(major_intersection)
        
        # Add labels for the main intersection
        self.ax.text(intersection_x, intersection_y - 0.6, 'Main\nIntersection', 
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Draw other intersections as smaller circles
        for i in range(grid_width):
            for j in range(grid_height):
                if (i, j) != self.env.intersection_cell:
                    circle = patches.Circle((i, j), 0.1, color='orange', alpha=0.3)
                    self.ax.add_patch(circle)
    
    def _draw_vehicles(self, vehicles):
        """Draw vehicles with velocity indicators and enhanced visuals."""
        # Clear previous patches
        for patch in self.vehicle_patches + self.velocity_arrows:
            patch.remove()
        self.vehicle_patches.clear()
        self.velocity_arrows.clear()
        
        colors = ['blue', 'green', 'red', 'purple', 'brown', 'pink', 'orange', 'cyan', 'magenta', 'yellow']
        
        for i in range(len(vehicles)):
            # Skip inactive vehicles
            if vehicles[i, 6] == 0:
                continue
                
            x, y = vehicles[i, 0], vehicles[i, 1]
            vx, vy = vehicles[i, 2], vehicles[i, 3]
            color = colors[i % len(colors)]
            
            # Calculate speed for visual feedback
            speed = np.sqrt(vx**2 + vy**2)
            
            # Draw vehicle as a rectangle with size based on speed
            base_size = 0.2
            size_multiplier = 1 + speed * 0.1  # Larger when moving faster
            width = base_size * size_multiplier
            height = base_size * 0.8
            
            # Color intensity based on speed (brighter = faster)
            alpha = 0.6 + min(speed * 0.1, 0.4)
            
            vehicle_patch = patches.Rectangle(
                (x - width/2, y - height/2), width, height,
                color=color, alpha=alpha, linewidth=2, edgecolor='black'
            )
            self.ax.add_patch(vehicle_patch)
            self.vehicle_patches.append(vehicle_patch)
            
            # Add vehicle ID and speed
            speed_text = f'V{i}\n{speed:.1f}'
            text = self.ax.text(x, y + 0.35, speed_text, ha='center', va='center', 
                               fontsize=8, fontweight='bold', color='white',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))
            self.vehicle_patches.append(text)
            
            # Draw velocity arrow
            if speed > 0.1:  # Only draw arrow if moving
                arrow_scale = 0.3
                arrow = patches.FancyArrowPatch(
                    (x, y), (x + vx * arrow_scale, y + vy * arrow_scale),
                    arrowstyle='->', mutation_scale=15, color='white', linewidth=2
                )
                self.ax.add_patch(arrow)
                self.velocity_arrows.append(arrow)
            
            # Draw destination with line
            dest_x, dest_y = vehicles[i, 4], vehicles[i, 5]
            
            # Destination marker
            dest_patch = patches.Circle((dest_x, dest_y), 0.08, 
                                       color=color, alpha=0.4, linestyle='--', 
                                       linewidth=2, edgecolor=color)
            self.ax.add_patch(dest_patch)
            self.vehicle_patches.append(dest_patch)
            
            # Line to destination
            line = plt.Line2D([x, dest_x], [y, dest_y], color=color, alpha=0.3, linestyle=':', linewidth=1)
            self.ax.add_line(line)
            self.vehicle_patches.append(line)
    
    def _update_info_displays(self, step, total_reward, vehicles, collisions, episode, info):
        """Update both info displays with detailed information."""
        active_vehicles = int(np.sum(vehicles[:, 6]))
        total_vehicles = len(vehicles)
        
        # Main info panel
        main_info = f"Episode: {episode}\n"
        main_info += f"Step: {step}\n"
        main_info += f"Total Reward: {total_reward:.2f}\n"
        main_info += f"Active Vehicles: {active_vehicles}/{total_vehicles}\n"
        main_info += f"Collisions: {collisions}\n"
        main_info += f"Intersection Denials: {len(info.get('intersection_denials', []))}\n"
        main_info += f"Messages: {info.get('messages_sent', 0)}/{info.get('messages_delivered', 0)}"
        
        self.info_text.set_text(main_info)
        
        # Vehicle details panel
        vehicle_details = "Vehicle Details:\n"
        for i in range(min(len(vehicles), 8)):  # Show up to 8 vehicles
            if vehicles[i, 6] == 0:
                vehicle_details += f"V{i}: INACTIVE\n"
            else:
                x, y = vehicles[i, 0], vehicles[i, 1]
                vx, vy = vehicles[i, 2], vehicles[i, 3]
                dest_x, dest_y = vehicles[i, 4], vehicles[i, 5]
                speed = np.sqrt(vx**2 + vy**2)
                dist_to_dest = np.sqrt((x - dest_x)**2 + (y - dest_y)**2)
                
                status = "MOVING" if speed > 0.1 else "STOPPED"
                vehicle_details += f"V{i}: {status} ({speed:.1f})\n"
                vehicle_details += f"     Pos: ({x:.1f},{y:.1f})\n"
                vehicle_details += f"     Dest: {dist_to_dest:.1f} away\n"
        
        self.vehicle_info_text.set_text(vehicle_details)
    
    def run_episode(self, episode_num=1, max_steps=500, delay=0.2):
        """Run a single episode with enhanced visualization."""
        print(f"\n=== Enhanced Episode {episode_num} ===")
        print("Watch for vehicles speeding up (brighter colors) and slowing down at intersections!")
        
        obs, info = self.env.reset()
        total_reward = 0
        step = 0
        collisions = 0
        
        plt.ion()  # Interactive mode
        
        while step < max_steps:
            # Get action from model or random
            if self.model:
                action, _ = self.model.predict(obs, deterministic=True)
            else:
                action = self.env.action_space.sample()
            
            # Take step
            obs, reward, terminated, truncated, step_info = self.env.step(action)
            total_reward += reward
            step += 1
            
            # Count collisions
            if 'collisions' in step_info and step_info['collisions']:
                collisions += len(step_info['collisions'])
            
            # Update visualization
            self._draw_vehicles(self.env.vehicles)
            self._update_info_displays(step, total_reward, self.env.vehicles, collisions, episode_num, step_info)
            
            plt.draw()
            plt.pause(delay)
            
            # Print interesting events
            if step_info.get('intersection_denials'):
                print(f"  Step {step}: Vehicles {step_info['intersection_denials']} stopped at intersection")
            if step_info.get('collisions'):
                print(f"  Step {step}: Collision detected! {step_info['collisions']}")
            
            if terminated or truncated:
                break
        
        print(f"Enhanced Episode {episode_num} completed:")
        print(f"  Steps: {step}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Collisions: {collisions}")
        print(f"  Final active vehicles: {int(np.sum(self.env.vehicles[:, 6]))}")
        
        return total_reward, step, collisions

def main():
    parser = argparse.ArgumentParser(description='Enhanced visualization of traffic simulation')
    parser.add_argument('--model-path', type=str, default='ppo_city_traffic.zip',
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=200,
                       help='Maximum steps per episode')
    parser.add_argument('--delay', type=float, default=0.15,
                       help='Delay between steps (seconds)')
    parser.add_argument('--no-model', action='store_true',
                       help='Run with random actions instead of trained model')
    parser.add_argument('--grid-size', type=int, default=8,
                       help='Size of the city grid (larger = more interesting)')
    parser.add_argument('--num-vehicles', type=int, default=6,
                       help='Number of vehicles in simulation')
    
    args = parser.parse_args()
    
    print("🚗 Enhanced Traffic Simulation Visualizer 🚗")
    print("=" * 50)
    print(f"Grid Size: {args.grid_size}x{args.grid_size}")
    print(f"Vehicles: {args.num_vehicles}")
    print(f"Episodes: {args.episodes}")
    print("=" * 50)
    
    # Create environment with better settings
    env = CityTrafficEnv(
        grid_size=(args.grid_size, args.grid_size),
        max_vehicles=args.num_vehicles,
        multi_agent=False,  # Single agent mode for visualization
        debug=False
    )
    
    # Load model
    model = None
    if not args.no_model:
        try:
            model = PPO.load(args.model_path)
            print(f"✅ Loaded trained model from {args.model_path}")
        except Exception as e:
            print(f"❌ Could not load model from {args.model_path}: {e}")
            print("🎲 Running with random actions instead")
    else:
        print("🎲 Running with random actions")
    
    # Create enhanced visualizer
    visualizer = EnhancedTrafficVisualizer(env, model)
    
    # Run episodes
    total_rewards = []
    total_steps = []
    total_collisions = []
    
    try:
        for episode in range(1, args.episodes + 1):
            reward, steps, collisions = visualizer.run_episode(
                episode_num=episode,
                max_steps=args.max_steps,
                delay=args.delay
            )
            total_rewards.append(reward)
            total_steps.append(steps)
            total_collisions.append(collisions)
            
            # Wait between episodes
            if episode < args.episodes:
                print(f"\nWaiting 2 seconds before next episode...")
                time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n⚠️  Visualization interrupted by user")
    
    # Print summary
    if total_rewards:
        print(f"\n🏁 FINAL SUMMARY 🏁")
        print(f"Episodes completed: {len(total_rewards)}")
        print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"Average episode length: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f} steps")
        print(f"Total collisions across all episodes: {sum(total_collisions)}")
        print(f"Average collisions per episode: {np.mean(total_collisions):.2f}")
        
        if np.mean(total_steps) > 50:
            print("✅ Good! Episodes are running long enough to see interesting behavior")
        else:
            print("⚠️  Episodes are quite short. Try larger grid or fewer vehicles")
    
    plt.ioff()
    input("\nPress Enter to close the visualization window...")
    plt.close()

if __name__ == "__main__":
    main() 