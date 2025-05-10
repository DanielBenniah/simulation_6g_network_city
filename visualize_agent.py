#!/usr/bin/env python3
"""
Visualization script to watch the trained agent control vehicles in the traffic environment.
This script loads the trained PPO model and runs episodes with visual feedback.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import time
import argparse
from stable_baselines3 import PPO
from city_traffic_env import CityTrafficEnv

class TrafficVisualizer:
    """Visualizes the traffic environment and agent behavior."""
    
    def __init__(self, env, model=None):
        self.env = env
        self.model = model
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        grid_width, grid_height = env.grid_size
        self.ax.set_xlim(0, grid_width)
        self.ax.set_ylim(0, grid_height)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Traffic Simulation - Trained Agent')
        
        # Vehicle patches for animation
        self.vehicle_patches = []
        self.info_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                     verticalalignment='top', fontsize=10,
                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Draw grid lines for intersections
        self._draw_grid()
    
    def _draw_grid(self):
        """Draw the city grid with intersections."""
        grid_width, grid_height = self.env.grid_size
        # Draw roads (horizontal and vertical lines)
        for i in range(grid_width + 1):
            # Vertical roads
            self.ax.axvline(x=i, color='gray', linewidth=2, alpha=0.5)
        for i in range(grid_height + 1):
            # Horizontal roads  
            self.ax.axhline(y=i, color='gray', linewidth=2, alpha=0.5)
        
        # Highlight intersections
        for i in range(1, grid_width):
            for j in range(1, grid_height):
                circle = patches.Circle((i, j), 0.1, color='red', alpha=0.3)
                self.ax.add_patch(circle)
    
    def _draw_vehicles(self, vehicles):
        """Draw vehicles on the grid."""
        # Clear previous vehicle patches
        for patch in self.vehicle_patches:
            patch.remove()
        self.vehicle_patches.clear()
        
        colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink']
        
        for i in range(len(vehicles)):
            # Skip inactive vehicles
            if vehicles[i, 6] == 0:  # vehicles[i, 6] is the active flag
                continue
                
            x, y = vehicles[i, 0], vehicles[i, 1]  # position
            color = colors[i % len(colors)]
            
            # Draw vehicle as a rectangle
            width, height = 0.15, 0.1
            vehicle_patch = patches.Rectangle(
                (x - width/2, y - height/2), width, height,
                color=color, alpha=0.8
            )
            self.ax.add_patch(vehicle_patch)
            self.vehicle_patches.append(vehicle_patch)
            
            # Add vehicle ID
            text = self.ax.text(x, y + 0.2, f'V{i}', ha='center', va='center', 
                               fontsize=8, fontweight='bold')
            self.vehicle_patches.append(text)
            
            # Draw destination
            dest_x, dest_y = vehicles[i, 4], vehicles[i, 5]  # route destination
            dest_patch = patches.Circle((dest_x, dest_y), 0.05, 
                                       color=color, alpha=0.3, linestyle='--')
            self.ax.add_patch(dest_patch)
            self.vehicle_patches.append(dest_patch)
    
    def _update_info_text(self, step, total_reward, vehicles, collisions, episode):
        """Update the information text display."""
        active_vehicles = int(np.sum(vehicles[:, 6]))  # Count active vehicles (where vehicles[i, 6] == 1)
        total_vehicles = len(vehicles)
        
        info = f"Episode: {episode}\n"
        info += f"Step: {step}\n"
        info += f"Total Reward: {total_reward:.2f}\n"
        info += f"Active Vehicles: {active_vehicles}/{total_vehicles}\n"
        info += f"Collisions: {collisions}"
        
        self.info_text.set_text(info)
    
    def run_episode(self, episode_num=1, max_steps=500, delay=0.1):
        """Run a single episode with visualization."""
        print(f"\n=== Episode {episode_num} ===")
        
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
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            step += 1
            
            # Count collisions
            if 'collision' in info and info['collision']:
                collisions += 1
            
            # Update visualization
            self._draw_vehicles(self.env.vehicles)
            self._update_info_text(step, total_reward, self.env.vehicles, collisions, episode_num)
            
            plt.draw()
            plt.pause(delay)
            
            if terminated or truncated:
                break
        
        print(f"Episode {episode_num} completed:")
        print(f"  Steps: {step}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Collisions: {collisions}")
        print(f"  Active vehicles remaining: {int(np.sum(self.env.vehicles[:, 6]))}")
        
        return total_reward, step, collisions

def main():
    parser = argparse.ArgumentParser(description='Visualize trained traffic agent')
    parser.add_argument('--model-path', type=str, default='ppo_city_traffic.zip',
                       help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per episode')
    parser.add_argument('--delay', type=float, default=0.1,
                       help='Delay between steps (seconds)')
    parser.add_argument('--no-model', action='store_true',
                       help='Run with random actions instead of trained model')
    parser.add_argument('--grid-size', type=int, default=5,
                       help='Size of the city grid')
    parser.add_argument('--num-vehicles', type=int, default=3,
                       help='Number of vehicles in simulation')
    
    args = parser.parse_args()
    
    # Create environment
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
            print(f"Loaded trained model from {args.model_path}")
        except Exception as e:
            print(f"Could not load model from {args.model_path}: {e}")
            print("Running with random actions instead")
    else:
        print("Running with random actions")
    
    # Create visualizer
    visualizer = TrafficVisualizer(env, model)
    
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
            
            # Wait a bit between episodes
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
    
    # Print summary
    if total_rewards:
        print(f"\n=== Summary ===")
        print(f"Episodes run: {len(total_rewards)}")
        print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"Average steps: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
        print(f"Total collisions: {sum(total_collisions)}")
        print(f"Average collisions per episode: {np.mean(total_collisions):.2f}")
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main() 