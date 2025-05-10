#!/usr/bin/env python3
"""
Realistic traffic visualizer showing vehicles that actually navigate to destinations
with proper pathfinding, traffic awareness, and realistic behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import argparse
from stable_baselines3 import PPO
from realistic_traffic_env import RealisticTrafficEnv

class RealisticTrafficVisualizer:
    """Visualizer for realistic traffic simulation with proper navigation."""
    
    def __init__(self, env, model=None):
        self.env = env
        self.model = model
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        grid_width, grid_height = env.grid_size
        self.ax.set_xlim(-0.5, grid_width - 0.5)
        self.ax.set_ylim(-0.5, grid_height - 0.5)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('🚗 Realistic Traffic - Vehicles Navigate to Destinations! 🎯', fontsize=16, fontweight='bold')
        
        # Vehicle patches
        self.vehicle_patches = []
        
        # Info displays
        self.info_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                     verticalalignment='top', fontsize=11,
                                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
        
        self.nav_text = self.ax.text(0.98, 0.98, '', transform=self.ax.transAxes, 
                                    verticalalignment='top', horizontalalignment='right', fontsize=10,
                                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
        
        # Draw grid
        self._draw_realistic_grid()
    
    def _draw_realistic_grid(self):
        """Draw realistic city grid with roads."""
        grid_width, grid_height = self.env.grid_size
        
        # Draw roads
        for i in range(grid_width):
            self.ax.axvline(x=i, color='darkgray', linewidth=5, alpha=0.7)
        for i in range(grid_height):
            self.ax.axhline(y=i, color='darkgray', linewidth=5, alpha=0.7)
        
        # Main intersection
        intersection_x, intersection_y = self.env.intersection_cell
        main_intersection = patches.Circle((intersection_x, intersection_y), 0.3, 
                                         color='red', alpha=0.4, linewidth=2, 
                                         edgecolor='darkred')
        self.ax.add_patch(main_intersection)
        
        # Label
        self.ax.text(intersection_x, intersection_y - 0.7, 'Main\nIntersection', 
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))
        
        # Other intersections
        for i in range(grid_width):
            for j in range(grid_height):
                if (i, j) != self.env.intersection_cell:
                    circle = patches.Circle((i, j), 0.08, color='orange', alpha=0.4)
                    self.ax.add_patch(circle)
    
    def _get_navigation_status(self, vehicle_idx, vehicles):
        """Get navigation status for a vehicle."""
        x, y = vehicles[vehicle_idx, 0], vehicles[vehicle_idx, 1]
        dest_x, dest_y = vehicles[vehicle_idx, 4], vehicles[vehicle_idx, 5]
        vx, vy = vehicles[vehicle_idx, 2], vehicles[vehicle_idx, 3]
        
        # Distance to destination
        dist_to_dest = np.sqrt((x - dest_x)**2 + (y - dest_y)**2)
        
        # Current direction
        if abs(vx) > abs(vy):
            current_dir = "→" if vx > 0 else "←"
        else:
            current_dir = "↓" if vy > 0 else "↑"
        
        # Desired direction
        dx, dy = dest_x - x, dest_y - y
        if abs(dx) > abs(dy):
            desired_dir = "→" if dx > 0 else "←"
        else:
            desired_dir = "↓" if dy > 0 else "↑"
        
        # Status
        if dist_to_dest < 0.7:
            status = "ARRIVED"
        elif current_dir == desired_dir:
            status = "ON_ROUTE"
        else:
            status = "TURNING"
        
        return status, dist_to_dest, current_dir, desired_dir
    
    def _draw_vehicles(self, vehicles):
        """Draw vehicles with navigation indicators."""
        # Clear previous patches
        for patch in self.vehicle_patches:
            patch.remove()
        self.vehicle_patches.clear()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                 '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
        
        active_count = 0
        total_speed = 0
        destinations_reached = 0
        
        for i in range(len(vehicles)):
            if vehicles[i, 6] == 0:  # Skip inactive vehicles
                continue
                
            active_count += 1
            x, y = vehicles[i, 0], vehicles[i, 1]
            vx, vy = vehicles[i, 2], vehicles[i, 3]
            dest_x, dest_y = vehicles[i, 4], vehicles[i, 5]
            speed = np.sqrt(vx**2 + vy**2)
            total_speed += speed
            
            color = colors[i % len(colors)]
            
            # Get navigation status
            nav_status, dist_to_dest, current_dir, desired_dir = self._get_navigation_status(i, vehicles)
            
            # Vehicle size based on speed
            base_size = 0.2
            size_factor = 1.0 + speed * 0.2
            width = base_size * size_factor
            height = base_size * 0.8
            
            # Color intensity based on navigation status
            if nav_status == "ARRIVED":
                alpha = 1.0
                edge_color = 'gold'
                edge_width = 3
            elif nav_status == "ON_ROUTE":
                alpha = 0.8
                edge_color = 'green'
                edge_width = 2
            else:  # TURNING
                alpha = 0.6
                edge_color = 'orange'
                edge_width = 2
            
            # Draw vehicle
            vehicle_rect = patches.Rectangle(
                (x - width/2, y - height/2), width, height,
                color=color, alpha=alpha, linewidth=edge_width, edgecolor=edge_color
            )
            self.ax.add_patch(vehicle_rect)
            self.vehicle_patches.append(vehicle_rect)
            
            # Vehicle info
            if nav_status == "ARRIVED":
                status_emoji = "🎯"
            elif nav_status == "ON_ROUTE":
                status_emoji = "✅"
            else:
                status_emoji = "🔄"
            
            vehicle_text = f'{status_emoji}\nV{i}\n{speed:.1f}'
            text = self.ax.text(x, y + 0.35, vehicle_text, ha='center', va='center', 
                               fontsize=8, fontweight='bold', color='white',
                               bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.9))
            self.vehicle_patches.append(text)
            
            # Direction arrow
            if speed > 0.1:
                arrow_length = 0.4
                arrow = patches.FancyArrowPatch(
                    (x, y), (x + vx/speed * arrow_length, y + vy/speed * arrow_length),
                    arrowstyle='->', mutation_scale=15, color='white', linewidth=2
                )
                self.ax.add_patch(arrow)
                self.vehicle_patches.append(arrow)
            
            # Destination with path line
            dest_circle = patches.Circle((dest_x, dest_y), 0.12, 
                                       color=color, alpha=0.6, linewidth=2, 
                                       edgecolor=color)
            self.ax.add_patch(dest_circle)
            self.vehicle_patches.append(dest_circle)
            
            # Destination label
            dest_text = self.ax.text(dest_x, dest_y, f'D{i}', ha='center', va='center', 
                                   fontsize=7, fontweight='bold', color='white')
            self.vehicle_patches.append(dest_text)
            
            # Path line (dashed if far, solid if close)
            line_style = '--' if dist_to_dest > 2 else '-'
            line_alpha = 0.6 if nav_status == "ON_ROUTE" else 0.3
            line = plt.Line2D([x, dest_x], [y, dest_y], color=color, alpha=line_alpha, 
                            linestyle=line_style, linewidth=2)
            self.ax.add_line(line)
            self.vehicle_patches.append(line)
            
            # Distance indicator
            if dist_to_dest < 0.7:
                destinations_reached += 1
        
        return active_count, total_speed, destinations_reached
    
    def _update_displays(self, step, total_reward, vehicles, collisions, episode, info, 
                        active_count, total_speed, destinations_reached):
        """Update information displays."""
        avg_speed = total_speed / max(active_count, 1)
        
        # Main info
        main_info = f"Episode: {episode} | Step: {step}\n"
        main_info += f"Total Reward: {total_reward:.2f}\n"
        main_info += f"Active Vehicles: {active_count}\n"
        main_info += f"Avg Speed: {avg_speed:.2f}\n"
        main_info += f"Collisions: {collisions}\n"
        main_info += f"Destinations Reached: {info.get('destinations_reached', 0)}"
        
        self.info_text.set_text(main_info)
        
        # Navigation details
        nav_details = "Navigation Status:\n"
        for i in range(min(len(vehicles), 6)):
            if vehicles[i, 6] == 0:
                nav_details += f"V{i}: INACTIVE\n"
            else:
                nav_status, dist_to_dest, current_dir, desired_dir = self._get_navigation_status(i, vehicles)
                
                if nav_status == "ARRIVED":
                    nav_details += f"V{i}: ARRIVED! 🎯\n"
                elif nav_status == "ON_ROUTE":
                    nav_details += f"V{i}: {current_dir} → D{i} ({dist_to_dest:.1f})\n"
                else:
                    nav_details += f"V{i}: {current_dir}→{desired_dir} TURN ({dist_to_dest:.1f})\n"
        
        self.nav_text.set_text(nav_details)
    
    def run_realistic_episode(self, episode_num=1, max_steps=300, delay=0.2):
        """Run a realistic traffic episode."""
        print(f"\n🎯 === REALISTIC TRAFFIC EPISODE {episode_num} ===")
        print("🔍 Watch for:")
        print("   • Vehicles navigating DIRECTLY toward destinations (D0, D1, etc.)")
        print("   • Status indicators: 🎯 (arrived) ✅ (on route) 🔄 (turning)")
        print("   • Vehicles slowing down when paths are blocked")
        print("   • New destinations assigned when vehicles arrive")
        
        obs, info = self.env.reset()
        total_reward = 0
        step = 0
        collisions = 0
        total_destinations_reached = 0
        
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
                print(f"   💥 Step {step}: Collision! Vehicles stopped")
            
            if step_info.get('destinations_reached', 0) > 0:
                total_destinations_reached += step_info['destinations_reached']
                print(f"   🎯 Step {step}: {step_info['destinations_reached']} vehicle(s) reached destination!")
            
            # Update visualization
            active_count, total_speed, current_arrivals = self._draw_vehicles(self.env.vehicles)
            self._update_displays(step, total_reward, self.env.vehicles, collisions, 
                                episode_num, step_info, active_count, total_speed, current_arrivals)
            
            plt.draw()
            plt.pause(delay)
            
            # Progress updates
            if step % 30 == 0:
                avg_speed = total_speed / max(active_count, 1)
                print(f"   📊 Step {step}: {active_count} vehicles, avg speed {avg_speed:.2f}, "
                     f"{total_destinations_reached} destinations reached")
            
            if terminated or truncated:
                break
        
        print(f"\n📈 REALISTIC EPISODE {episode_num} SUMMARY:")
        print(f"   Duration: {step} steps")
        print(f"   Total Reward: {total_reward:.2f}")
        print(f"   Collisions: {collisions}")
        print(f"   Destinations Reached: {total_destinations_reached}")
        print(f"   Final Active Vehicles: {active_count}")
        
        return total_reward, step, collisions, total_destinations_reached

def main():
    parser = argparse.ArgumentParser(description='Realistic traffic simulation with proper navigation')
    parser.add_argument('--episodes', type=int, default=2,
                       help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=200,
                       help='Maximum steps per episode')
    parser.add_argument('--delay', type=float, default=0.25,
                       help='Animation delay (seconds)')
    parser.add_argument('--grid-size', type=int, default=8,
                       help='Grid size')
    parser.add_argument('--num-vehicles', type=int, default=4,
                       help='Number of vehicles')
    parser.add_argument('--use-trained-model', action='store_true',
                       help='Use trained model instead of random actions')
    
    args = parser.parse_args()
    
    print("🎯" * 25)
    print("🚗 REALISTIC TRAFFIC SIMULATION 🚗")
    print("🎯" * 25)
    print(f"Grid: {args.grid_size}x{args.grid_size}")
    print(f"Vehicles: {args.num_vehicles}")
    print(f"Episodes: {args.episodes}")
    print("Features:")
    print("  ✅ Vehicles navigate to actual destinations")
    print("  ✅ Realistic pathfinding and turning")
    print("  ✅ Traffic awareness and collision avoidance")
    print("  ✅ Speed variations based on traffic conditions")
    print("🎯" * 25)
    
    # Create realistic environment
    env = RealisticTrafficEnv(
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
    visualizer = RealisticTrafficVisualizer(env, model)
    
    # Run episodes
    total_destinations = 0
    try:
        for episode in range(1, args.episodes + 1):
            reward, steps, collisions, destinations = visualizer.run_realistic_episode(
                episode_num=episode,
                max_steps=args.max_steps,
                delay=args.delay
            )
            total_destinations += destinations
            
            if episode < args.episodes:
                print(f"\n⏳ Waiting 3 seconds before next episode...")
                time.sleep(3)
    
    except KeyboardInterrupt:
        print("\n⚠️  Simulation interrupted by user")
    
    print(f"\n🏁 FINAL REALISTIC TRAFFIC SUMMARY:")
    print(f"Total destinations reached across all episodes: {total_destinations}")
    print(f"This demonstrates vehicles actually navigating to their intended destinations!")
    
    input("Press Enter to close...")
    plt.close()

if __name__ == "__main__":
    main() 