#!/usr/bin/env python3
"""
Smart Highway Visualizer with 6G Communication
==============================================

Visualizer for the Smart Highway Environment that shows:
- Directional lanes with proper vehicle movement (X or Y axis only)
- 6G V2V/V2I communication at intersections
- Clear intersection definitions for traffic light integration
- Journey time tracking and traffic analytics
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
from environments.smart_highway_env import SmartHighwayEnv

class SmartHighwayVisualizer:
    """Enhanced visualizer for Smart Highway with 6G communication."""
    
    def __init__(self, env, model=None):
        self.env = env
        self.model = model
        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        
        grid_width, grid_height = env.grid_size
        self.ax.set_xlim(-0.8, grid_width-0.2)
        self.ax.set_ylim(-0.8, grid_height-0.2)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('🛣️ Smart Highway with 6G V2V/V2I Communication 📡', 
                         fontsize=18, fontweight='bold')
        
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
        
        self.intersection_text = self.ax.text(0.98, 0.02, '', transform=self.ax.transAxes, 
                                            verticalalignment='bottom', horizontalalignment='right', fontsize=10,
                                            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9))
        
        # Draw highway infrastructure
        self._draw_highway_infrastructure()
        
        # Direction colors for vehicles (only 2 directions now)
        self.direction_colors = {
            0: '#FF6B6B',  # L2R - Red (Horizontal Traffic)
            1: '#45B7D1',  # T2B - Blue (Vertical Traffic)
        }
    
    def _draw_highway_infrastructure(self):
        """Draw highway lanes, intersections, and 6G infrastructure."""
        grid_width, grid_height = self.env.grid_size
        
        # Draw lane separators
        for i in range(grid_width + 1):
            # Vertical lane separators with dashed lines for lanes
            if i > 0 and i < grid_width:
                self.ax.axvline(x=i-0.5, color='gray', linewidth=1, alpha=0.3)
                # Lane markers
                for y in np.arange(0, grid_height, 0.5):
                    self.ax.plot([i-0.7, i-0.3], [y, y], 'w-', linewidth=1, alpha=0.5)
                    self.ax.plot([i-0.3, i+0.3], [y, y], 'w-', linewidth=1, alpha=0.5)
        
        for i in range(grid_height + 1):
            # Horizontal lane separators
            if i > 0 and i < grid_height:
                self.ax.axhline(y=i-0.5, color='gray', linewidth=1, alpha=0.3)
                # Lane markers
                for x in np.arange(0, grid_width, 0.5):
                    self.ax.plot([x, x], [i-0.7, i-0.3], 'w-', linewidth=1, alpha=0.5)
                    self.ax.plot([x, x], [i-0.3, i+0.3], 'w-', linewidth=1, alpha=0.5)
        
        # Draw major intersections with enhanced 6G infrastructure
        intersection_info = self.env.get_intersection_info()
        for intersection in intersection_info['intersections']:
            x, y = intersection['position']
            intersection_type = intersection['type']
            size = intersection.get('size', 0.6)
            
            if intersection_type == 'main':
                # Main intersection with enhanced visualization
                # Safety zone
                safety_circle = patches.Circle((y, x), 1.5, color='orange', alpha=0.2, 
                                             linewidth=1, edgecolor='orange', linestyle='--')
                self.ax.add_patch(safety_circle)
                
                # Main intersection core
                circle = patches.Circle((y, x), size, color='red', alpha=0.8, 
                                      linewidth=4, edgecolor='darkred')
                self.ax.add_patch(circle)
                self.ax.text(y, x, '📡\n6G\nMAIN\n🚦', ha='center', va='center', 
                           fontsize=8, fontweight='bold', color='white')
            else:
                # Major intersection
                # Safety zone
                safety_circle = patches.Circle((y, x), 1.2, color='yellow', alpha=0.15, 
                                             linewidth=1, edgecolor='yellow', linestyle='--')
                self.ax.add_patch(safety_circle)
                
                # Major intersection core
                circle = patches.Circle((y, x), size, color='orange', alpha=0.7, 
                                      linewidth=3, edgecolor='darkorange')
                self.ax.add_patch(circle)
                self.ax.text(y, x, '📡\n6G\n🚦', ha='center', va='center', 
                           fontsize=7, fontweight='bold', color='white')
            
            # Draw lane approach indicators
            self._draw_intersection_lanes(x, y, size)
        
        # Draw lane direction indicators
        self._draw_lane_indicators()
        
        # Add legend
        legend_elements = [
            patches.Circle((0, 0), 0.1, color='red', alpha=0.8, label='🏛️ Main 6G Intersection'),
            patches.Circle((0, 0), 0.1, color='orange', alpha=0.7, label='📡 Major 6G Intersection'),
            patches.Rectangle((0, 0), 0.1, 0.1, color='#FF6B6B', alpha=0.8, label='🚗 Horizontal Traffic (L2R)'),
            patches.Rectangle((0, 0), 0.1, 0.1, color='#45B7D1', alpha=0.8, label='🚐 Vertical Traffic (T2B)'),
            patches.Circle((0, 0), 0.05, color='orange', alpha=0.2, label='🛡️ Safety Zones'),
        ]
        self.ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.85))
    
    def _draw_lane_indicators(self):
        """Draw lane direction indicators exactly where vehicles travel."""
        grid_width, grid_height = self.env.grid_size
        
        # Define exact lane positions that match vehicle spawning
        lane_positions = [2, 4, 6, 8]  # Vehicles only travel on these lanes
        
        # Horizontal lane indicators (L2R only) - exactly where vehicles travel
        for y in lane_positions:
            # Multiple L2R lane indicators at exact vehicle positions
            for lane_offset in [-0.4, -0.2, 0.0, 0.2, 0.4]:
                self.ax.annotate('', xy=(grid_width*0.9, y+lane_offset), xytext=(grid_width*0.1, y+lane_offset),
                               arrowprops=dict(arrowstyle='->', color='#FF6B6B', lw=1.5, alpha=0.5))
                
                # Draw lane boundaries
                if lane_offset in [-0.4, 0.4]:  # Outer lanes
                    self.ax.axhline(y=y+lane_offset+0.1, color='white', linewidth=1, alpha=0.3, linestyle='-')
                    self.ax.axhline(y=y+lane_offset-0.1, color='white', linewidth=1, alpha=0.3, linestyle='-')
        
        # Vertical lane indicators (T2B only) - exactly where vehicles travel
        for x in lane_positions:
            # Multiple T2B lane indicators at exact vehicle positions
            for lane_offset in [-0.4, -0.2, 0.0, 0.2, 0.4]:
                self.ax.annotate('', xy=(x+lane_offset, grid_height*0.9), xytext=(x+lane_offset, grid_height*0.1),
                               arrowprops=dict(arrowstyle='->', color='#45B7D1', lw=1.5, alpha=0.5))
                
                # Draw lane boundaries
                if lane_offset in [-0.4, 0.4]:  # Outer lanes
                    self.ax.axvline(x=x+lane_offset+0.1, color='white', linewidth=1, alpha=0.3, linestyle='-')
                    self.ax.axvline(x=x+lane_offset-0.1, color='white', linewidth=1, alpha=0.3, linestyle='-')
    
    def _draw_intersection_lanes(self, x, y, size):
        """Draw approach lanes for intersections."""
        # Draw approach lane markers
        lane_offsets = [-0.4, -0.2, 0.0, 0.2, 0.4]
        
        # Horizontal approach lanes
        for offset in lane_offsets:
            # Entry markers
            entry_marker = patches.Rectangle((y-size-0.3, x+offset-0.05), 0.2, 0.1, 
                                           color='white', alpha=0.8)
            self.ax.add_patch(entry_marker)
            # Exit markers  
            exit_marker = patches.Rectangle((y+size+0.1, x+offset-0.05), 0.2, 0.1, 
                                          color='white', alpha=0.8)
            self.ax.add_patch(exit_marker)
        
        # Vertical approach lanes
        for offset in lane_offsets:
            # Entry markers
            entry_marker = patches.Rectangle((y+offset-0.05, x-size-0.3), 0.1, 0.2, 
                                           color='white', alpha=0.8)
            self.ax.add_patch(entry_marker)
            # Exit markers
            exit_marker = patches.Rectangle((y+offset-0.05, x+size+0.1), 0.1, 0.2, 
                                          color='white', alpha=0.8)
            self.ax.add_patch(exit_marker)
    
    def _draw_vehicles(self, vehicles):
        """Draw vehicles with proper lane positioning and direction indicators."""
        # Clear previous vehicle patches
        for patch in self.vehicle_patches:
            patch.remove()
        self.vehicle_patches.clear()
        
        active_count = 0
        total_speed = 0
        direction_counts = {0: 0, 1: 0}  # Only 2 directions now
        
        for i in range(len(vehicles)):
            if vehicles[i, 6] == 0:  # Skip inactive vehicles
                continue
                
            active_count += 1
            x, y = vehicles[i, 0], vehicles[i, 1]  # position
            vx, vy = vehicles[i, 2], vehicles[i, 3]  # velocity
            direction = int(vehicles[i, 4])
            lane_offset = vehicles[i, 5]
            speed = np.sqrt(vx**2 + vy**2)
            total_speed += speed
            direction_counts[direction] += 1
            
            color = self.direction_colors[direction]
            
            # Apply lane offset for proper lane positioning
            if direction == 0:  # Horizontal movement (L2R)
                visual_x, visual_y = x, y + lane_offset
            else:  # Vertical movement (T2B)
                visual_x, visual_y = x + lane_offset, y
            
            # Vehicle size based on speed
            base_size = 0.15
            size_factor = 1.0 + speed * 0.3
            vehicle_size = base_size * size_factor
            
            # Draw vehicle as rectangle with direction
            if direction == 0:  # Horizontal vehicles (wider)
                width, height = vehicle_size * 1.5, vehicle_size
            else:  # Vertical vehicles (taller)
                width, height = vehicle_size, vehicle_size * 1.5
            
            vehicle_patch = patches.Rectangle((visual_y - width/2, visual_x - height/2), 
                                            width, height, color=color, alpha=0.8, 
                                            linewidth=2, edgecolor='black')
            self.ax.add_patch(vehicle_patch)
            self.vehicle_patches.append(vehicle_patch)
            
            # Vehicle ID and speed
            vehicle_text = f'V{i}\n{speed:.1f}'
            text_color = 'white' if direction == 1 else 'black'  # White for vertical, black for horizontal
            text = self.ax.text(visual_y, visual_x, vehicle_text, ha='center', va='center', 
                               fontsize=7, fontweight='bold', color=text_color)
            self.vehicle_patches.append(text)
            
            # Draw destination marker
            dest_x, dest_y = vehicles[i, 8], vehicles[i, 9]
            dest_patch = patches.Rectangle((dest_y-0.08, dest_x-0.08), 0.16, 0.16, 
                                         color=color, alpha=0.4, linestyle='--', 
                                         linewidth=2, fill=False)
            self.ax.add_patch(dest_patch)
            self.vehicle_patches.append(dest_patch)
            
            # Draw velocity arrow
            if speed > 0.2:
                arrow_scale = 0.6
                arrow = patches.FancyArrowPatch((visual_y, visual_x), 
                                              (visual_y + vy*arrow_scale, visual_x + vx*arrow_scale),
                                              arrowstyle='->', mutation_scale=12, 
                                              color='white', linewidth=2, alpha=0.9)
                self.ax.add_patch(arrow)
                self.vehicle_patches.append(arrow)
        
        return active_count, total_speed, direction_counts
    
    def _update_displays(self, step, total_reward, active_count, total_speed, direction_counts, step_info, episode):
        """Update all information displays."""
        avg_speed = total_speed / max(active_count, 1)
        
        # Main simulation info
        main_info = f"🛣️ SMART HIGHWAY - 6G V2V/V2I SYSTEM\n"
        main_info += f"Episode: {episode} | Step: {step}\n"
        main_info += f"Agent Reward: {total_reward:.2f}\n"
        main_info += f"Active Vehicles: {active_count}\n"
        main_info += f"Average Speed: {avg_speed:.2f}\n"
        main_info += f"Horizontal (L2R): {direction_counts[0]}\n"
        main_info += f"Vertical (T2B): {direction_counts[1]}"
        
        self.info_text.set_text(main_info)
        
        # 6G Communication statistics
        messages_sent = step_info.get('messages_sent', 0)
        messages_delivered = step_info.get('messages_delivered', 0)
        reservations = len(step_info.get('intersection_reservations', []))
        collisions_prevented = len(step_info.get('collisions_prevented', []))
        actual_collisions = len(step_info.get('actual_collisions', []))
        delivery_rate = (messages_delivered / max(messages_sent, 1)) * 100
        
        comm_info = f"📡 6G COMMUNICATION STATUS\n"
        comm_info += f"Messages Sent: {messages_sent}\n"
        comm_info += f"Messages Delivered: {messages_delivered}\n"
        comm_info += f"Delivery Rate: {delivery_rate:.1f}%\n"
        comm_info += f"🚦 Intersection Reservations: {reservations}\n"
        comm_info += f"🛡️  6G Prevented: {collisions_prevented}\n"
        comm_info += f"💥 Actual Collisions: {actual_collisions}\n"
        comm_info += f"✅ 6G System: {'LEARNING' if actual_collisions == 0 else 'ACTIVE'}"
        
        self.comm_text.set_text(comm_info)
        
        # Journey and intersection statistics
        stats = self.env.get_statistics()
        intersection_info = self.env.get_intersection_info()
        
        journey_info = f"🚗 TRAFFIC FLOW ANALYTICS\n"
        journey_info += f"Total Spawned: {stats['total_spawned']}\n"
        journey_info += f"Completed Journeys: {stats['total_completed']}\n"
        
        if stats['avg_journey_time'] > 0:
            journey_info += f"Avg Journey Time: {stats['avg_journey_time']:.1f}s\n"
            journey_info += f"Journey Range: {stats['min_journey_time']:.1f}-{stats['max_journey_time']:.1f}s\n"
            journey_info += f"System Efficiency: {stats['efficiency']:.1f}%"
        
        self.journey_text.set_text(journey_info)
        
        # Intersection information
        intersection_text = f"🏛️ INTERSECTION MANAGEMENT\n"
        intersection_text += f"Total Intersections: {intersection_info['count']}\n"
        intersection_text += f"Main Intersection: ✅\n"
        intersection_text += f"Traffic Light Ready: 🚦\n"
        intersection_text += f"(Future Integration Point)\n"
        intersection_text += f"6G-Managed Crossings: {reservations}"
        
        self.intersection_text.set_text(intersection_text)
    
    def run_smart_highway_episode(self, episode_num=1, max_steps=300, delay=0.12):
        """Run a smart highway episode with 6G communication."""
        print(f"\n🛣️ === SMART HIGHWAY 6G EPISODE {episode_num} ===")
        print("🎯 Key Features:")
        print("   • Vehicles move ONLY in X-axis OR Y-axis (no random directions)")
        print("   • Clear source → destination navigation")
        print("   • Dedicated lanes prevent same-axis collisions")
        print("   • 6G V2V/V2I communication prevents cross-axis collisions")
        print("   • Clear intersection definitions for traffic light integration")
        print("   • Actions: maintain speed, accelerate, brake (NO direction changes)")
        
        obs, info = self.env.reset()
        total_reward = 0
        step = 0
        total_6g_prevented = 0
        
        plt.ion()
        
        while step < max_steps:
            # Get action (random for demo, or from trained model)
            if self.model:
                action, _ = self.model.predict(obs, deterministic=True)
                if step == 1:  # Only print once per episode
                    print("   🧠 Using TRAINED model for agent decisions")
            else:
                action = np.random.randint(0, 3)  # 0=maintain, 1=accelerate, 2=brake
                if step == 1:  # Only print once per episode
                    print("   🎲 Using RANDOM actions for agent (training needed)")
            
            # Take step
            obs, reward, terminated, truncated, step_info = self.env.step(action)
            total_reward += reward
            step += 1
            
            # Track 6G collision prevention and actual collisions
            collisions_prevented = len(step_info.get('collisions_prevented', []))
            actual_collisions = len(step_info.get('actual_collisions', []))
            total_6g_prevented += collisions_prevented
            
            if actual_collisions > 0:
                print(f"   💥 Step {step}: {actual_collisions} ACTUAL collision(s) occurred!")
            elif collisions_prevented > 0:
                print(f"   🛡️  Step {step}: 6G prevented {collisions_prevented} potential collision(s)")
            
            # Track 6G communication activity
            if step_info.get('messages_sent', 0) > 0 and step % 25 == 0:
                delivery_rate = (step_info.get('messages_delivered', 0) / step_info['messages_sent']) * 100
                reservations = len(step_info.get('intersection_reservations', []))
                print(f"   📡 Step {step}: 6G activity - {step_info['messages_sent']} msgs, {reservations} reservations, {delivery_rate:.1f}% delivery")
            
            # Update visualization
            active_count, total_speed, direction_counts = self._draw_vehicles(self.env.vehicles)
            self._update_displays(step, total_reward, active_count, total_speed, direction_counts, step_info, episode_num)
            
            plt.draw()
            plt.pause(delay)
            
            # Progress updates
            if step % 50 == 0:
                print(f"   📊 Step {step}: {active_count} vehicles, avg speed {total_speed/max(active_count,1):.2f}")
                stats = self.env.get_statistics()
                print(f"      Journey stats: {stats['total_completed']}/{stats['total_spawned']} completed, {stats['efficiency']:.1f}% efficiency")
            
            if terminated or truncated:
                print(f"   ⚠️  Episode terminated at step {step}")
                break
        
        # Final statistics
        final_stats = self.env.get_statistics()
        
        print(f"\n📈 SMART HIGHWAY EPISODE {episode_num} SUMMARY:")
        print(f"   Duration: {step} steps")
        print(f"   Agent Total Reward: {total_reward:.2f}")
        print(f"   🛡️  Total Collisions Prevented by 6G: {total_6g_prevented}")
        print(f"   Final Active Vehicles: {active_count}")
        print(f"   📊 Traffic Flow Performance:")
        print(f"     Vehicles Spawned: {final_stats['total_spawned']}")
        print(f"     Journeys Completed: {final_stats['total_completed']}")
        print(f"     System Efficiency: {final_stats['efficiency']:.1f}%")
        
        if final_stats['avg_journey_time'] > 0:
            print(f"     Average Journey Time: {final_stats['avg_journey_time']:.1f} steps")
            print(f"     Journey Time Range: {final_stats['min_journey_time']:.1f}-{final_stats['max_journey_time']:.1f} steps")
        
        return total_reward, step, total_6g_prevented

def main():
    parser = argparse.ArgumentParser(description='Smart Highway with 6G communication')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of episodes')
    parser.add_argument('--max-steps', type=int, default=300,
                       help='Maximum steps per episode')
    parser.add_argument('--delay', type=float, default=0.12,
                       help='Animation delay (seconds)')
    parser.add_argument('--grid-size', type=int, default=10,
                       help='Grid size')
    parser.add_argument('--max-vehicles', type=int, default=16,
                       help='Maximum vehicles')
    parser.add_argument('--spawn-rate', type=float, default=0.4,
                       help='Vehicle spawn rate')
    parser.add_argument('--use-trained-model', action='store_true',
                       help='Use trained model instead of random actions')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    print("🛣️" * 25)
    print("🚗 SMART HIGHWAY with 6G V2V/V2I COMMUNICATION 🚗")
    print("🛣️" * 25)
    print(f"Grid: {args.grid_size}x{args.grid_size}")
    print(f"Max Vehicles: {args.max_vehicles}")
    print(f"Spawn Rate: {args.spawn_rate}")
    print(f"Episodes: {args.episodes}")
    print("\n🎯 This simulation demonstrates:")
    print("  🛣️  Proper directional lanes (X or Y axis movement only)")
    print("  📍 Clear source → destination navigation")
    print("  🚫 No random direction changes")
    print("  📡 6G V2V/V2I communication for intersection management")
    print("  🛡️  Collision prevention through 6G reservations")
    print("  🚦 Intersection definitions ready for traffic lights")
    print("🛣️" * 25)
    
    # Create smart highway environment
    env = SmartHighwayEnv(
        grid_size=(args.grid_size, args.grid_size),
        max_vehicles=args.max_vehicles,
        spawn_rate=args.spawn_rate,
        debug=args.debug
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
        print("🎲 Using random actions for all vehicles")
    
    # Create visualizer
    visualizer = SmartHighwayVisualizer(env, model)
    
    # Run episodes
    total_prevented = 0
    try:
        for episode in range(1, args.episodes + 1):
            reward, steps, prevented = visualizer.run_smart_highway_episode(
                episode_num=episode,
                max_steps=args.max_steps,
                delay=args.delay
            )
            total_prevented += prevented
            
            if episode < args.episodes:
                print(f"\n⏳ Waiting 3 seconds before next episode...")
                time.sleep(3)
    
    except KeyboardInterrupt:
        print("\n⚠️  Smart highway simulation interrupted by user")
    
    print(f"\n🏁 FINAL SMART HIGHWAY SUMMARY:")
    print(f"🛡️  Total collisions prevented by 6G: {total_prevented}")
    print(f"🎯 This demonstrates proper directional movement + 6G communication!")
    print(f"🚦 Intersections are ready for future traffic light integration!")
    
    input("Press Enter to close...")
    plt.close()

if __name__ == "__main__":
    main() 