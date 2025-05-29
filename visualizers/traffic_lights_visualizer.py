#!/usr/bin/env python3
"""
Traffic Lights Real-Time Visualizer
===================================

Real-time visualization of traffic lights system with vehicle queuing,
traffic light states, and journey time analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import sys
import os
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from environments.smart_highway_env import SmartHighwayEnv
from stable_baselines3 import PPO

class TrafficLightsVisualizer:
    """Real-time visualizer for traffic lights system."""
    
    def __init__(self, enable_6g=True, enable_traffic_lights=True):
        self.enable_6g = enable_6g
        self.enable_traffic_lights = enable_traffic_lights
        
        # Create environment with traffic lights
        self.env = SmartHighwayEnv(
            grid_size=(10, 10),
            max_vehicles=20,
            spawn_rate=0.5,
            enable_6g=enable_6g,
            enable_traffic_lights=enable_traffic_lights,
            debug=False
        )
        
        # Load trained model if available
        self.model = None
        try:
            self.model = PPO.load("../trained_models/ppo_smart_highway")
            print("✅ Using trained model for intelligent behavior")
        except:
            print("ℹ️ No trained model found - using random actions")
        
        # Setup visualization
        self.fig, ((self.ax_main, self.ax_lights), (self.ax_stats, self.ax_journey)) = plt.subplots(2, 2, figsize=(18, 14))
        self.fig.suptitle(f'🚦 TRAFFIC LIGHTS SIMULATION (6G: {"ON" if enable_6g else "OFF"}, Lights: {"ON" if enable_traffic_lights else "OFF"})', 
                         fontsize=16, fontweight='bold')
        
        # Journey time tracking
        self.journey_times = []
        self.wait_times = []
        self.simulation_data = {
            'steps': [],
            'queued_vehicles': [],
            'traffic_efficiency': [],
            'total_spawned': [],
            'total_completed': []
        }
        
    def run_traffic_lights_demo(self, max_steps=400, delay=0.1):
        """Run traffic lights demonstration with real-time visualization."""
        print("🚦" * 25)
        print("🚗 TRAFFIC LIGHTS SYSTEM DEMONSTRATION 🚗")
        print("🚦" * 25)
        print("🎯 Features:")
        print("  🚦 Active traffic light control with timing cycles")
        print("  🚗 Vehicle queuing at red lights")
        print("  ⏱️ Real-time journey time tracking")
        print("  📊 Traffic efficiency monitoring")
        if self.enable_6g:
            print("  📡 6G communication for coordination")
        print("🚦" * 25)
        
        obs, info = self.env.reset()
        plt.ion()
        
        for step in range(max_steps):
            # Get action
            if self.model:
                action, _ = self.model.predict(obs, deterministic=True)
            else:
                action = self.env.action_space.sample()
            
            # Take step
            obs, reward, terminated, truncated, step_info = self.env.step(action)
            
            # Update simulation data
            stats = self.env.get_statistics()
            self.simulation_data['steps'].append(step)
            self.simulation_data['queued_vehicles'].append(stats.get('queued_vehicles', 0))
            self.simulation_data['traffic_efficiency'].append(stats.get('traffic_efficiency', 100))
            self.simulation_data['total_spawned'].append(stats['total_spawned'])
            self.simulation_data['total_completed'].append(stats['total_completed'])
            
            # Update visualization
            self.update_visualization(step, stats, step_info)
            
            # Progress updates
            if step % 50 == 0:
                completion_rate = (stats['total_completed'] / max(stats['total_spawned'], 1)) * 100
                print(f"Step {step}: Spawned={stats['total_spawned']}, Completed={stats['total_completed']}, "
                      f"Rate={completion_rate:.1f}%, Queued={stats.get('queued_vehicles', 0)}")
            
            # Control animation speed
            plt.pause(delay)
            
            if terminated or truncated:
                break
        
        # Final statistics
        final_stats = self.env.get_statistics()
        print(f"\n🏁 SIMULATION COMPLETED!")
        print(f"📊 Final Statistics:")
        print(f"  Total Vehicles: {final_stats['total_spawned']} spawned, {final_stats['total_completed']} completed")
        print(f"  Journey Time: {final_stats['avg_journey_time']:.2f} steps average")
        print(f"  Collisions: {final_stats['collision_count']}")
        if self.enable_traffic_lights:
            print(f"  Wait Time: {final_stats['avg_wait_time_per_vehicle']:.2f} steps per vehicle")
            print(f"  Traffic Efficiency: {final_stats['traffic_efficiency']:.1f}%")
        
        plt.ioff()
        plt.show()
        
        return final_stats
    
    def update_visualization(self, step, stats, step_info):
        """Update all visualization components."""
        # Clear all axes
        self.ax_main.clear()
        self.ax_lights.clear()
        self.ax_stats.clear()
        self.ax_journey.clear()
        
        # Main simulation view
        self.draw_main_simulation()
        
        # Traffic lights status
        self.draw_traffic_lights_panel(step_info)
        
        # Statistics panel
        self.draw_statistics_panel(step, stats)
        
        # Journey time analysis
        self.draw_journey_analysis(stats)
        
        plt.tight_layout()
    
    def draw_main_simulation(self):
        """Draw the main simulation view with traffic lights and vehicles."""
        self.ax_main.set_xlim(-1, self.env.grid_size[1] + 1)
        self.ax_main.set_ylim(-1, self.env.grid_size[0] + 1)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title('🚦 Smart Highway with Traffic Lights', fontweight='bold')
        
        # Draw grid
        for i in range(self.env.grid_size[0] + 1):
            self.ax_main.axhline(y=i, color='lightgray', linewidth=0.5, alpha=0.5)
        for j in range(self.env.grid_size[1] + 1):
            self.ax_main.axvline(x=j, color='lightgray', linewidth=0.5, alpha=0.5)
        
        # Draw intersections and traffic lights
        for intersection in self.env.intersections:
            int_x, int_y = intersection['position']
            
            # Draw intersection area
            intersection_rect = patches.Rectangle(
                (int_x - 0.5, int_y - 0.5), 1, 1,
                linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.3
            )
            self.ax_main.add_patch(intersection_rect)
            
            # Draw traffic light
            if self.enable_traffic_lights:
                light_state = intersection['traffic_light']['current_state']
                self.draw_traffic_light(int_x, int_y, light_state)
            
            # Label intersection
            self.ax_main.text(int_x, int_y - 0.7, intersection['id'].replace('intersection_', ''),
                            ha='center', va='center', fontsize=8, fontweight='bold')
        
        # Draw vehicles
        active_vehicles, queued_vehicles = self.draw_vehicles()
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Moving Vehicle'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Queued Vehicle'),
            plt.Line2D([0], [0], marker='o', color='green', markersize=10, label='Green Light'),
            plt.Line2D([0], [0], marker='o', color='red', markersize=10, label='Red Light')
        ]
        self.ax_main.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        return active_vehicles, queued_vehicles
    
    def draw_traffic_light(self, x, y, state):
        """Draw a traffic light at intersection."""
        light_size = 0.15
        
        # Traffic light pole
        self.ax_main.plot([x, x], [y + 0.6, y + 0.9], 'k-', linewidth=3)
        
        # Light housing
        light_bg = patches.Rectangle(
            (x - light_size, y + 0.6), 2 * light_size, 0.3,
            linewidth=1, edgecolor='black', facecolor='darkgray'
        )
        self.ax_main.add_patch(light_bg)
        
        # Determine light colors based on state
        horizontal_color = 'gray'
        vertical_color = 'gray'
        
        if state == 'green_horizontal':
            horizontal_color = 'green'
        elif state == 'yellow_horizontal':
            horizontal_color = 'yellow'
        elif state == 'green_vertical':
            vertical_color = 'green'
        elif state == 'yellow_vertical':
            vertical_color = 'yellow'
        
        # Draw horizontal direction light (left side)
        h_light = patches.Circle((x - 0.1, y + 0.75), 0.05, facecolor=horizontal_color, edgecolor='black')
        self.ax_main.add_patch(h_light)
        
        # Draw vertical direction light (right side)
        v_light = patches.Circle((x + 0.1, y + 0.75), 0.05, facecolor=vertical_color, edgecolor='black')
        self.ax_main.add_patch(v_light)
        
        # Add directional arrows
        if horizontal_color == 'green':
            self.ax_main.annotate('→', xy=(x - 0.1, y + 0.75), ha='center', va='center', 
                                fontsize=8, fontweight='bold', color='white')
        if vertical_color == 'green':
            self.ax_main.annotate('↓', xy=(x + 0.1, y + 0.75), ha='center', va='center', 
                                fontsize=8, fontweight='bold', color='white')
    
    def draw_vehicles(self):
        """Draw all vehicles with queue status."""
        active_count = 0
        queued_count = 0
        
        for i in range(self.env.max_vehicles):
            if self.env.vehicles[i, 6] == 0:  # Not active
                continue
            
            x, y = self.env.vehicles[i, 0], self.env.vehicles[i, 1]
            direction = int(self.env.vehicles[i, 4])
            lane_offset = self.env.vehicles[i, 5]
            
            # Adjust position for lane visualization
            if direction == 0:  # L2R
                y_draw = y + lane_offset
                x_draw = x
            else:  # T2B
                x_draw = x + lane_offset
                y_draw = y
            
            # Check if vehicle is queued
            is_queued = i in self.env.queued_vehicles if self.enable_traffic_lights else False
            
            # Vehicle appearance
            if is_queued:
                # Queued vehicle - red square
                vehicle_marker = patches.Rectangle(
                    (x_draw - 0.1, y_draw - 0.1), 0.2, 0.2,
                    facecolor='red', edgecolor='darkred', linewidth=2
                )
                self.ax_main.add_patch(vehicle_marker)
                queued_count += 1
            else:
                # Moving vehicle - blue circle
                vehicle_marker = patches.Circle(
                    (x_draw, y_draw), 0.1,
                    facecolor='blue', edgecolor='darkblue', linewidth=1
                )
                self.ax_main.add_patch(vehicle_marker)
            
            active_count += 1
            
            # Vehicle ID
            self.ax_main.text(x_draw, y_draw, str(i), ha='center', va='center', 
                            fontsize=6, fontweight='bold', color='white')
            
            # Direction arrow
            if direction == 0:  # L2R
                self.ax_main.annotate('', xy=(x_draw + 0.15, y_draw), xytext=(x_draw - 0.15, y_draw),
                                    arrowprops=dict(arrowstyle='->', color='black', lw=1))
            else:  # T2B
                self.ax_main.annotate('', xy=(x_draw, y_draw + 0.15), xytext=(x_draw, y_draw - 0.15),
                                    arrowprops=dict(arrowstyle='->', color='black', lw=1))
            
            # Destination
            dest_x, dest_y = self.env.vehicles[i, 8], self.env.vehicles[i, 9]
            self.ax_main.plot(dest_x, dest_y, 'X', color='green', markersize=8, alpha=0.7)
        
        return active_count, queued_count
    
    def draw_traffic_lights_panel(self, step_info):
        """Draw traffic lights status panel."""
        self.ax_lights.set_title('🚦 Traffic Lights Status', fontweight='bold')
        self.ax_lights.axis('off')
        
        if not self.enable_traffic_lights:
            self.ax_lights.text(0.5, 0.5, 'TRAFFIC LIGHTS DISABLED', 
                              ha='center', va='center', fontsize=16, fontweight='bold', color='red')
            return
        
        # Get current traffic light states
        light_states = step_info.get('traffic_lights', {})
        
        y_pos = 0.9
        for intersection_id, light_info in light_states.items():
            state = light_info['state']
            position = light_info['position']
            cycle_time = light_info['cycle_time']
            
            # Format intersection name
            display_name = intersection_id.replace('intersection_', 'Int. ')
            
            # State color
            if 'green_horizontal' in state:
                state_color = 'green'
                state_text = "🟢 GREEN ↔"
            elif 'yellow_horizontal' in state:
                state_color = 'orange'
                state_text = "🟡 YELLOW ↔"
            elif 'green_vertical' in state:
                state_color = 'green'
                state_text = "🟢 GREEN ↕"
            elif 'yellow_vertical' in state:
                state_color = 'orange'
                state_text = "🟡 YELLOW ↕"
            else:
                state_color = 'red'
                state_text = "🔴 RED"
            
            # Display information
            self.ax_lights.text(0.1, y_pos, f"{display_name}:", fontweight='bold', fontsize=10)
            self.ax_lights.text(0.4, y_pos, state_text, color=state_color, fontweight='bold', fontsize=10)
            self.ax_lights.text(0.7, y_pos, f"T:{cycle_time:.1f}s", fontsize=9)
            
            y_pos -= 0.15
            
            if y_pos < 0.1:
                break
    
    def draw_statistics_panel(self, step, stats):
        """Draw real-time statistics panel."""
        self.ax_stats.set_title('📊 Real-Time Statistics', fontweight='bold')
        self.ax_stats.axis('off')
        
        # Current statistics
        active_vehicles = stats['active_vehicles']
        total_spawned = stats['total_spawned']
        total_completed = stats['total_completed']
        completion_rate = stats['completion_rate']
        collision_count = stats['collision_count']
        
        stats_text = f"""
🕐 Simulation Time: {step} steps
🚗 Active Vehicles: {active_vehicles}
📈 Total Spawned: {total_spawned}
✅ Total Completed: {total_completed}
📊 Completion Rate: {completion_rate:.1f}%
💥 Collisions: {collision_count}
"""
        
        if self.enable_traffic_lights:
            queued_vehicles = stats.get('queued_vehicles', 0)
            total_wait_time = stats.get('total_wait_time', 0)
            avg_wait_time = stats.get('avg_wait_time_per_vehicle', 0)
            traffic_efficiency = stats.get('traffic_efficiency', 100)
            
            stats_text += f"""
🚦 Queued Vehicles: {queued_vehicles}
⏱️ Total Wait Time: {total_wait_time} steps
⏰ Avg Wait/Vehicle: {avg_wait_time:.2f} steps
🎯 Traffic Efficiency: {traffic_efficiency:.1f}%
"""
        
        self.ax_stats.text(0.1, 0.9, stats_text, fontsize=11, verticalalignment='top')
    
    def draw_journey_analysis(self, stats):
        """Draw journey time analysis charts."""
        if len(self.simulation_data['steps']) < 10:
            self.ax_journey.text(0.5, 0.5, 'Collecting Data...', ha='center', va='center', fontsize=14)
            self.ax_journey.set_title('⏱️ Journey Time Analysis', fontweight='bold')
            return
        
        # Plot queued vehicles over time
        steps = self.simulation_data['steps'][-100:]  # Last 100 steps
        queued = self.simulation_data['queued_vehicles'][-100:]
        efficiency = self.simulation_data['traffic_efficiency'][-100:]
        
        # Create twin axis for efficiency
        ax_twin = self.ax_journey.twinx()
        
        # Plot queued vehicles
        line1 = self.ax_journey.plot(steps, queued, 'r-', linewidth=2, label='Queued Vehicles')
        self.ax_journey.set_ylabel('Queued Vehicles', color='r')
        self.ax_journey.tick_params(axis='y', labelcolor='r')
        
        # Plot traffic efficiency
        line2 = ax_twin.plot(steps, efficiency, 'b-', linewidth=2, label='Traffic Efficiency (%)')
        ax_twin.set_ylabel('Traffic Efficiency (%)', color='b')
        ax_twin.tick_params(axis='y', labelcolor='b')
        ax_twin.set_ylim(0, 100)
        
        self.ax_journey.set_xlabel('Simulation Step')
        self.ax_journey.set_title('⏱️ Traffic Flow Analysis', fontweight='bold')
        self.ax_journey.grid(True, alpha=0.3)
        
        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        self.ax_journey.legend(lines, labels, loc='upper left')

def main():
    parser = argparse.ArgumentParser(description='Traffic Lights Real-Time Visualizer')
    parser.add_argument('--steps', type=int, default=400,
                       help='Number of simulation steps')
    parser.add_argument('--delay', type=float, default=0.1,
                       help='Animation delay (seconds)')
    parser.add_argument('--no-6g', action='store_true',
                       help='Disable 6G communication')
    parser.add_argument('--no-lights', action='store_true',
                       help='Disable traffic lights')
    
    args = parser.parse_args()
    
    visualizer = TrafficLightsVisualizer(
        enable_6g=not args.no_6g,
        enable_traffic_lights=not args.no_lights
    )
    
    # Run the demonstration
    visualizer.run_traffic_lights_demo(max_steps=args.steps, delay=args.delay)

if __name__ == "__main__":
    main() 