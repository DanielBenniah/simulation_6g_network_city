#!/usr/bin/env python3
"""
Enhanced Multi-Agent Smart Highway Visualizer
===========================================

Smooth, high-FPS visualization of multiple learning agents with:
- Interpolated vehicle movements (60+ FPS)
- Visual speed indicators and acceleration feedback
- Real-time 6G communication visualization
- Detailed collision detection and prevention metrics
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import sys
import os
import argparse
import time
from collections import deque
import threading
import queue

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import PPO
from train_multi_agent_highway import create_multi_agent_environment

class SmoothMultiAgentVisualizer:
    """Enhanced visualizer with smooth animations and detailed metrics."""
    
    def __init__(self, env, model=None, window_size=(15, 10), target_fps=60):
        self.env = env
        self.model = model
        self.window_size = window_size
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        
        # Animation state
        self.current_step = 0
        self.simulation_running = False
        self.last_vehicle_states = None
        self.interpolation_progress = 0.0
        
        # Enhanced vehicle tracking
        self.vehicle_trails = {}  # Store recent positions for trails
        self.vehicle_speeds = {}  # Track speed history
        self.communication_history = deque(maxlen=30)  # Recent communications
        
        # Colors for different agents
        self.agent_colors = [
            '#FF6B6B',  # Red
            '#4ECDC4',  # Teal  
            '#45B7D1',  # Blue
            '#96CEB4',  # Green
            '#FFEAA7',  # Yellow
            '#DDA0DD',  # Plum
            '#FFB347',  # Orange
            '#98D8C8'   # Mint
        ]
        
        # Performance tracking with more detail
        self.performance_history = {
            'rewards': deque(maxlen=200),
            'collisions_prevented': deque(maxlen=200),
            'actual_collisions': deque(maxlen=200),
            'coordination_score': deque(maxlen=200),
            'avg_speed': deque(maxlen=200),
            'communication_efficiency': deque(maxlen=200)
        }
        
        # Setup matplotlib for smooth animation
        plt.style.use('dark_background')
        self.fig, ((self.ax_main, self.ax_metrics), (self.ax_comm, self.ax_speed)) = plt.subplots(
            2, 2, figsize=window_size, gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [3, 1]}
        )
        
        self.setup_visualization()
        
    def setup_visualization(self):
        """Setup the enhanced visualization components."""
        # Main highway view
        self.ax_main.set_xlim(-1, self.env.base_env.grid_size[1] + 1)
        self.ax_main.set_ylim(-1, self.env.base_env.grid_size[0] + 1)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title('🤖 Enhanced Multi-Agent Smart Highway 🤖', fontsize=16, color='white')
        self.ax_main.set_facecolor('#0a0a0a')
        
        # Draw enhanced highway infrastructure
        self.draw_enhanced_infrastructure()
        
        # Metrics panel
        self.ax_metrics.set_title('📊 Real-Time Metrics', fontsize=12, color='white')
        self.ax_metrics.set_facecolor('#0a0a0a')
        
        # Communication panel
        self.ax_comm.set_title('📡 6G Communication', fontsize=12, color='white')
        self.ax_comm.set_facecolor('#0a0a0a')
        
        # Speed analysis panel
        self.ax_speed.set_title('🚗 Speed Analysis', fontsize=12, color='white')
        self.ax_speed.set_facecolor('#0a0a0a')
        
        # Initialize visualization elements
        self.vehicle_patches = {}
        self.agent_labels = {}
        self.communication_lines = []
        self.speed_indicators = {}
        self.trail_lines = {}
        
    def draw_enhanced_infrastructure(self):
        """Draw enhanced highway infrastructure with better visual feedback."""
        # Enhanced lane markings
        lane_y_positions = [2, 4, 6, 8]
        for y in lane_y_positions:
            # Main lane lines
            self.ax_main.axhline(y, color='white', linewidth=2, alpha=0.8, linestyle='-')
            
            # Sub-lane markings with better visibility
            for offset in [-0.4, -0.2, 0.2, 0.4]:
                self.ax_main.axhline(y + offset, color='#666666', linewidth=1, alpha=0.6, linestyle=':')
        
        # Enhanced vertical lanes
        lane_x_positions = [2, 4, 6, 8]
        for x in lane_x_positions:
            self.ax_main.axvline(x, color='white', linewidth=2, alpha=0.8, linestyle='-')
            for offset in [-0.4, -0.2, 0.2, 0.4]:
                self.ax_main.axvline(x + offset, color='#666666', linewidth=1, alpha=0.6, linestyle=':')
        
        # Enhanced intersections with 6G infrastructure
        intersections = self.env.base_env.intersections
        for intersection in intersections:
            x, y = intersection['position']
            
            # Intersection area with gradient effect
            for radius in [intersection['size'], intersection['size']*0.7, intersection['size']*0.4]:
                alpha = 0.3 * (1 - radius/intersection['size'])
                intersection_patch = patches.Circle(
                    (x, y), radius, 
                    facecolor='#2C3E50', edgecolor='#3498DB', linewidth=1, alpha=alpha
                )
                self.ax_main.add_patch(intersection_patch)
            
            # Enhanced 6G tower with communication range
            comm_range = patches.Circle((x, y), 2.5, fill=False, edgecolor='#E74C3C', 
                                      linewidth=1, alpha=0.3, linestyle='--')
            self.ax_main.add_patch(comm_range)
            
            # 6G tower symbol
            self.ax_main.plot(x, y, marker='^', color='#E74C3C', markersize=10, alpha=0.9)
            self.ax_main.plot(x, y, marker='^', color='#FF6B35', markersize=6, alpha=0.7)
            
            # Enhanced intersection label
            self.ax_main.text(x, y-1.0, f"6G-{intersection['id'].split('_')[-2:]}", 
                            ha='center', va='center', fontsize=9, color='#3498DB', fontweight='bold')
    
    def interpolate_vehicle_position(self, old_state, new_state, progress):
        """Smoothly interpolate between vehicle positions."""
        if old_state is None:
            return new_state
        
        # Linear interpolation for position
        x = old_state[0] + (new_state[0] - old_state[0]) * progress
        y = old_state[1] + (new_state[1] - old_state[1]) * progress
        
        # Smooth velocity interpolation
        vx = old_state[2] + (new_state[2] - old_state[2]) * progress
        vy = old_state[3] + (new_state[3] - old_state[3]) * progress
        
        interpolated = new_state.copy()
        interpolated[0] = x
        interpolated[1] = y
        interpolated[2] = vx
        interpolated[3] = vy
        
        return interpolated
    
    def update_vehicle_trails(self, vehicle_id, x, y):
        """Update vehicle movement trails."""
        if vehicle_id not in self.vehicle_trails:
            self.vehicle_trails[vehicle_id] = deque(maxlen=15)  # Trail length
        
        self.vehicle_trails[vehicle_id].append((x, y))
    
    def update_speed_tracking(self, vehicle_id, vx, vy):
        """Track vehicle speed changes."""
        speed = np.sqrt(vx**2 + vy**2)
        if vehicle_id not in self.vehicle_speeds:
            self.vehicle_speeds[vehicle_id] = deque(maxlen=10)
        
        self.vehicle_speeds[vehicle_id].append(speed)
    
    def update_smooth_visualization(self, simulation_data, interpolation_progress):
        """Update visualization with smooth interpolation."""
        try:
            # Clear previous dynamic elements
            for patch in list(self.vehicle_patches.values()):
                patch.remove()
            for label in list(self.agent_labels.values()):
                label.remove()
            for line in self.communication_lines:
                line.remove()
            for indicator in list(self.speed_indicators.values()):
                if hasattr(indicator, 'remove'):
                    indicator.remove()
            for trail in list(self.trail_lines.values()):
                for line in trail:
                    line.remove()
                
            self.vehicle_patches.clear()
            self.agent_labels.clear()
            self.communication_lines.clear()
            self.speed_indicators.clear()
            self.trail_lines.clear()
            
            # Get current and interpolated vehicle states
            vehicles = simulation_data['vehicles']
            info = simulation_data.get('info', {})
            active_agents = []
            
            # Draw vehicles with smooth interpolation
            for i in range(vehicles.shape[0]):
                if vehicles[i, 6] == 1:  # Active vehicle
                    active_agents.append(i)
                    
                    # Get interpolated position
                    current_state = vehicles[i]
                    if self.last_vehicle_states is not None and i < len(self.last_vehicle_states):
                        interpolated_state = self.interpolate_vehicle_position(
                            self.last_vehicle_states[i], current_state, interpolation_progress
                        )
                    else:
                        interpolated_state = current_state
                    
                    x, y = interpolated_state[0], interpolated_state[1]
                    vx, vy = interpolated_state[2], interpolated_state[3]
                    direction = int(current_state[4])
                    lane_offset = current_state[5]
                    
                    # Update tracking
                    self.update_vehicle_trails(i, x, y)
                    self.update_speed_tracking(i, vx, vy)
                    
                    # Vehicle color
                    color = self.agent_colors[i % len(self.agent_colors)]
                    
                    # Draw vehicle trail
                    if i in self.vehicle_trails and len(self.vehicle_trails[i]) > 1:
                        trail_positions = list(self.vehicle_trails[i])
                        trail_x = [pos[0] for pos in trail_positions]
                        trail_y = [pos[1] for pos in trail_positions]
                        
                        # Create fading trail effect
                        for j in range(len(trail_x)-1):
                            alpha = (j+1) / len(trail_x) * 0.5
                            trail_line = self.ax_main.plot(
                                trail_x[j:j+2], trail_y[j:j+2], 
                                color=color, linewidth=2, alpha=alpha
                            )[0]
                            if i not in self.trail_lines:
                                self.trail_lines[i] = []
                            self.trail_lines[i].append(trail_line)
                    
                    # Vehicle position with lane offset
                    if direction == 0:  # L2R (horizontal)
                        width, height = 0.7, 0.35
                        angle = 0
                        x_draw, y_draw = x, y + lane_offset
                    else:  # T2B (vertical)
                        width, height = 0.35, 0.7
                        angle = 90
                        x_draw, y_draw = x + lane_offset, y
                    
                    # Vehicle body with speed-based effects
                    speed = np.sqrt(vx**2 + vy**2)
                    # Slightly larger vehicle when moving fast
                    speed_factor = 1 + min(speed * 0.1, 0.3)
                    
                    vehicle_patch = patches.Rectangle(
                        (x_draw - width*speed_factor/2, y_draw - height*speed_factor/2), 
                        width*speed_factor, height*speed_factor,
                        angle=angle, facecolor=color, edgecolor='white', 
                        linewidth=2, alpha=0.9
                    )
                    self.vehicle_patches[i] = vehicle_patch
                    self.ax_main.add_patch(vehicle_patch)
                    
                    # Agent ID label
                    label = self.ax_main.text(
                        x_draw, y_draw, f'A{i}', ha='center', va='center', 
                        fontsize=9, fontweight='bold', color='black'
                    )
                    self.agent_labels[i] = label
                    
                    # Enhanced speed indicator
                    if speed > 0.1:
                        # Speed arrow
                        arrow_length = min(speed * 0.4, 1.2)
                        if direction == 0:  # Horizontal
                            dx, dy = arrow_length * np.sign(vx), 0
                        else:  # Vertical
                            dx, dy = 0, arrow_length * np.sign(vy)
                        
                        arrow = self.ax_main.annotate('', 
                            xy=(x_draw + dx*0.7, y_draw + dy*0.7),
                            xytext=(x_draw - dx*0.3, y_draw - dy*0.3),
                            arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.8)
                        )
                        self.speed_indicators[i] = arrow
                        
                        # Speed text
                        speed_text = self.ax_main.text(
                            x_draw + 0.8, y_draw + 0.8, f'{speed:.1f}', 
                            fontsize=7, color=color, fontweight='bold', alpha=0.8
                        )
                        self.communication_lines.append(speed_text)
                    
                    # Destination marker with progress indicator
                    dest_x, dest_y = current_state[8], current_state[9]
                    distance_to_dest = np.sqrt((x - dest_x)**2 + (y - dest_y)**2)
                    progress_marker_size = max(8, 15 - distance_to_dest)
                    
                    self.ax_main.plot(dest_x, dest_y, marker='X', color=color, 
                                    markersize=progress_marker_size, alpha=0.8, 
                                    markeredgecolor='white', markeredgewidth=1)
                    
                    # Progress line to destination
                    progress_line = self.ax_main.plot([x, dest_x], [y, dest_y], 
                                                    color=color, linewidth=1, alpha=0.3, linestyle='--')[0]
                    self.communication_lines.append(progress_line)
            
            # Enhanced 6G communication visualization
            self.draw_enhanced_communication(active_agents, info)
            
            # Update all metrics panels
            self.update_enhanced_metrics(active_agents, info, interpolation_progress)
            
            # Update title with real-time stats
            avg_speed = np.mean([np.sqrt(vehicles[i, 2]**2 + vehicles[i, 3]**2) 
                               for i in active_agents]) if active_agents else 0
            
            self.ax_main.set_title(
                f'🤖 Enhanced Multi-Agent Highway | Agents: {len(active_agents)} | '
                f'Avg Speed: {avg_speed:.1f} | Time: {self.env.base_env.sim_time}s 🤖', 
                fontsize=12, color='white'
            )
            
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def draw_enhanced_communication(self, active_agents, info):
        """Draw enhanced 6G communication visualization."""
        # V2V communication with signal strength
        for i, agent1 in enumerate(active_agents):
            for agent2 in active_agents[i+1:]:
                x1, y1 = self.env.base_env.vehicles[agent1, 0], self.env.base_env.vehicles[agent1, 1]
                x2, y2 = self.env.base_env.vehicles[agent2, 0], self.env.base_env.vehicles[agent2, 1]
                
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if distance < 3.5:  # Extended communication range
                    # Signal strength based on distance
                    signal_strength = max(0.2, 1 - distance/3.5)
                    line_width = 1 + signal_strength * 2
                    
                    # Animated communication effect
                    animation_phase = (time.time() * 3) % (2 * np.pi)
                    alpha = 0.3 + 0.3 * np.sin(animation_phase)
                    
                    comm_line = self.ax_main.plot([x1, x2], [y1, y2], 
                                                color='#00FF41', linewidth=line_width, 
                                                alpha=alpha * signal_strength, linestyle=':')[0]
                    self.communication_lines.append(comm_line)
                    
                    # Communication data packets (animated dots)
                    packet_progress = (animation_phase / (2 * np.pi)) % 1.0
                    packet_x = x1 + (x2 - x1) * packet_progress
                    packet_y = y1 + (y2 - y1) * packet_progress
                    
                    packet_dot = self.ax_main.plot(packet_x, packet_y, 'o', 
                                                 color='#00FF41', markersize=4, alpha=0.8)[0]
                    self.communication_lines.append(packet_dot)
        
        # V2I communication with intersection management
        intersections = self.env.base_env.intersections
        for agent_id in active_agents:
            x, y = self.env.base_env.vehicles[agent_id, 0], self.env.base_env.vehicles[agent_id, 1]
            
            for intersection in intersections:
                int_x, int_y = intersection['position']
                distance = np.sqrt((x - int_x)**2 + (y - int_y)**2)
                
                if distance < 3.0:  # V2I communication range
                    # Signal strength visualization
                    signal_strength = max(0.3, 1 - distance/3.0)
                    
                    # Pulsing communication link
                    pulse_phase = (time.time() * 2) % (2 * np.pi)
                    pulse_alpha = 0.4 + 0.3 * np.sin(pulse_phase)
                    
                    comm_line = self.ax_main.plot([x, int_x], [y, int_y], 
                                                color='#FF6B35', linewidth=2, 
                                                alpha=pulse_alpha * signal_strength, linestyle='-.')[0]
                    self.communication_lines.append(comm_line)
                    
                    # Reservation status indicator
                    status_color = '#00FF00' if signal_strength > 0.7 else '#FFD700'
                    status_dot = self.ax_main.plot(int_x, int_y, 'o', 
                                                 color=status_color, markersize=6, alpha=0.7)[0]
                    self.communication_lines.append(status_dot)
    
    def update_enhanced_metrics(self, active_agents, info, interpolation_progress):
        """Update all metrics panels with detailed information."""
        # Clear all metrics panels
        self.ax_metrics.clear()
        self.ax_comm.clear()
        self.ax_speed.clear()
        
        # Set backgrounds and titles
        for ax, title in [(self.ax_metrics, '📊 Real-Time Metrics'), 
                         (self.ax_comm, '📡 6G Communication'), 
                         (self.ax_speed, '🚗 Speed Analysis')]:
            ax.set_facecolor('#0a0a0a')
            ax.set_title(title, fontsize=10, color='white')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        
        # Main metrics panel
        collision_count = len(info.get('actual_collisions', []))
        prevented_count = len(info.get('collisions_prevented', []))
        messages_sent = info.get('messages_sent', 0)
        messages_delivered = info.get('messages_delivered', 0)
        
        # Calculate average speed
        avg_speed = np.mean([np.sqrt(self.env.base_env.vehicles[i, 2]**2 + 
                                   self.env.base_env.vehicles[i, 3]**2) 
                           for i in active_agents]) if active_agents else 0
        
        metrics_text = [
            f"🤖 Active Agents: {len(active_agents)}",
            f"⚡ Avg Speed: {avg_speed:.2f} m/s",
            f"🛡️ Prevented: {prevented_count}",
            f"💥 Collisions: {collision_count}",
            f"📡 Messages: {messages_sent}",
            f"📨 Delivered: {messages_delivered}",
            f"⏱️ Sim Time: {self.env.base_env.sim_time}s",
            f"🎬 FPS: {1/self.frame_time:.0f}"
        ]
        
        for i, text in enumerate(metrics_text):
            color = 'white'
            if 'Prevented' in text and prevented_count > 0:
                color = '#00FF00'
            elif 'Collisions' in text and collision_count > 0:
                color = '#FF6B6B'
            elif 'Speed' in text:
                color = '#4ECDC4'
            
            self.ax_metrics.text(0.05, 0.9 - i*0.1, text, 
                               transform=self.ax_metrics.transAxes,
                               fontsize=9, color=color, fontweight='bold')
        
        # Communication panel with real-time data
        if messages_sent > 0:
            delivery_rate = (messages_delivered / messages_sent) * 100
        else:
            delivery_rate = 0
            
        comm_efficiency = info.get('collision_prevention_rate', 0)
        
        comm_text = [
            f"📊 Delivery Rate: {delivery_rate:.1f}%",
            f"🎯 Prevention Rate: {comm_efficiency:.1f}%",
            f"📡 V2V Links: {len(active_agents) * (len(active_agents)-1) // 2}",
            f"🏗️ V2I Towers: {len(self.env.base_env.intersections)}",
        ]
        
        for i, text in enumerate(comm_text):
            color = '#00FF41' if 'Rate:' in text and float(text.split(':')[1].strip('%').strip().split('%')[0]) > 50 else '#FFD93D'
            self.ax_comm.text(0.05, 0.8 - i*0.15, text,
                            transform=self.ax_comm.transAxes,
                            fontsize=9, color=color, fontweight='bold')
        
        # Speed analysis panel
        if active_agents:
            speeds = [np.sqrt(self.env.base_env.vehicles[i, 2]**2 + self.env.base_env.vehicles[i, 3]**2) 
                     for i in active_agents]
            max_speed = max(speeds)
            min_speed = min(speeds)
            
            speed_text = [
                f"🚀 Max: {max_speed:.2f} m/s",
                f"🐌 Min: {min_speed:.2f} m/s",
                f"📈 Range: {max_speed - min_speed:.2f}",
                f"📊 Std Dev: {np.std(speeds):.2f}"
            ]
            
            for i, text in enumerate(speed_text):
                self.ax_speed.text(0.05, 0.8 - i*0.15, text,
                                 transform=self.ax_speed.transAxes,
                                 fontsize=9, color='#4ECDC4', fontweight='bold')
            
            # Speed distribution mini-chart
            if len(speeds) > 1:
                speed_bins = np.linspace(0, max(2.0, max_speed), 5)
                hist, _ = np.histogram(speeds, bins=speed_bins)
                bar_width = 0.15
                bar_x = np.linspace(0.1, 0.9, len(hist))
                max_hist = max(hist) if max(hist) > 0 else 1
                
                for j, (x, h) in enumerate(zip(bar_x, hist)):
                    bar_height = 0.2 * (h / max_hist)
                    bar = patches.Rectangle((x - bar_width/2, 0.1), bar_width, bar_height,
                                          facecolor='#4ECDC4', alpha=0.7)
                    self.ax_speed.add_patch(bar)
    
    def run_smooth_episode(self, max_steps=500, simulation_delay=0.1):
        """Run episode with smooth high-FPS visualization."""
        obs, info = self.env.reset()
        total_reward = 0
        step = 0
        
        print(f"🎬 Starting smooth multi-agent visualization...")
        print(f"   Target FPS: {self.target_fps}")
        print(f"   Active agents: {len(obs) if isinstance(obs, dict) else 'N/A'}")
        
        self.last_vehicle_states = None
        
        while step < max_steps:
            # Get simulation step
            step_start_time = time.time()
            
            # Get actions from model or random
            if self.model:
                if isinstance(obs, dict):
                    actions = []
                    for i in range(self.env.num_agents):
                        agent_obs = obs.get(f"agent_{i}", np.zeros(self.env.base_env.single_obs_space.shape))
                        action, _ = self.model.predict(agent_obs, deterministic=True)
                        actions.append(action)
                else:
                    actions, _ = self.model.predict(obs, deterministic=True)
            else:
                if isinstance(obs, dict):
                    actions = [np.random.randint(0, 3) for _ in range(len(obs))]
                else:
                    actions = [np.random.randint(0, 3) for _ in range(self.env.num_agents)]
            
            # Store previous vehicle states for interpolation
            self.last_vehicle_states = self.env.base_env.vehicles.copy()
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(actions)
            
            if isinstance(reward, dict):
                total_reward += sum(reward.values())
            else:
                total_reward += reward
            
            # Smooth animation between simulation steps
            simulation_data = {
                'vehicles': self.env.base_env.vehicles,
                'info': info
            }
            
            # Animate interpolation between previous and current state
            interpolation_steps = max(1, int(simulation_delay / self.frame_time))
            for interp_step in range(interpolation_steps):
                frame_start_time = time.time()
                
                interpolation_progress = (interp_step + 1) / interpolation_steps
                self.update_smooth_visualization(simulation_data, interpolation_progress)
                
                # Maintain target FPS
                frame_elapsed = time.time() - frame_start_time
                sleep_time = max(0, self.frame_time - frame_elapsed)
                if sleep_time > 0:
                    plt.pause(sleep_time)
                else:
                    plt.pause(0.001)  # Minimum pause for matplotlib
            
            step += 1
            
            if terminated or truncated:
                break
        
        print(f"   Episode completed: {step} steps, reward: {total_reward:.2f}")
        return total_reward, step

def main():
    parser = argparse.ArgumentParser(description='Enhanced Multi-Agent Smart Highway Visualizer')
    parser.add_argument('--model-path', type=str, default='trained_models/ppo_multi_agent_highway.zip',
                       help='Path to trained multi-agent model')
    parser.add_argument('--episodes', type=int, default=2,
                       help='Number of episodes to visualize')
    parser.add_argument('--max-steps', type=int, default=300,
                       help='Maximum steps per episode')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS for smooth animation')
    parser.add_argument('--sim-delay', type=float, default=0.3,
                       help='Delay between simulation steps (seconds)')
    parser.add_argument('--no-model', action='store_true',
                       help='Run with random actions instead of trained model')
    parser.add_argument('--num-agents', type=int, default=4,
                       help='Number of learning agents')
    
    args = parser.parse_args()
    
    # Create environment with enhanced settings for better collision scenarios
    env = create_multi_agent_environment(args.num_agents)
    # Increase spawn rate and reduce grid size for more interactions
    env.base_env.spawn_rate = 0.5  # Higher spawn rate
    env.base_env.max_vehicles = args.num_agents + 2  # More vehicles for interactions
    
    # Load model
    model = None
    if not args.no_model:
        try:
            model = PPO.load(args.model_path)
            print(f"✅ Loaded multi-agent model from {args.model_path}")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            print("🎲 Running with random actions")
    else:
        print("🎲 Running with random actions")
    
    # Create enhanced visualizer
    visualizer = SmoothMultiAgentVisualizer(env, model, target_fps=args.fps)
    
    print(f"🎬 Enhanced Multi-Agent Smart Highway Visualization")
    print(f"   Episodes: {args.episodes}")
    print(f"   Agents: {args.num_agents}")
    print(f"   Target FPS: {args.fps}")
    print(f"   Max steps: {args.max_steps}")
    print("   Press Ctrl+C to stop early")
    
    try:
        total_rewards = []
        total_steps = []
        
        for episode in range(args.episodes):
            print(f"\n🎭 Episode {episode + 1}/{args.episodes}")
            reward, steps = visualizer.run_smooth_episode(args.max_steps, args.sim_delay)
            total_rewards.append(reward)
            total_steps.append(steps)
            
            if episode < args.episodes - 1:
                print("   Press Enter to continue to next episode...")
                input()
        
        print(f"\n📊 Enhanced Multi-Agent Performance Summary:")
        print(f"   Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"   Average Steps: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
        print(f"   🤖 Smooth multi-agent coordination visualized!")
        
        plt.show()
        
    except KeyboardInterrupt:
        print("\n⏹️ Visualization stopped by user")
    except Exception as e:
        print(f"❌ Error during visualization: {e}")

if __name__ == "__main__":
    main() 