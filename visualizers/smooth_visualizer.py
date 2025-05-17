#!/usr/bin/env python3
"""
Smooth Multi-Agent Smart Highway Visualizer
==========================================

High-FPS visualization with:
- Smooth interpolated vehicle movements
- Real-time speed visualization
- Enhanced 6G communication effects
- Better collision detection display
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os
import argparse
import time
from collections import deque

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import PPO
from train_multi_agent_highway import create_multi_agent_environment

class SmoothVisualizer:
    """Smooth multi-agent visualizer with enhanced features."""
    
    def __init__(self, env, model=None, target_fps=30):
        self.env = env
        self.model = model
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        
        # Vehicle tracking for smooth movement
        self.prev_positions = {}
        self.vehicle_trails = {}
        self.speed_history = {}
        
        # Colors for different agents
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
                      '#FFEAA7', '#DDA0DD', '#FFB347', '#98D8C8']
        
        # Setup matplotlib
        plt.style.use('dark_background')
        self.fig, (self.ax_main, self.ax_info) = plt.subplots(
            1, 2, figsize=(16, 10), gridspec_kw={'width_ratios': [3, 1]}
        )
        
        self.setup_display()
        
    def setup_display(self):
        """Setup the display components."""
        # Main view
        self.ax_main.set_xlim(-1, self.env.base_env.grid_size[1] + 1)
        self.ax_main.set_ylim(-1, self.env.base_env.grid_size[0] + 1)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title('🤖 Smooth Multi-Agent Highway 🤖', fontsize=16, color='white')
        self.ax_main.set_facecolor('#0a0a0a')
        
        # Info panel
        self.ax_info.set_title('📊 Live Metrics', fontsize=12, color='white')
        self.ax_info.set_facecolor('#0a0a0a')
        self.ax_info.axis('off')
        
        # Draw highway infrastructure
        self.draw_highway()
        
    def draw_highway(self):
        """Draw the highway infrastructure."""
        # Lane markings
        for y in [2, 4, 6, 8]:
            self.ax_main.axhline(y, color='white', linewidth=2, alpha=0.8)
            for offset in [-0.4, -0.2, 0.2, 0.4]:
                self.ax_main.axhline(y + offset, color='gray', linewidth=1, alpha=0.4)
        
        for x in [2, 4, 6, 8]:
            self.ax_main.axvline(x, color='white', linewidth=2, alpha=0.8)
            for offset in [-0.4, -0.2, 0.2, 0.4]:
                self.ax_main.axvline(x + offset, color='gray', linewidth=1, alpha=0.4)
        
        # Intersections with 6G towers
        for intersection in self.env.base_env.intersections:
            x, y = intersection['position']
            
            # Intersection area
            circle = patches.Circle((x, y), intersection['size'], 
                                  facecolor='#2C3E50', edgecolor='#3498DB', 
                                  linewidth=2, alpha=0.7)
            self.ax_main.add_patch(circle)
            
            # 6G tower
            self.ax_main.plot(x, y, marker='^', color='#E74C3C', markersize=10)
            
            # Communication range
            range_circle = patches.Circle((x, y), 2.5, fill=False, 
                                        edgecolor='#E74C3C', linewidth=1, 
                                        alpha=0.3, linestyle='--')
            self.ax_main.add_patch(range_circle)
    
    def interpolate_position(self, vehicle_id, current_pos, progress):
        """Smoothly interpolate vehicle position."""
        if vehicle_id not in self.prev_positions:
            return current_pos
        
        prev_pos = self.prev_positions[vehicle_id]
        return prev_pos + (current_pos - prev_pos) * progress
    
    def update_trails(self, vehicle_id, pos):
        """Update vehicle movement trails."""
        if vehicle_id not in self.vehicle_trails:
            self.vehicle_trails[vehicle_id] = deque(maxlen=20)
        self.vehicle_trails[vehicle_id].append(pos)
    
    def draw_vehicle(self, vehicle_id, x, y, vx, vy, direction, lane_offset, interpolation=1.0):
        """Draw a vehicle with smooth movement and effects."""
        color = self.colors[vehicle_id % len(self.colors)]
        
        # Position with interpolation
        if vehicle_id in self.prev_positions and interpolation < 1.0:
            prev_x, prev_y = self.prev_positions[vehicle_id]
            x = prev_x + (x - prev_x) * interpolation
            y = prev_y + (y - prev_y) * interpolation
        
        # Update tracking
        self.update_trails(vehicle_id, (x, y))
        
        # Vehicle orientation and size
        speed = np.sqrt(vx**2 + vy**2)
        if direction == 0:  # Horizontal
            width, height = 0.7, 0.35
            angle = 0
            x_draw, y_draw = x, y + lane_offset
        else:  # Vertical
            width, height = 0.35, 0.7
            angle = 90
            x_draw, y_draw = x + lane_offset, y
        
        # Speed-based visual effects
        speed_factor = 1 + min(speed * 0.15, 0.4)
        alpha = min(0.9, 0.6 + speed * 0.3)
        
        # Vehicle body
        vehicle = patches.Rectangle(
            (x_draw - width*speed_factor/2, y_draw - height*speed_factor/2),
            width*speed_factor, height*speed_factor,
            angle=angle, facecolor=color, edgecolor='white',
            linewidth=2, alpha=alpha
        )
        self.ax_main.add_patch(vehicle)
        
        # Agent label
        self.ax_main.text(x_draw, y_draw, f'A{vehicle_id}', 
                         ha='center', va='center', fontsize=8, 
                         fontweight='bold', color='black')
        
        # Speed indicator
        if speed > 0.2:
            arrow_length = min(speed * 0.5, 1.0)
            if direction == 0:
                dx, dy = arrow_length * np.sign(vx), 0
            else:
                dx, dy = 0, arrow_length * np.sign(vy)
            
            self.ax_main.annotate('', 
                xy=(x_draw + dx*0.6, y_draw + dy*0.6),
                xytext=(x_draw - dx*0.4, y_draw - dy*0.4),
                arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.8)
            )
            
            # Speed text
            self.ax_main.text(x_draw + 0.8, y_draw - 0.8, f'{speed:.1f}', 
                             fontsize=7, color=color, fontweight='bold')
        
        # Movement trail
        if vehicle_id in self.vehicle_trails and len(self.vehicle_trails[vehicle_id]) > 1:
            trail = list(self.vehicle_trails[vehicle_id])
            for i in range(len(trail)-1):
                alpha_trail = (i+1) / len(trail) * 0.4
                x1, y1 = trail[i]
                x2, y2 = trail[i+1]
                self.ax_main.plot([x1, x2], [y1, y2], color=color, 
                                 linewidth=1.5, alpha=alpha_trail)
        
        return x_draw, y_draw
    
    def draw_communications(self, active_agents, info):
        """Draw 6G communication effects."""
        current_time = time.time()
        
        # V2V communication
        for i, agent1 in enumerate(active_agents):
            for agent2 in active_agents[i+1:]:
                x1, y1 = self.env.base_env.vehicles[agent1, 0], self.env.base_env.vehicles[agent1, 1]
                x2, y2 = self.env.base_env.vehicles[agent2, 0], self.env.base_env.vehicles[agent2, 1]
                
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if distance < 3.5:
                    # Animated signal strength
                    signal_strength = max(0.2, 1 - distance/3.5)
                    animation_phase = (current_time * 4) % (2 * np.pi)
                    alpha = 0.3 + 0.3 * np.sin(animation_phase) * signal_strength
                    
                    self.ax_main.plot([x1, x2], [y1, y2], 
                                     color='#00FF41', linewidth=2, 
                                     alpha=alpha, linestyle=':')
                    
                    # Data packet animation
                    packet_progress = (animation_phase / (2 * np.pi)) % 1.0
                    packet_x = x1 + (x2 - x1) * packet_progress
                    packet_y = y1 + (y2 - y1) * packet_progress
                    self.ax_main.plot(packet_x, packet_y, 'o', 
                                     color='#00FF41', markersize=3, alpha=0.8)
        
        # V2I communication
        for agent_id in active_agents:
            x, y = self.env.base_env.vehicles[agent_id, 0], self.env.base_env.vehicles[agent_id, 1]
            
            for intersection in self.env.base_env.intersections:
                int_x, int_y = intersection['position']
                distance = np.sqrt((x - int_x)**2 + (y - int_y)**2)
                
                if distance < 3.0:
                    signal_strength = max(0.3, 1 - distance/3.0)
                    pulse_phase = (current_time * 3) % (2 * np.pi)
                    alpha = 0.4 + 0.3 * np.sin(pulse_phase) * signal_strength
                    
                    self.ax_main.plot([x, int_x], [y, int_y], 
                                     color='#FF6B35', linewidth=2, 
                                     alpha=alpha, linestyle='-.')
    
    def update_info_panel(self, active_agents, info, step, total_reward):
        """Update the information panel."""
        self.ax_info.clear()
        self.ax_info.set_facecolor('#0a0a0a')
        self.ax_info.axis('off')
        
        # Calculate metrics
        collision_count = len(info.get('actual_collisions', []))
        prevented_count = len(info.get('collisions_prevented', []))
        messages_sent = info.get('messages_sent', 0)
        messages_delivered = info.get('messages_delivered', 0)
        
        # Vehicle speeds
        speeds = []
        for agent_id in active_agents:
            vx, vy = self.env.base_env.vehicles[agent_id, 2], self.env.base_env.vehicles[agent_id, 3]
            speeds.append(np.sqrt(vx**2 + vy**2))
        
        avg_speed = np.mean(speeds) if speeds else 0
        max_speed = max(speeds) if speeds else 0
        
        # Display metrics
        metrics = [
            f"🤖 Active Agents: {len(active_agents)}",
            f"⏱️ Step: {step}",
            f"🎯 Total Reward: {total_reward:.1f}",
            f"",
            f"🚗 Speed Metrics:",
            f"  Average: {avg_speed:.2f} m/s",
            f"  Maximum: {max_speed:.2f} m/s",
            f"",
            f"📡 Communication:",
            f"  Messages: {messages_sent}",
            f"  Delivered: {messages_delivered}",
            f"  Rate: {(messages_delivered/max(messages_sent,1)*100):.1f}%",
            f"",
            f"🛡️ Safety:",
            f"  Prevented: {prevented_count}",
            f"  Collisions: {collision_count}",
            f"  Prevention Rate: {info.get('collision_prevention_rate', 0):.1f}%",
            f"",
            f"🎬 Visualization:",
            f"  Target FPS: {self.target_fps}",
            f"  Simulation Time: {self.env.base_env.sim_time}s"
        ]
        
        for i, text in enumerate(metrics):
            color = 'white'
            if 'Prevented:' in text and prevented_count > 0:
                color = '#00FF00'
            elif 'Collisions:' in text and collision_count > 0:
                color = '#FF6B6B'
            elif 'Speed' in text or 'Average:' in text or 'Maximum:' in text:
                color = '#4ECDC4'
            elif 'Communication' in text or 'Messages' in text or 'Rate:' in text:
                color = '#00FF41'
                
            self.ax_info.text(0.05, 0.95 - i*0.04, text, 
                             transform=self.ax_info.transAxes,
                             fontsize=9, color=color, fontweight='bold' if not text.startswith('  ') else 'normal')
    
    def run_smooth_episode(self, max_steps=300, sim_delay=0.2):
        """Run episode with smooth visualization."""
        obs, info = self.env.reset()
        total_reward = 0
        step = 0
        
        print(f"🎬 Starting smooth visualization...")
        print(f"   Target FPS: {self.target_fps}")
        print(f"   Simulation delay: {sim_delay}s")
        
        # Store initial positions
        vehicles = self.env.base_env.vehicles
        for i in range(vehicles.shape[0]):
            if vehicles[i, 6] == 1:  # Active
                self.prev_positions[i] = np.array([vehicles[i, 0], vehicles[i, 1]])
        
        while step < max_steps:
            step_start_time = time.time()
            
            # Get actions
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
            
            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(actions)
            
            if isinstance(reward, dict):
                total_reward += sum(reward.values())
            else:
                total_reward += reward
            
            # Smooth animation with interpolation
            interpolation_frames = max(1, int(sim_delay / self.frame_time))
            
            for frame in range(interpolation_frames):
                frame_start = time.time()
                
                # Clear main plot
                self.ax_main.clear()
                self.setup_display()
                
                interpolation_progress = (frame + 1) / interpolation_frames
                vehicles = self.env.base_env.vehicles
                active_agents = []
                
                # Draw vehicles with interpolation
                for i in range(vehicles.shape[0]):
                    if vehicles[i, 6] == 1:  # Active
                        active_agents.append(i)
                        x, y = vehicles[i, 0], vehicles[i, 1]
                        vx, vy = vehicles[i, 2], vehicles[i, 3]
                        direction = int(vehicles[i, 4])
                        lane_offset = vehicles[i, 5]
                        
                        # Draw with interpolation
                        self.draw_vehicle(i, x, y, vx, vy, direction, lane_offset, interpolation_progress)
                        
                        # Destination marker
                        dest_x, dest_y = vehicles[i, 8], vehicles[i, 9]
                        color = self.colors[i % len(self.colors)]
                        self.ax_main.plot(dest_x, dest_y, marker='X', color=color, 
                                         markersize=8, alpha=0.7, markeredgecolor='white')
                
                # Draw communication effects
                self.draw_communications(active_agents, info)
                
                # Update info panel
                self.update_info_panel(active_agents, info, step, total_reward)
                
                # Update title
                avg_speed = np.mean([np.sqrt(vehicles[i, 2]**2 + vehicles[i, 3]**2) 
                                   for i in active_agents]) if active_agents else 0
                self.ax_main.set_title(
                    f'🤖 Smooth Multi-Agent Highway | Agents: {len(active_agents)} | '
                    f'Speed: {avg_speed:.1f} m/s | Step: {step} 🤖', 
                    fontsize=12, color='white'
                )
                
                # Maintain FPS
                frame_elapsed = time.time() - frame_start
                sleep_time = max(0, self.frame_time - frame_elapsed)
                plt.pause(max(0.001, sleep_time))
            
            # Update previous positions
            for i in active_agents:
                self.prev_positions[i] = np.array([vehicles[i, 0], vehicles[i, 1]])
            
            step += 1
            
            if terminated or truncated:
                break
        
        print(f"   Episode completed: {step} steps, reward: {total_reward:.2f}")
        return total_reward, step

def main():
    parser = argparse.ArgumentParser(description='Smooth Multi-Agent Visualizer')
    parser.add_argument('--model-path', type=str, default='trained_models/ppo_multi_agent_highway.zip')
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--sim-delay', type=float, default=0.2)
    parser.add_argument('--no-model', action='store_true')
    parser.add_argument('--num-agents', type=int, default=4)
    
    args = parser.parse_args()
    
    # Create environment with settings that promote interactions
    # Create environment with more vehicles for better interactions
    from environments.smart_highway_env import SmartHighwayEnv
    from train_multi_agent_highway import MultiAgentWrapper
    
    base_env = SmartHighwayEnv(
        grid_size=(10, 10),
        max_vehicles=args.num_agents + 4,  # More vehicles for interactions
        spawn_rate=0.6,  # Higher spawn rate
        multi_agent=True,
        debug=False
    )
    env = MultiAgentWrapper(base_env)
    env.num_agents = args.num_agents  # Set for compatibility
    
    # Load model
    model = None
    if not args.no_model:
        try:
            model = PPO.load(args.model_path)
            print(f"✅ Loaded model from {args.model_path}")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            print("🎲 Using random actions")
    else:
        print("🎲 Using random actions")
    
    # Create visualizer
    visualizer = SmoothVisualizer(env, model, target_fps=args.fps)
    
    print(f"🎬 Smooth Multi-Agent Visualization")
    print(f"   Episodes: {args.episodes}")
    print(f"   Target FPS: {args.fps}")
    print(f"   Simulation delay: {args.sim_delay}s")
    
    try:
        for episode in range(args.episodes):
            print(f"\n🎭 Episode {episode + 1}/{args.episodes}")
            reward, steps = visualizer.run_smooth_episode(args.max_steps, args.sim_delay)
            
            if episode < args.episodes - 1:
                input("   Press Enter for next episode...")
        
        plt.show()
        
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user")

if __name__ == "__main__":
    main() 