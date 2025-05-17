#!/usr/bin/env python3
"""
Multi-Agent Smart Highway Visualizer
===================================

Visualize multiple learning agents coordinating in the smart highway environment
with 6G communication, intersection management, and real-time performance metrics.
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

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from stable_baselines3 import PPO
from train_multi_agent_highway import create_multi_agent_environment

class MultiAgentHighwayVisualizer:
    """Visualizer for multi-agent smart highway simulation."""
    
    def __init__(self, env, model=None, window_size=(12, 10)):
        self.env = env
        self.model = model
        self.window_size = window_size
        
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
        
        # Performance tracking
        self.performance_history = {
            'rewards': deque(maxlen=100),
            'collisions': deque(maxlen=100),
            'coordination_score': deque(maxlen=100),
            'episode_lengths': deque(maxlen=100)
        }
        
        # Communication visualization
        self.communication_links = []
        self.intersection_reservations = []
        
        # Setup matplotlib
        plt.style.use('dark_background')
        self.fig, (self.ax_main, self.ax_metrics) = plt.subplots(
            1, 2, figsize=window_size, gridspec_kw={'width_ratios': [2, 1]}
        )
        
        self.setup_visualization()
        
    def setup_visualization(self):
        """Setup the visualization components."""
        # Main highway view
        self.ax_main.set_xlim(-1, self.env.base_env.grid_size[1] + 1)
        self.ax_main.set_ylim(-1, self.env.base_env.grid_size[0] + 1)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title('🤖 Multi-Agent Smart Highway 🤖', fontsize=16, color='white')
        self.ax_main.set_facecolor('#1a1a1a')
        
        # Draw highway infrastructure
        self.draw_highway_infrastructure()
        
        # Metrics panel
        self.ax_metrics.set_title('📊 Multi-Agent Performance', fontsize=12, color='white')
        self.ax_metrics.set_facecolor('#1a1a1a')
        
        # Initialize vehicle patches
        self.vehicle_patches = {}
        self.agent_labels = {}
        self.communication_lines = []
        
    def draw_highway_infrastructure(self):
        """Draw the highway infrastructure with lanes and intersections."""
        # Highway lanes (horizontal)
        lane_y_positions = [2, 4, 6, 8]
        for y in lane_y_positions:
            # Lane lines
            self.ax_main.axhline(y, color='white', linewidth=1, alpha=0.3, linestyle='--')
            
            # Multiple sub-lanes
            for offset in [-0.4, -0.2, 0.2, 0.4]:
                self.ax_main.axhline(y + offset, color='gray', linewidth=0.5, alpha=0.2)
        
        # Highway lanes (vertical)  
        lane_x_positions = [2, 4, 6, 8]
        for x in lane_x_positions:
            # Lane lines
            self.ax_main.axvline(x, color='white', linewidth=1, alpha=0.3, linestyle='--')
            
            # Multiple sub-lanes
            for offset in [-0.4, -0.2, 0.2, 0.4]:
                self.ax_main.axvline(x + offset, color='gray', linewidth=0.5, alpha=0.2)
        
        # Draw intersections with 6G infrastructure
        intersections = self.env.base_env.intersections
        for intersection in intersections:
            x, y = intersection['position']
            
            # Intersection area
            intersection_patch = patches.Circle(
                (x, y), intersection['size'], 
                facecolor='#2C3E50', edgecolor='#3498DB', linewidth=2, alpha=0.7
            )
            self.ax_main.add_patch(intersection_patch)
            
            # 6G tower symbol
            self.ax_main.plot(x, y, marker='^', color='#E74C3C', markersize=8, alpha=0.8)
            
            # Intersection label
            self.ax_main.text(x, y-0.8, f"6G-{intersection['id'].split('_')[-2:]}", 
                            ha='center', va='center', fontsize=8, color='#3498DB')
    
    def update_visualization(self, frame):
        """Update the visualization for each frame."""
        try:
            # Get current state
            obs, info = self.env.reset() if frame == 0 else self.get_current_state()
            
            # Clear previous vehicles and communication lines
            for patch in list(self.vehicle_patches.values()):
                patch.remove()
            for label in list(self.agent_labels.values()):
                label.remove()
            for line in self.communication_lines:
                line.remove()
                
            self.vehicle_patches.clear()
            self.agent_labels.clear()
            self.communication_lines.clear()
            
            # Get vehicle states
            vehicles = self.env.base_env.vehicles
            active_agents = []
            
            # Draw active vehicles
            for i in range(self.env.base_env.max_vehicles):
                if vehicles[i, 6] == 1:  # Active vehicle
                    active_agents.append(i)
                    x, y = vehicles[i, 0], vehicles[i, 1]
                    direction = int(vehicles[i, 4])
                    lane_offset = vehicles[i, 5]
                    
                    # Vehicle color
                    color = self.agent_colors[i % len(self.agent_colors)]
                    
                    # Vehicle shape based on direction
                    if direction == 0:  # L2R (horizontal)
                        width, height = 0.6, 0.3
                        angle = 0
                        x_draw, y_draw = x, y + lane_offset
                    else:  # T2B (vertical)
                        width, height = 0.3, 0.6
                        angle = 90
                        x_draw, y_draw = x + lane_offset, y
                    
                    # Vehicle body
                    vehicle_patch = patches.Rectangle(
                        (x_draw - width/2, y_draw - height/2), width, height,
                        angle=angle, facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9
                    )
                    self.vehicle_patches[i] = vehicle_patch
                    self.ax_main.add_patch(vehicle_patch)
                    
                    # Agent ID label
                    label = self.ax_main.text(
                        x_draw, y_draw, f'A{i}', ha='center', va='center', 
                        fontsize=8, fontweight='bold', color='black'
                    )
                    self.agent_labels[i] = label
                    
                    # Destination marker
                    dest_x, dest_y = vehicles[i, 8], vehicles[i, 9]
                    self.ax_main.plot(dest_x, dest_y, marker='X', color=color, 
                                    markersize=10, alpha=0.7, markeredgecolor='white')
                    
                    # Speed indicator (trail effect)
                    speed = np.sqrt(vehicles[i, 2]**2 + vehicles[i, 3]**2)
                    if speed > 0.1:
                        trail_length = min(speed * 0.5, 1.0)
                        if direction == 0:  # Horizontal
                            trail_x = [x_draw - trail_length, x_draw]
                            trail_y = [y_draw, y_draw]
                        else:  # Vertical
                            trail_x = [x_draw, x_draw]
                            trail_y = [y_draw - trail_length, y_draw]
                        
                        trail_line = self.ax_main.plot(trail_x, trail_y, color=color, 
                                                     linewidth=3, alpha=0.5)[0]
                        self.communication_lines.append(trail_line)
            
            # Draw 6G communication links
            self.draw_communication_links(active_agents)
            
            # Update performance metrics
            self.update_performance_metrics(active_agents, info)
            
            # Update title with current stats
            num_active = len(active_agents)
            self.ax_main.set_title(
                f'🤖 Multi-Agent Smart Highway | Active Agents: {num_active} | '
                f'Time: {self.env.base_env.sim_time}s 🤖', 
                fontsize=14, color='white'
            )
            
        except Exception as e:
            print(f"Visualization error: {e}")
            pass
    
    def draw_communication_links(self, active_agents):
        """Draw 6G communication links between agents."""
        # V2V communication (vehicle to vehicle)
        for i, agent1 in enumerate(active_agents):
            for agent2 in active_agents[i+1:]:
                x1, y1 = self.env.base_env.vehicles[agent1, 0], self.env.base_env.vehicles[agent1, 1]
                x2, y2 = self.env.base_env.vehicles[agent2, 0], self.env.base_env.vehicles[agent2, 1]
                
                # Only show communication if vehicles are close enough
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if distance < 3.0:  # Communication range
                    comm_line = self.ax_main.plot([x1, x2], [y1, y2], 
                                                color='#00FF41', linewidth=1, alpha=0.3, linestyle=':')[0]
                    self.communication_lines.append(comm_line)
        
        # V2I communication (vehicle to infrastructure)
        intersections = self.env.base_env.intersections
        for agent_id in active_agents:
            x, y = self.env.base_env.vehicles[agent_id, 0], self.env.base_env.vehicles[agent_id, 1]
            
            # Find nearby intersections
            for intersection in intersections:
                int_x, int_y = intersection['position']
                distance = np.sqrt((x - int_x)**2 + (y - int_y)**2)
                
                if distance < 2.5:  # V2I communication range
                    comm_line = self.ax_main.plot([x, int_x], [y, int_y], 
                                                color='#FF6B35', linewidth=1.5, alpha=0.5, linestyle='-.')[0]
                    self.communication_lines.append(comm_line)
    
    def update_performance_metrics(self, active_agents, info):
        """Update the performance metrics panel."""
        self.ax_metrics.clear()
        self.ax_metrics.set_facecolor('#1a1a1a')
        self.ax_metrics.set_title('📊 Multi-Agent Performance', fontsize=12, color='white')
        
        # Calculate metrics
        total_reward = sum(obs.get('reward', 0) for obs in [{}])  # Placeholder
        collision_count = len(info.get('actual_collisions', []))
        coordination_score = info.get('collision_prevention_rate', 0)
        
        # Update history
        self.performance_history['rewards'].append(total_reward)
        self.performance_history['collisions'].append(collision_count)
        self.performance_history['coordination_score'].append(coordination_score)
        
        # Current metrics display
        metrics_text = [
            f"🤖 Active Agents: {len(active_agents)}",
            f"📡 6G Messages: {info.get('messages_sent', 0)}",
            f"🛡️ Collisions Prevented: {len(info.get('collisions_prevented', []))}",
            f"💥 Actual Collisions: {collision_count}",
            f"📈 Coordination: {coordination_score:.1f}%",
            f"⏱️ Simulation Time: {self.env.base_env.sim_time}s"
        ]
        
        for i, text in enumerate(metrics_text):
            self.ax_metrics.text(0.05, 0.9 - i*0.1, text, transform=self.ax_metrics.transAxes,
                               fontsize=10, color='white', fontweight='bold')
        
        # Mini performance graphs
        if len(self.performance_history['rewards']) > 1:
            # Reward trend
            rewards = list(self.performance_history['rewards'])
            if rewards:
                y_pos = 0.3
                self.ax_metrics.text(0.05, y_pos, "Reward Trend:", transform=self.ax_metrics.transAxes,
                                   fontsize=9, color='#4ECDC4')
                
                # Simple sparkline
                x_vals = np.linspace(0.1, 0.9, len(rewards))
                y_vals = np.array(rewards)
                if len(y_vals) > 1:
                    y_normalized = 0.15 + 0.1 * (y_vals - np.min(y_vals)) / (np.max(y_vals) - np.min(y_vals) + 1e-6)
                    self.ax_metrics.plot(x_vals, y_normalized, color='#4ECDC4', linewidth=2, alpha=0.8)
        
        # Agent status
        agent_status_y = 0.05
        self.ax_metrics.text(0.05, agent_status_y, "Agent Status:", transform=self.ax_metrics.transAxes,
                           fontsize=9, color='#FFD93D')
        
        for i, agent_id in enumerate(active_agents[:4]):  # Show first 4 agents
            color = self.agent_colors[agent_id % len(self.agent_colors)]
            speed = np.sqrt(self.env.base_env.vehicles[agent_id, 2]**2 + 
                          self.env.base_env.vehicles[agent_id, 3]**2)
            status_text = f"A{agent_id}: {speed:.1f} m/s"
            self.ax_metrics.text(0.1 + (i%2)*0.4, agent_status_y - 0.05 - (i//2)*0.04, 
                               status_text, transform=self.ax_metrics.transAxes,
                               fontsize=8, color=color)
        
        self.ax_metrics.set_xlim(0, 1)
        self.ax_metrics.set_ylim(0, 1)
        self.ax_metrics.axis('off')
    
    def get_current_state(self):
        """Get current environment state."""
        # This is a placeholder - in real implementation, you'd get the actual current state
        return {}, {}
    
    def run_episode(self, max_steps=500, delay=0.1):
        """Run one episode with visualization."""
        obs, info = self.env.reset()
        total_reward = 0
        step = 0
        
        print(f"🎬 Starting multi-agent episode visualization...")
        print(f"   Active agents: {len(obs) if isinstance(obs, dict) else 'N/A'}")
        
        while step < max_steps:
            # Get actions from model or random
            if self.model:
                if isinstance(obs, dict):
                    # Multi-agent: need to handle dict observations
                    actions = []
                    for i in range(self.env.num_agents):
                        agent_obs = obs.get(f"agent_{i}", np.zeros(self.env.base_env.single_obs_space.shape))
                        action, _ = self.model.predict(agent_obs, deterministic=True)
                        actions.append(action)
                else:
                    actions, _ = self.model.predict(obs, deterministic=True)
            else:
                # Random actions
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
            
            # Update visualization
            self.update_visualization(step)
            
            # Pause for visualization
            plt.pause(delay)
            
            step += 1
            
            if terminated or truncated:
                break
        
        print(f"   Episode completed: {step} steps, reward: {total_reward:.2f}")
        return total_reward, step

def main():
    parser = argparse.ArgumentParser(description='Multi-Agent Smart Highway Visualizer')
    parser.add_argument('--model-path', type=str, default='trained_models/ppo_multi_agent_highway.zip',
                       help='Path to trained multi-agent model')
    parser.add_argument('--episodes', type=int, default=3,
                       help='Number of episodes to visualize')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per episode')
    parser.add_argument('--delay', type=float, default=0.1,
                       help='Delay between steps (seconds)')
    parser.add_argument('--no-model', action='store_true',
                       help='Run with random actions instead of trained model')
    parser.add_argument('--num-agents', type=int, default=4,
                       help='Number of learning agents')
    
    args = parser.parse_args()
    
    # Create environment
    env = create_multi_agent_environment(args.num_agents)
    
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
    
    # Create visualizer
    visualizer = MultiAgentHighwayVisualizer(env, model)
    
    print(f"🎬 Multi-Agent Smart Highway Visualization")
    print(f"   Episodes: {args.episodes}")
    print(f"   Agents: {args.num_agents}")
    print(f"   Max steps: {args.max_steps}")
    print("   Press Ctrl+C to stop early")
    
    try:
        total_rewards = []
        total_steps = []
        
        for episode in range(args.episodes):
            print(f"\n🎭 Episode {episode + 1}/{args.episodes}")
            reward, steps = visualizer.run_episode(args.max_steps, args.delay)
            total_rewards.append(reward)
            total_steps.append(steps)
            
            if episode < args.episodes - 1:
                print("   Press Enter to continue to next episode...")
                input()
        
        print(f"\n📊 Multi-Agent Performance Summary:")
        print(f"   Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"   Average Steps: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
        print(f"   🤖 Multi-agent coordination visualized!")
        
        plt.show()
        
    except KeyboardInterrupt:
        print("\n⏹️ Visualization stopped by user")
    except Exception as e:
        print(f"❌ Error during visualization: {e}")

if __name__ == "__main__":
    main() 