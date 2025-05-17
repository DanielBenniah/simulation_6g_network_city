#!/usr/bin/env python3
"""
Test Smooth Visualization with Enhanced Collision Detection
=========================================================

Simple test to demonstrate:
- Smooth 30+ FPS vehicle movements
- Enhanced 6G communication visualization  
- Real collision detection and prevention
- Speed-based visual effects
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from environments.smart_highway_env import SmartHighwayEnv
from stable_baselines3 import PPO

def run_smooth_demo():
    """Run a smooth visualization demo."""
    print("🎬 Starting Enhanced Smart Highway Demo")
    print("="*50)
    
    # Create environment with enhanced collision detection
    env = SmartHighwayEnv(
        grid_size=(10, 10),
        max_vehicles=8,  # Manageable number for visualization
        spawn_rate=0.7,  # High spawn rate for interactions
        multi_agent=False,  # Keep it simple
        debug=True  # Enable debug for collision info
    )
    
    # Try to load trained model
    model = None
    try:
        model = PPO.load("trained_models/ppo_smart_highway")
        print("✅ Loaded trained model")
    except:
        print("⚠️ Using random actions")
    
    # Setup matplotlib for smooth animation
    plt.style.use('dark_background')
    fig, (ax_main, ax_info) = plt.subplots(1, 2, figsize=(16, 8), 
                                          gridspec_kw={'width_ratios': [3, 1]})
    
    # Colors for vehicles
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
              '#FFEAA7', '#DDA0DD', '#FFB347', '#98D8C8']
    
    def setup_highway_view():
        """Setup the highway visualization."""
        ax_main.clear()
        ax_main.set_xlim(-1, 11)
        ax_main.set_ylim(-1, 11)
        ax_main.set_aspect('equal')
        ax_main.set_facecolor('#0a0a0a')
        ax_main.set_title('🚗 Enhanced Smart Highway with 6G Communication 🚗', 
                         fontsize=14, color='white')
        
        # Draw lanes
        for y in [2, 4, 6, 8]:
            ax_main.axhline(y, color='white', linewidth=2, alpha=0.8)
            for offset in [-0.3, -0.1, 0.1, 0.3]:
                ax_main.axhline(y + offset, color='gray', linewidth=1, alpha=0.3)
        
        for x in [2, 4, 6, 8]:
            ax_main.axvline(x, color='white', linewidth=2, alpha=0.8)
            for offset in [-0.3, -0.1, 0.1, 0.3]:
                ax_main.axvline(x + offset, color='gray', linewidth=1, alpha=0.3)
        
        # Draw intersections with 6G towers
        for intersection in env.intersections:
            x, y = intersection['position']
            
            # Intersection circle
            circle = patches.Circle((x, y), intersection['size'], 
                                  facecolor='#2C3E50', edgecolor='#3498DB', 
                                  linewidth=2, alpha=0.7)
            ax_main.add_patch(circle)
            
            # 6G tower
            ax_main.plot(x, y, marker='^', color='#E74C3C', markersize=12)
            
            # Communication range
            range_circle = patches.Circle((x, y), 2.5, fill=False, 
                                        edgecolor='#E74C3C', linewidth=1, 
                                        alpha=0.3, linestyle='--')
            ax_main.add_patch(range_circle)
    
    def draw_vehicle(i, x, y, vx, vy, direction, lane_offset):
        """Draw a vehicle with enhanced visual effects."""
        color = colors[i % len(colors)]
        speed = np.sqrt(vx**2 + vy**2)
        
        # Vehicle position
        if direction == 0:  # Horizontal
            width, height = 0.8, 0.4
            angle = 0
            x_draw, y_draw = x, y + lane_offset
        else:  # Vertical
            width, height = 0.4, 0.8
            angle = 90
            x_draw, y_draw = x + lane_offset, y
        
        # Speed-based effects
        speed_factor = 1 + min(speed * 0.1, 0.3)
        alpha = min(0.9, 0.7 + speed * 0.2)
        
        # Vehicle body
        vehicle = patches.Rectangle(
            (x_draw - width*speed_factor/2, y_draw - height*speed_factor/2),
            width*speed_factor, height*speed_factor,
            angle=angle, facecolor=color, edgecolor='white',
            linewidth=2, alpha=alpha
        )
        ax_main.add_patch(vehicle)
        
        # Vehicle ID
        ax_main.text(x_draw, y_draw, f'V{i}', ha='center', va='center', 
                    fontsize=9, fontweight='bold', color='black')
        
        # Speed arrow
        if speed > 0.3:
            arrow_length = min(speed * 0.4, 1.0)
            if direction == 0:
                dx, dy = arrow_length * np.sign(vx), 0
            else:
                dx, dy = 0, arrow_length * np.sign(vy)
            
            ax_main.annotate('', 
                xy=(x_draw + dx*0.7, y_draw + dy*0.7),
                xytext=(x_draw - dx*0.3, y_draw - dy*0.3),
                arrowprops=dict(arrowstyle='->', color=color, lw=2, alpha=0.8)
            )
            
            # Speed text
            ax_main.text(x_draw + 1, y_draw - 1, f'{speed:.1f}', 
                        fontsize=8, color=color, fontweight='bold')
        
        # Destination
        dest_x, dest_y = env.vehicles[i, 8], env.vehicles[i, 9]
        ax_main.plot(dest_x, dest_y, marker='X', color=color, 
                    markersize=10, alpha=0.7, markeredgecolor='white')
        
        return x_draw, y_draw
    
    def draw_communications(active_vehicles, info):
        """Draw 6G communication effects."""
        current_time = time.time()
        
        # V2V communication
        for i in range(len(active_vehicles)):
            for j in range(i+1, len(active_vehicles)):
                v1, v2 = active_vehicles[i], active_vehicles[j]
                x1, y1 = env.vehicles[v1, 0], env.vehicles[v1, 1]
                x2, y2 = env.vehicles[v2, 0], env.vehicles[v2, 1]
                
                distance = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if distance < 3.5:
                    # Animated signal
                    signal_strength = max(0.2, 1 - distance/3.5)
                    animation_phase = (current_time * 3) % (2 * np.pi)
                    alpha = 0.3 + 0.3 * np.sin(animation_phase) * signal_strength
                    
                    ax_main.plot([x1, x2], [y1, y2], 
                                color='#00FF41', linewidth=2, 
                                alpha=alpha, linestyle=':')
        
        # V2I communication
        for v in active_vehicles:
            x, y = env.vehicles[v, 0], env.vehicles[v, 1]
            
            for intersection in env.intersections:
                int_x, int_y = intersection['position']
                distance = np.sqrt((x - int_x)**2 + (y - int_y)**2)
                
                if distance < 3.0:
                    signal_strength = max(0.3, 1 - distance/3.0)
                    pulse_phase = (current_time * 2) % (2 * np.pi)
                    alpha = 0.4 + 0.3 * np.sin(pulse_phase) * signal_strength
                    
                    ax_main.plot([x, int_x], [y, int_y], 
                                color='#FF6B35', linewidth=2, 
                                alpha=alpha, linestyle='-.')
    
    def update_info_panel(step, info, active_vehicles):
        """Update information panel."""
        ax_info.clear()
        ax_info.set_facecolor('#0a0a0a')
        ax_info.axis('off')
        
        # Calculate metrics
        collision_count = len(info.get('actual_collisions', []))
        prevented_count = len(info.get('collisions_prevented', []))
        messages_sent = info.get('messages_sent', 0)
        messages_delivered = info.get('messages_delivered', 0)
        
        # Vehicle speeds
        speeds = [np.sqrt(env.vehicles[i, 2]**2 + env.vehicles[i, 3]**2) 
                 for i in active_vehicles]
        avg_speed = np.mean(speeds) if speeds else 0
        
        # Metrics
        metrics = [
            f"🎬 Enhanced Demo",
            f"⏱️ Step: {step}",
            f"🚗 Active Vehicles: {len(active_vehicles)}",
            f"⚡ Avg Speed: {avg_speed:.2f} m/s",
            "",
            f"📡 6G Communication:",
            f"  Messages: {messages_sent}",
            f"  Delivered: {messages_delivered}",
            f"  Rate: {(messages_delivered/max(messages_sent,1)*100):.1f}%",
            "",
            f"🛡️ Safety System:",
            f"  Prevented: {prevented_count}",
            f"  Collisions: {collision_count}",
            f"  Rate: {info.get('collision_prevention_rate', 0):.1f}%",
            "",
            f"🎯 Status:",
            f"  Sim Time: {env.sim_time}s",
            f"  Total Spawned: {env.total_spawned}",
            f"  Completed: {env.total_completed}"
        ]
        
        for i, text in enumerate(metrics):
            color = 'white'
            if 'Prevented:' in text and prevented_count > 0:
                color = '#00FF00'
            elif 'Collisions:' in text and collision_count > 0:
                color = '#FF6B6B'
            elif 'Speed' in text or 'Rate:' in text:
                color = '#4ECDC4'
            
            ax_info.text(0.05, 0.95 - i*0.045, text, 
                        transform=ax_info.transAxes,
                        fontsize=10, color=color, 
                        fontweight='bold' if not text.startswith('  ') else 'normal')
    
    # Run simulation
    obs, _ = env.reset()
    step = 0
    target_fps = 25
    frame_time = 1.0 / target_fps
    
    print(f"🎯 Running demo at {target_fps} FPS")
    print("   Watch for:")
    print("   • Smooth vehicle movements")
    print("   • 6G communication effects")
    print("   • Collision detection & prevention")
    print("   • Speed-based visual effects")
    print("   Press Ctrl+C to stop")
    
    try:
        while step < 200:
            frame_start = time.time()
            
            # Get action
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.random.randint(0, 3)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Visualization
            setup_highway_view()
            
            # Find active vehicles
            active_vehicles = []
            for i in range(env.max_vehicles):
                if env.vehicles[i, 6] == 1:  # Active
                    active_vehicles.append(i)
                    x, y = env.vehicles[i, 0], env.vehicles[i, 1]
                    vx, vy = env.vehicles[i, 2], env.vehicles[i, 3]
                    direction = int(env.vehicles[i, 4])
                    lane_offset = env.vehicles[i, 5]
                    
                    draw_vehicle(i, x, y, vx, vy, direction, lane_offset)
            
            # Draw communication effects
            draw_communications(active_vehicles, info)
            
            # Update info panel
            update_info_panel(step, info, active_vehicles)
            
            # Print collision info
            if info.get('actual_collisions'):
                print(f"💥 Step {step}: {len(info['actual_collisions'])} collisions!")
            if info.get('collisions_prevented'):
                print(f"🛡️ Step {step}: Prevented {len(info['collisions_prevented'])} collisions")
            
            # Maintain FPS
            frame_elapsed = time.time() - frame_start
            sleep_time = max(0, frame_time - frame_elapsed)
            plt.pause(max(0.001, sleep_time))
            
            step += 1
            
            if terminated or truncated:
                print(f"Episode ended at step {step}")
                break
        
        print("\n🎉 Demo completed!")
        print(f"Final stats:")
        print(f"  Total collisions: {env.collision_count}")
        print(f"  Vehicles spawned: {env.total_spawned}")
        print(f"  Vehicles completed: {env.total_completed}")
        
        plt.show()
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo stopped by user")

if __name__ == "__main__":
    run_smooth_demo() 