#!/usr/bin/env python3
"""
Simplified Smart Highway Environment for Training
================================================

A streamlined version of SmartHighwayEnv optimized for fast RL training:
- Reduced computational complexity
- Simplified collision detection
- Optimized observation space
- Better reward scaling
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SimpleHighwayEnv:
    """Simplified highway environment optimized for training speed."""
    
    def __init__(self, grid_size=(10, 10), max_vehicles=8, spawn_rate=0.2, debug=False):
        """Initialize simplified environment with reduced complexity."""
        self.grid_size = grid_size
        self.max_vehicles = max_vehicles  # Reduced from 24 to 8
        self.spawn_rate = spawn_rate  # Reduced from 0.6 to 0.2
        self.debug = debug
        
        # Action space: 0=maintain, 1=accelerate, 2=brake
        self.action_space = spaces.Discrete(3)
        
        # Simplified vehicle state: [x, y, vx, vy, direction, active]
        self.vehicles = np.zeros((max_vehicles, 6), dtype=np.float32)
        
        # Speed parameters
        self.min_speed = 0.5
        self.max_speed = 1.5  # Reduced for stability
        self.acceleration = 0.2
        self.brake_factor = 0.7
        
        # Simulation state
        self.sim_time = 0
        self.collision_count = 0
        self.completed_journeys = 0
        self.steps_since_spawn = 0
        
    def reset(self, **kwargs):
        """Reset environment."""
        self.vehicles.fill(0)
        self.sim_time = 0
        self.collision_count = 0
        self.completed_journeys = 0
        self.steps_since_spawn = 0
        
        # Spawn agent vehicle (vehicle 0)
        self._spawn_agent()
        
        # Spawn 2-3 other vehicles
        for _ in range(np.random.randint(2, 4)):
            self._spawn_vehicle()
            
        return self._get_observation(), {}
    
    def _spawn_agent(self):
        """Spawn the agent vehicle."""
        # Agent always spawns horizontally (left to right)
        self.vehicles[0] = [
            0,  # x: start at left edge
            5,  # y: middle lane
            1.0,  # vx: moving right
            0.0,  # vy: no vertical movement
            0,  # direction: 0=horizontal
            1   # active
        ]
    
    def _spawn_vehicle(self):
        """Spawn a random vehicle."""
        for i in range(1, self.max_vehicles):
            if self.vehicles[i, 5] == 0:  # Not active
                direction = np.random.randint(0, 2)
                
                if direction == 0:  # Horizontal (left to right)
                    self.vehicles[i] = [
                        0,  # x: left edge
                        np.random.choice([3, 5, 7]),  # y: lane positions
                        np.random.uniform(0.7, 1.3),  # vx
                        0,  # vy
                        0,  # direction
                        1   # active
                    ]
                else:  # Vertical (top to bottom)
                    self.vehicles[i] = [
                        np.random.choice([3, 5, 7]),  # x: lane positions
                        0,  # y: top edge
                        0,  # vx
                        np.random.uniform(0.7, 1.3),  # vy
                        1,  # direction
                        1   # active
                    ]
                return True
        return False
    
    def step(self, action):
        """Execute one step."""
        self.sim_time += 1
        self.steps_since_spawn += 1
        
        # Apply agent action
        if self.vehicles[0, 5] == 1:  # Agent active
            self._apply_action(0, action)
        
        # Simple random actions for other vehicles
        for i in range(1, self.max_vehicles):
            if self.vehicles[i, 5] == 1:
                # Simple behavior: mostly maintain speed
                random_action = np.random.choice([0, 0, 0, 1, 2], p=[0.7, 0.1, 0.1, 0.1, 0.0])
                self._apply_action(i, random_action)
        
        # Move vehicles
        self._move_vehicles()
        
        # Check collisions (simplified)
        collision_penalty = self._check_collisions()
        
        # Check completions
        self._check_completions()
        
        # Spawn new vehicles occasionally
        if self.steps_since_spawn > 20 and np.random.random() < self.spawn_rate:
            if self._spawn_vehicle():
                self.steps_since_spawn = 0
        
        # Calculate reward
        reward = self._calculate_reward(collision_penalty)
        
        # Check if agent completed journey or collided
        terminated = (self.vehicles[0, 5] == 0)  # Agent deactivated
        truncated = self.sim_time > 500  # Max episode length
        
        observation = self._get_observation()
        info = {
            'collision_count': self.collision_count,
            'completed_journeys': self.completed_journeys,
            'actual_collisions': [{'count': collision_penalty}] if collision_penalty > 0 else []
        }
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, vehicle_id, action):
        """Apply action to vehicle."""
        if self.vehicles[vehicle_id, 5] == 0:  # Not active
            return
            
        direction = int(self.vehicles[vehicle_id, 4])
        
        if action == 1:  # Accelerate
            if direction == 0:  # Horizontal
                self.vehicles[vehicle_id, 2] = min(self.max_speed, 
                                                 self.vehicles[vehicle_id, 2] + self.acceleration)
            else:  # Vertical
                self.vehicles[vehicle_id, 3] = min(self.max_speed, 
                                                 self.vehicles[vehicle_id, 3] + self.acceleration)
                
        elif action == 2:  # Brake
            if direction == 0:  # Horizontal
                self.vehicles[vehicle_id, 2] = max(self.min_speed, 
                                                 self.vehicles[vehicle_id, 2] * self.brake_factor)
            else:  # Vertical
                self.vehicles[vehicle_id, 3] = max(self.min_speed, 
                                                 self.vehicles[vehicle_id, 3] * self.brake_factor)
    
    def _move_vehicles(self):
        """Move all vehicles."""
        for i in range(self.max_vehicles):
            if self.vehicles[i, 5] == 0:  # Not active
                continue
                
            # Update position
            self.vehicles[i, 0] += self.vehicles[i, 2]  # x += vx
            self.vehicles[i, 1] += self.vehicles[i, 3]  # y += vy
    
    def _check_collisions(self):
        """Simplified collision detection."""
        collision_penalty = 0
        
        # Check agent against other vehicles
        if self.vehicles[0, 5] == 1:  # Agent active
            agent_x, agent_y = self.vehicles[0, 0], self.vehicles[0, 1]
            
            for i in range(1, self.max_vehicles):
                if self.vehicles[i, 5] == 1:  # Other vehicle active
                    other_x, other_y = self.vehicles[i, 0], self.vehicles[i, 1]
                    distance = np.sqrt((agent_x - other_x)**2 + (agent_y - other_y)**2)
                    
                    if distance < 0.5:  # Collision
                        collision_penalty = 1
                        self.collision_count += 1
                        # Don't deactivate vehicles - just penalty
                        if self.debug:
                            print(f"[COLLISION] Agent vs Vehicle {i}")
        
        return collision_penalty
    
    def _check_completions(self):
        """Check if vehicles completed their journeys."""
        for i in range(self.max_vehicles):
            if self.vehicles[i, 5] == 0:  # Not active
                continue
                
            x, y = self.vehicles[i, 0], self.vehicles[i, 1]
            
            # Check if left the grid (completed journey)
            if (x > self.grid_size[1] or x < -1 or 
                y > self.grid_size[0] or y < -1):
                
                self.vehicles[i, 5] = 0  # Deactivate
                self.completed_journeys += 1
                
                if i == 0:  # Agent completed
                    if self.debug:
                        print(f"[SUCCESS] Agent completed journey at step {self.sim_time}")
    
    def _calculate_reward(self, collision_penalty):
        """Calculate reward for the agent."""
        if self.vehicles[0, 5] == 0:  # Agent not active
            return 0
        
        # Progress reward (moving toward goal)
        agent_x = self.vehicles[0, 0]
        progress_reward = agent_x * 0.1  # Reward for moving right
        
        # Speed reward
        speed = self.vehicles[0, 2]
        speed_reward = speed * 0.5
        
        # Collision penalty
        collision_penalty_scaled = collision_penalty * 10
        
        # Completion bonus
        completion_bonus = 50 if self.vehicles[0, 0] > self.grid_size[1] else 0
        
        # Base survival reward
        survival_reward = 1
        
        total_reward = (progress_reward + speed_reward + survival_reward + 
                       completion_bonus - collision_penalty_scaled)
        
        return np.clip(total_reward, -20, 60)  # Clip to reasonable range
    
    def _get_observation(self):
        """Get simplified observation."""
        if self.vehicles[0, 5] == 0:  # Agent not active
            return np.zeros(20, dtype=np.float32)
        
        # Agent state: [x, y, vx, vy]
        agent_state = self.vehicles[0, :4]
        
        # Normalize agent position
        agent_state[0] /= self.grid_size[1]  # x position
        agent_state[1] /= self.grid_size[0]  # y position
        agent_state[2] /= self.max_speed     # vx
        agent_state[3] /= self.max_speed     # vy
        
        # Find 2 nearest vehicles
        agent_x, agent_y = self.vehicles[0, 0], self.vehicles[0, 1]
        distances = []
        
        for i in range(1, self.max_vehicles):
            if self.vehicles[i, 5] == 1:  # Active
                other_x, other_y = self.vehicles[i, 0], self.vehicles[i, 1]
                dist = np.sqrt((agent_x - other_x)**2 + (agent_y - other_y)**2)
                distances.append((dist, i))
        
        distances.sort()
        
        # Get states of 2 nearest vehicles
        nearby_vehicles = []
        for _, i in distances[:2]:
            # Normalize positions relative to agent
            rel_x = (self.vehicles[i, 0] - agent_x) / 5.0
            rel_y = (self.vehicles[i, 1] - agent_y) / 5.0
            rel_vx = self.vehicles[i, 2] / self.max_speed
            rel_vy = self.vehicles[i, 3] / self.max_speed
            nearby_vehicles.extend([rel_x, rel_y, rel_vx, rel_vy])
        
        # Pad if fewer than 2 vehicles
        while len(nearby_vehicles) < 8:
            nearby_vehicles.extend([0, 0, 0, 0])
        
        # Goal information (normalized)
        goal_x = self.grid_size[1] / self.grid_size[1]  # Always 1.0
        goal_y = agent_state[1]  # Same y as agent
        goal_distance = (self.grid_size[1] - self.vehicles[0, 0]) / self.grid_size[1]
        
        # Grid boundaries (normalized)
        boundaries = [
            self.vehicles[0, 0] / self.grid_size[1],  # Distance to right edge
            self.vehicles[0, 1] / self.grid_size[0],  # Distance to bottom edge
            (self.grid_size[1] - self.vehicles[0, 0]) / self.grid_size[1],  # Distance to left
            (self.grid_size[0] - self.vehicles[0, 1]) / self.grid_size[0]   # Distance to top
        ]
        
        # Time information
        time_info = [self.sim_time / 500.0]  # Normalized time
        
        observation = np.concatenate([
            agent_state,           # 4 elements
            nearby_vehicles[:8],   # 8 elements (2 vehicles * 4 each)
            [goal_x, goal_y, goal_distance],  # 3 elements
            boundaries,            # 4 elements
            time_info             # 1 element
        ])  # Total: 20 elements
        
        return observation.astype(np.float32) 