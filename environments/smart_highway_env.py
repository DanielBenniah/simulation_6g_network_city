#!/usr/bin/env python3
"""
Smart Highway Environment with 6G V2V/V2I Communication
======================================================

A hybrid environment that combines:
- 6G V2V/V2I communication and intersection management (from city_traffic_env)
- Directional lanes and realistic vehicle movement (from highway_traffic_env)
- Clear intersection definitions for future traffic light integration

Vehicle Rules:
- Vehicles move only in X-axis OR Y-axis (no diagonal movement)
- Same-axis vehicles use different lanes (no collision possible)
- Cross-axis vehicles communicate at intersections to avoid collisions
- Actions: maintain speed, accelerate, brake (no direction changes)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.intersection_manager import IntersectionManager
from utils.comm_module import CommModule

class SmartHighwayEnv:
    """Smart highway with 6G communication and proper lane management."""
    
    def __init__(self, grid_size=(10, 10), max_vehicles=16, spawn_rate=0.4, debug=False):
        """
        Initialize Smart Highway Environment.
        
        Args:
            grid_size: Size of the highway grid (rows, cols)
            max_vehicles: Maximum number of vehicles
            spawn_rate: Rate of spawning new vehicles
            debug: Enable debug output
        """
        self.grid_size = grid_size
        self.max_vehicles = max_vehicles
        self.spawn_rate = spawn_rate
        self.debug = debug
        
        # Action space: 0=maintain, 1=accelerate, 2=brake
        self.action_space = spaces.Discrete(3)
        
        # Vehicle state: [x, y, vx, vy, direction, lane_offset, active, spawn_time, dest_x, dest_y]
        self.vehicles = np.zeros((max_vehicles, 10), dtype=np.float32)
        
        # Direction codes: 0=L2R, 1=R2L, 2=T2B, 3=B2T
        self.direction_codes = {
            0: "L2R",  # Left to Right
            1: "R2L",  # Right to Left  
            2: "T2B",  # Top to Bottom
            3: "B2T"   # Bottom to Top
        }
        
        # Lane offsets for separation
        self.lane_offset = 0.3
        
        # Speed parameters
        self.min_speed = 0.5
        self.max_speed = 2.0
        self.acceleration = 0.3
        self.brake_factor = 0.5
        
        # 6G Communication system
        self.comm = CommModule()
        self.intersection_manager = IntersectionManager()
        
        # Define intersections clearly for future traffic light integration
        self.intersections = self._define_intersections()
        
        # Simulation state
        self.sim_time = 0
        self.vehicle_spawn_times = {}
        self.journey_times = []
        self.total_spawned = 0
        self.total_completed = 0
        self.collision_count = 0
        
    def _define_intersections(self):
        """Define all intersections in the grid for traffic light integration."""
        intersections = []
        rows, cols = self.grid_size
        
        # Create intersections at every grid point where vehicles can cross
        for row in range(1, rows - 1):  # Skip borders
            for col in range(1, cols - 1):  # Skip borders
                intersection = {
                    'id': f"intersection_{row}_{col}",
                    'position': (row, col),
                    'type': 'main' if (row, col) == (rows//2, cols//2) else 'minor',
                    'lanes': {
                        'horizontal': [(row, col-0.3), (row, col+0.3)],  # L2R, R2L lanes
                        'vertical': [(row-0.3, col), (row+0.3, col)]     # T2B, B2T lanes
                    },
                    'has_traffic_light': False,  # For future integration
                    'light_state': 'green'       # For future integration
                }
                intersections.append(intersection)
        
        if self.debug:
            print(f"[SMART_HIGHWAY] Defined {len(intersections)} intersections")
            
        return intersections
    
    def reset(self, **kwargs):
        """Reset the environment."""
        self.vehicles.fill(0)
        self.sim_time = 0
        self.vehicle_spawn_times = {}
        self.journey_times = []
        self.total_spawned = 0
        self.total_completed = 0
        self.collision_count = 0
        
        # Reset communication systems
        self.comm = CommModule()
        self.intersection_manager = IntersectionManager()
        
        # Spawn initial vehicles
        for _ in range(min(4, self.max_vehicles)):
            self._spawn_vehicle()
            
        return self._get_observation(), {}
    
    def _spawn_vehicle(self):
        """Spawn a new vehicle in a proper lane."""
        # Find empty slot
        for i in range(self.max_vehicles):
            if self.vehicles[i, 6] == 0:  # Not active
                direction = np.random.randint(0, 4)
                
                if direction == 0:  # L2R (X-axis movement only)
                    x, y = 0, np.random.randint(1, self.grid_size[0] - 1)
                    vx, vy = np.random.uniform(self.min_speed, self.max_speed), 0.0  # PURE X movement
                    dest_x, dest_y = self.grid_size[1] - 1, y
                    lane_offset = -self.lane_offset
                    
                elif direction == 1:  # R2L (X-axis movement only)
                    x, y = self.grid_size[1] - 1, np.random.randint(1, self.grid_size[0] - 1)
                    vx, vy = -np.random.uniform(self.min_speed, self.max_speed), 0.0  # PURE X movement
                    dest_x, dest_y = 0, y
                    lane_offset = self.lane_offset
                    
                elif direction == 2:  # T2B (Y-axis movement only)
                    x, y = np.random.randint(1, self.grid_size[1] - 1), 0
                    vx, vy = 0.0, np.random.uniform(self.min_speed, self.max_speed)  # PURE Y movement
                    dest_x, dest_y = x, self.grid_size[0] - 1
                    lane_offset = -self.lane_offset
                    
                else:  # B2T (Y-axis movement only)
                    x, y = np.random.randint(1, self.grid_size[1] - 1), self.grid_size[0] - 1
                    vx, vy = 0.0, -np.random.uniform(self.min_speed, self.max_speed)  # PURE Y movement
                    dest_x, dest_y = x, 0
                    lane_offset = self.lane_offset
                
                # Set vehicle state
                self.vehicles[i] = [x, y, vx, vy, direction, lane_offset, 1, 
                                   self.sim_time, dest_x, dest_y]
                
                self.vehicle_spawn_times[i] = self.sim_time
                self.total_spawned += 1
                
                if self.debug:
                    print(f"[SMART_HIGHWAY] Spawned vehicle {i} ({self.direction_codes[direction]}) at ({x:.1f}, {y:.1f}) -> ({dest_x}, {dest_y})")
                
                return True
        return False
    
    def step(self, action):
        """Execute one simulation step."""
        self.sim_time += 1
        
        # Handle agent action (vehicle 0)
        if self.vehicles[0, 6] == 1:  # If agent vehicle is active
            self._apply_action(0, action)
        
        # Apply random actions to other vehicles for now
        for i in range(1, self.max_vehicles):
            if self.vehicles[i, 6] == 1:
                random_action = np.random.randint(0, 3)
                self._apply_action(i, random_action)
        
        # 6G Communication phase
        step_info = self._handle_6g_communication()
        
        # Move vehicles
        self._move_vehicles()
        
        # Check for arrivals and departures
        self._check_arrivals()
        
        # Spawn new vehicles
        if np.random.random() < self.spawn_rate:
            self._spawn_vehicle()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        terminated = False  # Never terminate in continuous mode
        truncated = False
        
        observation = self._get_observation()
        
        return observation, reward, terminated, truncated, step_info
    
    def _apply_action(self, vehicle_id, action):
        """Apply action to a vehicle (maintain pure X or Y axis movement)."""
        if self.vehicles[vehicle_id, 6] == 0:  # Not active
            return
            
        direction = int(self.vehicles[vehicle_id, 4])
        
        if action == 1:  # Accelerate
            if direction == 0:  # L2R (X-axis only)
                self.vehicles[vehicle_id, 2] = min(self.max_speed, 
                                                 self.vehicles[vehicle_id, 2] + self.acceleration)
                self.vehicles[vehicle_id, 3] = 0.0  # Ensure Y velocity stays 0
                
            elif direction == 1:  # R2L (X-axis only)
                self.vehicles[vehicle_id, 2] = max(-self.max_speed, 
                                                 self.vehicles[vehicle_id, 2] - self.acceleration)
                self.vehicles[vehicle_id, 3] = 0.0  # Ensure Y velocity stays 0
                
            elif direction == 2:  # T2B (Y-axis only)
                self.vehicles[vehicle_id, 3] = min(self.max_speed, 
                                                 self.vehicles[vehicle_id, 3] + self.acceleration)
                self.vehicles[vehicle_id, 2] = 0.0  # Ensure X velocity stays 0
                
            else:  # B2T (Y-axis only)
                self.vehicles[vehicle_id, 3] = max(-self.max_speed, 
                                                 self.vehicles[vehicle_id, 3] - self.acceleration)
                self.vehicles[vehicle_id, 2] = 0.0  # Ensure X velocity stays 0
                    
        elif action == 2:  # Brake
            if direction == 0:  # L2R
                self.vehicles[vehicle_id, 2] *= self.brake_factor
                self.vehicles[vehicle_id, 3] = 0.0  # Ensure Y velocity stays 0
                if self.vehicles[vehicle_id, 2] < self.min_speed:
                    self.vehicles[vehicle_id, 2] = self.min_speed
                    
            elif direction == 1:  # R2L
                self.vehicles[vehicle_id, 2] *= self.brake_factor
                self.vehicles[vehicle_id, 3] = 0.0  # Ensure Y velocity stays 0
                if self.vehicles[vehicle_id, 2] > -self.min_speed:
                    self.vehicles[vehicle_id, 2] = -self.min_speed
                    
            elif direction == 2:  # T2B
                self.vehicles[vehicle_id, 3] *= self.brake_factor
                self.vehicles[vehicle_id, 2] = 0.0  # Ensure X velocity stays 0
                if self.vehicles[vehicle_id, 3] < self.min_speed:
                    self.vehicles[vehicle_id, 3] = self.min_speed
                    
            else:  # B2T
                self.vehicles[vehicle_id, 3] *= self.brake_factor
                self.vehicles[vehicle_id, 2] = 0.0  # Ensure X velocity stays 0
                if self.vehicles[vehicle_id, 3] > -self.min_speed:
                    self.vehicles[vehicle_id, 3] = -self.min_speed
    
    def _handle_6g_communication(self):
        """Handle 6G V2V/V2I communication for intersection management."""
        messages_sent = 0
        messages_delivered = 0
        intersection_reservations = []
        collisions_prevented = []
        
        # Check vehicles approaching intersections
        for i in range(self.max_vehicles):
            if self.vehicles[i, 6] == 0:  # Not active
                continue
                
            x, y = self.vehicles[i, 0], self.vehicles[i, 1]
            vx, vy = self.vehicles[i, 2], self.vehicles[i, 3]
            direction = int(self.vehicles[i, 4])
            
            # Find nearby intersections
            for intersection in self.intersections:
                int_x, int_y = intersection['position']
                distance = np.sqrt((x - int_x)**2 + (y - int_y)**2)
                
                # If approaching intersection (within communication range)
                if distance < 2.0:
                    # Send 6G message for intersection reservation
                    message = {
                        'vehicle_id': i,
                        'position': (x, y),
                        'velocity': (vx, vy),
                        'direction': direction,
                        'intersection': intersection['id'],
                        'arrival_time': self._estimate_arrival_time(i, intersection)
                    }
                    
                    messages_sent += 1
                    
                    # Process message through 6G system
                    # Create path based on vehicle direction
                    direction = int(self.vehicles[i, 4])
                    if direction == 0:  # L2R
                        path = (3, 1)  # West to East
                    elif direction == 1:  # R2L
                        path = (1, 3)  # East to West
                    elif direction == 2:  # T2B
                        path = (0, 2)  # North to South
                    else:  # B2T
                        path = (2, 0)  # South to North
                    
                    duration = 2.0  # Time to cross intersection
                    granted, slot = self.intersection_manager.request_reservation(
                        i, message['arrival_time'], duration, path
                    )
                    
                    if granted:
                        messages_delivered += 1
                        intersection_reservations.append({
                            'vehicle': i,
                            'intersection': intersection['id'],
                            'time': message['arrival_time'],
                            'slot': slot
                        })
                    else:
                        # Collision would have occurred - 6G prevented it
                        collisions_prevented.append({
                            'vehicle': i,
                            'intersection': intersection['id'],
                            'reason': 'intersection_occupied',
                            'suggested_time': slot
                        })
                        
                        # Apply emergency braking
                        self._apply_action(i, 2)  # Brake
        
        return {
            'messages_sent': messages_sent,
            'messages_delivered': messages_delivered,
            'intersection_reservations': intersection_reservations,
            'collisions_prevented': collisions_prevented,
            'collisions': []  # No actual collisions due to 6G prevention
        }
    
    def _estimate_arrival_time(self, vehicle_id, intersection):
        """Estimate when vehicle will arrive at intersection."""
        x, y = self.vehicles[vehicle_id, 0], self.vehicles[vehicle_id, 1]
        vx, vy = self.vehicles[vehicle_id, 2], self.vehicles[vehicle_id, 3]
        int_x, int_y = intersection['position']
        
        distance = np.sqrt((x - int_x)**2 + (y - int_y)**2)
        speed = np.sqrt(vx**2 + vy**2)
        
        if speed < 0.1:
            return self.sim_time + 100  # Far future if stopped
        
        return self.sim_time + (distance / speed)
    
    def _move_vehicles(self):
        """Move all vehicles according to their velocities (PURE X or Y axis movement)."""
        for i in range(self.max_vehicles):
            if self.vehicles[i, 6] == 0:  # Not active
                continue
                
            # Update position - ONLY in the direction the vehicle is traveling
            # NO diagonal movement - vehicles move ONLY in X-axis OR Y-axis
            self.vehicles[i, 0] += self.vehicles[i, 2]  # x += vx
            self.vehicles[i, 1] += self.vehicles[i, 3]  # y += vy
            
            # Lane offset is ONLY for visualization in the visualizer
            # Do NOT modify actual vehicle position here
    
    def _check_arrivals(self):
        """Check if vehicles have reached their destinations."""
        for i in range(self.max_vehicles):
            if self.vehicles[i, 6] == 0:  # Not active
                continue
                
            x, y = self.vehicles[i, 0], self.vehicles[i, 1]
            dest_x, dest_y = self.vehicles[i, 8], self.vehicles[i, 9]
            
            # Check if reached destination or left grid
            reached_dest = (abs(x - dest_x) < 0.5 and abs(y - dest_y) < 0.5)
            left_grid = (x < -0.5 or x >= self.grid_size[1] + 0.5 or 
                        y < -0.5 or y >= self.grid_size[0] + 0.5)
            
            if reached_dest or left_grid:
                # Record journey time
                if i in self.vehicle_spawn_times:
                    journey_time = self.sim_time - self.vehicle_spawn_times[i]
                    self.journey_times.append(journey_time)
                    del self.vehicle_spawn_times[i]
                    self.total_completed += 1
                
                # Deactivate vehicle
                self.vehicles[i, 6] = 0
                
                if self.debug:
                    status = "reached destination" if reached_dest else "left grid"
                    print(f"[SMART_HIGHWAY] Vehicle {i} {status}")
    
    def _calculate_reward(self):
        """Calculate reward for the agent."""
        # Reward for agent vehicle making progress
        if self.vehicles[0, 6] == 1:  # Agent active
            x, y = self.vehicles[0, 0], self.vehicles[0, 1]
            dest_x, dest_y = self.vehicles[0, 8], self.vehicles[0, 9]
            distance_to_dest = np.sqrt((x - dest_x)**2 + (y - dest_y)**2)
            
            # Higher reward for being closer to destination
            progress_reward = max(0, 10 - distance_to_dest)
            
            # Bonus for maintaining good speed
            speed = np.sqrt(self.vehicles[0, 2]**2 + self.vehicles[0, 3]**2)
            speed_reward = speed * 0.5
            
            return progress_reward + speed_reward
        
        return 0  # No reward if agent not active
    
    def _get_observation(self):
        """Get observation for the agent."""
        if self.vehicles[0, 6] == 0:  # Agent not active
            return np.zeros(20, dtype=np.float32)
        
        # Agent vehicle state
        agent_state = self.vehicles[0, :6]  # x, y, vx, vy, direction, lane_offset
        
        # Nearest vehicles (up to 3)
        other_vehicles = []
        agent_x, agent_y = self.vehicles[0, 0], self.vehicles[0, 1]
        
        distances = []
        for i in range(1, self.max_vehicles):
            if self.vehicles[i, 6] == 1:  # Active
                other_x, other_y = self.vehicles[i, 0], self.vehicles[i, 1]
                dist = np.sqrt((agent_x - other_x)**2 + (agent_y - other_y)**2)
                distances.append((dist, i))
        
        distances.sort()
        for _, i in distances[:3]:  # Take 3 nearest
            other_vehicles.extend(self.vehicles[i, :4])  # x, y, vx, vy
        
        # Pad if fewer than 3 vehicles
        while len(other_vehicles) < 12:
            other_vehicles.extend([0, 0, 0, 0])
        
        # Distance to destination
        dest_x, dest_y = self.vehicles[0, 8], self.vehicles[0, 9]
        dest_distance = [dest_x - agent_x, dest_y - agent_y]
        
        observation = np.concatenate([agent_state, other_vehicles[:12], dest_distance])
        return observation.astype(np.float32)
    
    def get_statistics(self):
        """Get journey and traffic statistics."""
        active_vehicles = int(np.sum(self.vehicles[:, 6]))
        
        stats = {
            'active_vehicles': active_vehicles,
            'total_spawned': self.total_spawned,
            'total_completed': self.total_completed,
            'collision_count': self.collision_count,
            'intersections': len(self.intersections)
        }
        
        if self.journey_times:
            stats.update({
                'avg_journey_time': np.mean(self.journey_times),
                'min_journey_time': np.min(self.journey_times),
                'max_journey_time': np.max(self.journey_times),
                'efficiency': (self.total_completed / max(self.total_spawned, 1)) * 100
            })
        else:
            stats.update({
                'avg_journey_time': 0,
                'min_journey_time': 0,
                'max_journey_time': 0,
                'efficiency': 0
            })
        
        return stats
    
    def get_intersection_info(self):
        """Get information about all intersections (for traffic light integration)."""
        return {
            'intersections': self.intersections,
            'count': len(self.intersections),
            'main_intersection': next((i for i in self.intersections if i['type'] == 'main'), None)
        } 