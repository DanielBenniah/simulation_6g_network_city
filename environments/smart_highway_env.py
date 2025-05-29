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
    
    def __init__(self, grid_size=(10, 10), max_vehicles=24, spawn_rate=0.6, 
                 multi_agent=False, debug=False, enable_6g=True, enable_traffic_lights=False):
        """
        Initialize Smart Highway Environment with optional multi-agent support.
        
        Args:
            grid_size: Size of the highway grid (rows, cols)
            max_vehicles: Maximum number of vehicles in simulation
            spawn_rate: Probability of spawning new vehicles each step
            multi_agent: If True, all vehicles are learning agents; if False, only vehicle 0 learns
            debug: Enable debug output
            enable_6g: If True, enable 6G communication; if False, disable for comparison
            enable_traffic_lights: If True, enable traffic light control and queuing
        """
        self.grid_size = grid_size
        self.max_vehicles = max_vehicles
        self.spawn_rate = spawn_rate
        self.multi_agent = multi_agent  # NEW: Multi-agent support
        self.debug = debug
        self.enable_6g = enable_6g  # NEW: Enable/disable 6G communication
        self.enable_traffic_lights = enable_traffic_lights  # NEW: Enable/disable traffic lights
        
        # Agent IDs for multi-agent mode
        self.agent_ids = [f"agent_{i}" for i in range(max_vehicles)] if multi_agent else None
        
        # Action and observation spaces
        self.single_action_space = spaces.Discrete(3)  # 0=maintain, 1=accelerate, 2=brake
        self.single_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)
        
        if multi_agent:
            self.action_space = spaces.Dict({
                f"agent_{i}": self.single_action_space for i in range(max_vehicles)
            })
            self.observation_space = spaces.Dict({
                f"agent_{i}": self.single_obs_space for i in range(max_vehicles)
            })
        else:
            self.action_space = self.single_action_space
            self.observation_space = self.single_obs_space
        
        # Vehicle state: [x, y, vx, vy, direction, lane_offset, active, spawn_time, dest_x, dest_y]
        # direction: 0=L2R (West->East), 1=T2B (North->South)
        self.vehicles = np.zeros((max_vehicles, 10), dtype=np.float32)
        
        # Highway parameters
        self.max_speed = 2.0
        self.min_speed = 0.5
        self.acceleration = 0.3
        self.brake_factor = 0.5
        
        # 6G Communication system (only if enabled)
        if self.enable_6g:
            self.comm = CommModule()
            self.intersection_manager = IntersectionManager()
        else:
            self.comm = None
            self.intersection_manager = None
        
        # Define intersections clearly for future traffic light integration
        self.intersections = self._define_intersections()
        
        # Traffic light timing parameters
        if self.enable_traffic_lights:
            self.light_cycle_duration = 20.0  # Total cycle time in simulation steps
            self.yellow_duration = 3.0  # Yellow light duration
            self.green_duration = (self.light_cycle_duration - 2 * self.yellow_duration) / 2  # Green for each direction
        
        # Simulation state
        self.sim_time = 0
        self.vehicle_spawn_times = {}
        self.journey_times = []
        self.total_spawned = 0
        self.total_completed = 0
        self.collision_count = 0
        self.queued_vehicles = []  # Track vehicles waiting at red lights
        self.total_wait_time = 0  # Track total waiting time for traffic lights
        
    def _define_intersections(self):
        """Define major intersections clearly for traffic light integration."""
        intersections = []
        rows, cols = self.grid_size
        
        # Create major intersections at lane crossing points
        lane_positions = [2, 4, 6, 8]  # Match vehicle lane positions
        for row in lane_positions:
            for col in lane_positions:
                intersection_type = 'main' if (row, col) == (rows//2, cols//2) else 'major'
                
                intersection = {
                    'id': f"intersection_{row}_{col}",
                    'position': (row, col),
                    'type': intersection_type,
                    'size': 0.8 if intersection_type == 'main' else 0.6,  # Visual size
                    'lanes': {
                        'horizontal': {
                            'approach_lanes': [
                                (row - 0.4, col), (row - 0.2, col), (row, col), (row + 0.2, col), (row + 0.4, col)
                            ],
                            'direction': 'L2R',
                            'speed_limit': self.max_speed
                        },
                        'vertical': {
                            'approach_lanes': [
                                (row, col - 0.4), (row, col - 0.2), (row, col), (row, col + 0.2), (row, col + 0.4)
                            ],
                            'direction': 'T2B', 
                            'speed_limit': self.max_speed
                        }
                    },
                    'traffic_light': {
                        'has_light': True,  # All major intersections have lights
                        'current_state': 'green_horizontal',  # green_horizontal, green_vertical, yellow, red
                        'cycle_time': 30.0,  # seconds per cycle
                        'last_change': 0.0,
                        'phases': ['green_horizontal', 'yellow_horizontal', 'green_vertical', 'yellow_vertical']
                    },
                    '6g_infrastructure': {
                        'tower_position': (row, col),
                        'communication_range': 3.0,
                        'message_capacity': 100,
                        'reliability': 0.99
                    },
                    'safety_zone': {
                        'radius': 1.5,
                        'emergency_brake_distance': 2.0,
                        'collision_detection_active': True
                    }
                }
                intersections.append(intersection)
        
        if self.debug:
            print(f"[SMART_HIGHWAY] Defined {len(intersections)} major intersections")
            for intersection in intersections:
                print(f"  - {intersection['id']} at {intersection['position']} ({intersection['type']})")
            
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
        self.queued_vehicles = []  # Track vehicles waiting at red lights
        self.total_wait_time = 0  # Track total waiting time for traffic lights
        
        # Reset communication systems
        if self.enable_6g:
            self.comm = CommModule()
            self.intersection_manager = IntersectionManager()
        else:
            self.comm = None
            self.intersection_manager = None
        
        # Spawn initial vehicles (more for higher density)
        for _ in range(min(8, self.max_vehicles)):
            self._spawn_vehicle()
            
        if self.multi_agent:
            # Multi-agent mode: return dict of observations for active vehicles
            observations = {}
            for i in range(self.max_vehicles):
                if self.vehicles[i, 6] == 1:  # Active vehicle
                    agent_id = f"agent_{i}"
                    observations[agent_id] = self._get_observation(i)
            return observations, {}
        else:
            # Single-agent mode: return observation for vehicle 0
            return self._get_observation(0), {}
    
    def _spawn_vehicle(self):
        """Spawn vehicles in a way that creates intersection conflicts for better 6G testing."""
        # Find empty slot
        for i in range(self.max_vehicles):
            if self.vehicles[i, 6] == 0:  # Not active
                # Bias spawning towards intersection areas to create conflicts
                intersection_bias = np.random.random() < 0.7  # 70% chance to spawn near intersections
                
                if intersection_bias:
                    # Spawn vehicles that will cross intersections
                    direction = np.random.randint(0, 2)
                    
                    if direction == 0:  # L2R (Horizontal)
                        # Spawn at intersection Y positions to create conflicts
                        lane_y_positions = [2, 4, 6, 8]  # Intersection positions
                        y = np.random.choice(lane_y_positions)
                        x = np.random.uniform(0, 2)  # Spawn closer to intersection
                        vx, vy = np.random.uniform(1.0, self.max_speed), 0.0
                        dest_x, dest_y = self.grid_size[1] - 1, y
                        
                    else:  # T2B (Vertical)
                        # Spawn at intersection X positions to create conflicts
                        lane_x_positions = [2, 4, 6, 8]  # Intersection positions
                        x = np.random.choice(lane_x_positions)
                        y = np.random.uniform(0, 2)  # Spawn closer to intersection
                        vx, vy = 0.0, np.random.uniform(1.0, self.max_speed)
                        dest_x, dest_y = x, self.grid_size[0] - 1
                else:
                    # Regular spawning (for diversity)
                    direction = np.random.randint(0, 2)
                    
                    if direction == 0:  # L2R
                        lane_y_positions = [2, 4, 6, 8]
                        y = np.random.choice(lane_y_positions)
                        x = 0
                        vx, vy = np.random.uniform(self.min_speed, self.max_speed), 0.0
                        dest_x, dest_y = self.grid_size[1] - 1, y
                        
                    else:  # T2B
                        lane_x_positions = [2, 4, 6, 8]
                        x = np.random.choice(lane_x_positions)
                        y = 0
                        vx, vy = 0.0, np.random.uniform(self.min_speed, self.max_speed)
                        dest_x, dest_y = x, self.grid_size[0] - 1
                
                # Lane positioning for visualization
                lane_positions = [-0.4, -0.2, 0.0, 0.2, 0.4]
                lane_offset = np.random.choice(lane_positions)
                
                # Set vehicle state
                self.vehicles[i] = [x, y, vx, vy, direction, lane_offset, 1, 
                                   self.sim_time, dest_x, dest_y]
                
                self.vehicle_spawn_times[i] = self.sim_time
                self.total_spawned += 1
                
                if self.debug:
                    print(f"[SMART_HIGHWAY] Spawned vehicle {i} (dir={direction}) at ({x:.1f}, {y:.1f}) -> ({dest_x}, {dest_y})")
                
                return True
        return False
    
    def step(self, actions):
        """Execute one simulation step with multi-agent support."""
        self.sim_time += 1
        
        # Handle actions for all vehicles
        if self.multi_agent:
            # Multi-agent mode: all active vehicles are learning agents
            for i in range(self.max_vehicles):
                if self.vehicles[i, 6] == 1:  # If vehicle is active
                    agent_id = f"agent_{i}"
                    agent_action = actions.get(agent_id, 0)  # Default to maintain if no action
                    self._apply_action(i, agent_action)
        else:
            # Single-agent mode: only vehicle 0 is learning
            if self.vehicles[0, 6] == 1:  # If agent vehicle is active
                self._apply_action(0, actions)
            
            # Apply random actions to other vehicles
            for i in range(1, self.max_vehicles):
                if self.vehicles[i, 6] == 1:
                    random_action = np.random.randint(0, 3)
                    self._apply_action(i, random_action)
        
        # 6G Communication phase
        step_info = self._handle_6g_communication()
        
        # Update traffic lights if enabled
        if self.enable_traffic_lights:
            self._update_traffic_lights()
            step_info['traffic_lights'] = self.get_traffic_light_states()
        
        # Move vehicles (respecting traffic lights if enabled)
        self._move_vehicles()
        
        # Check for arrivals and departures
        self._check_arrivals()
        
        # Spawn new vehicles
        if np.random.random() < self.spawn_rate:
            self._spawn_vehicle()
        
        # Add traffic light statistics to step_info
        if self.enable_traffic_lights:
            step_info['queued_vehicles'] = len(self.queued_vehicles)
            step_info['total_wait_time'] = self.total_wait_time
        
        # Calculate rewards for all vehicles
        if self.multi_agent:
            rewards = {}
            terminated = {}
            truncated = {}
            observations = {}
            
            # Calculate individual rewards and termination for each active agent
            for i in range(self.max_vehicles):
                agent_id = f"agent_{i}"
                if self.vehicles[i, 6] == 1:  # Active vehicle
                    # Calculate reward for this agent
                    agent_reward = self._calculate_reward(i)
                    
                    # Add collision penalty based on step_info
                    collision_penalty = 0
                    for collision in step_info.get('actual_collisions', []):
                        if i in collision['vehicles']:
                            collision_penalty += 5.0
                    agent_reward -= collision_penalty
                    
                    rewards[agent_id] = agent_reward
                    
                    # Check termination for this agent
                    agent_x, agent_y = self.vehicles[i, 0], self.vehicles[i, 1]
                    dest_x, dest_y = self.vehicles[i, 8], self.vehicles[i, 9]
                    distance_to_dest = np.sqrt((agent_x - dest_x)**2 + (agent_y - dest_y)**2)
                    
                    terminated[agent_id] = (distance_to_dest < 0.5 or  # Reached destination
                                          agent_x < -1 or agent_x > self.grid_size[1] + 1 or  # Left grid
                                          agent_y < -1 or agent_y > self.grid_size[0] + 1)
                    
                    truncated[agent_id] = self.sim_time > 200  # Max episode length
                    
                    # Get observation for this agent
                    observations[agent_id] = self._get_observation(i)
            
            return observations, rewards, terminated, truncated, step_info
            
        else:
            # Single-agent mode (original behavior)
            reward = self._calculate_reward(0)
            
            # Add collision penalty to reward based on step_info
            if step_info.get('actual_collisions'):
                collision_penalty = 0
                for collision in step_info['actual_collisions']:
                    if 0 in collision['vehicles']:  # Agent involved in collision
                        collision_penalty += 5.0
                reward -= collision_penalty
            
            # Check termination conditions for agent
            if self.vehicles[0, 6] == 1:  # Agent active
                agent_x, agent_y = self.vehicles[0, 0], self.vehicles[0, 1]
                dest_x, dest_y = self.vehicles[0, 8], self.vehicles[0, 9]
                distance_to_dest = np.sqrt((agent_x - dest_x)**2 + (agent_y - dest_y)**2)
                
                terminated = (distance_to_dest < 0.5 or  # Reached destination
                             agent_x < -1 or agent_x > self.grid_size[1] + 1 or  # Left grid
                             agent_y < -1 or agent_y > self.grid_size[0] + 1)
            else:
                terminated = True  # Agent not active
            
            truncated = self.sim_time > 200  # Max episode length
            
            observation = self._get_observation(0)
            
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
                
            else:  # T2B (Y-axis only)
                self.vehicles[vehicle_id, 3] = min(self.max_speed, 
                                                 self.vehicles[vehicle_id, 3] + self.acceleration)
                self.vehicles[vehicle_id, 2] = 0.0  # Ensure X velocity stays 0
                    
        elif action == 2:  # Brake
            if direction == 0:  # L2R
                self.vehicles[vehicle_id, 2] *= self.brake_factor
                self.vehicles[vehicle_id, 3] = 0.0  # Ensure Y velocity stays 0
                if self.vehicles[vehicle_id, 2] < self.min_speed:
                    self.vehicles[vehicle_id, 2] = self.min_speed
                    
            else:  # T2B
                self.vehicles[vehicle_id, 3] *= self.brake_factor
                self.vehicles[vehicle_id, 2] = 0.0  # Ensure X velocity stays 0
                if self.vehicles[vehicle_id, 3] < self.min_speed:
                    self.vehicles[vehicle_id, 3] = self.min_speed
    
    def _handle_6g_communication(self):
        """Handle 6G V2V/V2I communication for intersection management."""
        if not self.enable_6g:
            # When 6G is disabled, only do basic collision detection without prevention
            actual_collisions = []
            for i in range(self.max_vehicles):
                if self.vehicles[i, 6] == 0:  # Not active
                    continue
                for j in range(i + 1, self.max_vehicles):
                    if self.vehicles[j, 6] == 0:  # Not active
                        continue
                    
                    # Check physical proximity (actual collision)
                    x1, y1 = self.vehicles[i, 0], self.vehicles[i, 1]
                    x2, y2 = self.vehicles[j, 0], self.vehicles[j, 1]
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    
                    if distance < 0.3:  # Very close - actual collision
                        actual_collisions.append({
                            'vehicles': [i, j],
                            'position': ((x1 + x2) / 2, (y1 + y2) / 2),
                            'type': 'collision_no_6g',
                            'distance': distance
                        })
                        self.collision_count += 1
                        
                        # Emergency stop for both vehicles
                        self.vehicles[i, 2] *= 0.1  # Emergency brake
                        self.vehicles[i, 3] *= 0.1
                        self.vehicles[j, 2] *= 0.1
                        self.vehicles[j, 3] *= 0.1
                        
                        if self.debug:
                            print(f"[COLLISION_NO_6G] Collision between vehicles {i} and {j} at distance {distance:.3f}")
            
            # Return minimal info when 6G is disabled
            return {
                'messages_sent': 0,
                'messages_delivered': 0,
                'intersection_reservations': [],
                'collisions_prevented': [],
                'actual_collisions': actual_collisions,
                'collision_prevention_rate': 0.0,
                '6g_status': 'disabled'
            }
        
        # Original 6G communication code when enabled
        messages_sent = 0
        messages_delivered = 0
        intersection_reservations = []
        collisions_prevented = []
        actual_collisions = []
        
        # First, check for actual physical collisions (vehicles occupying same space)
        for i in range(self.max_vehicles):
            if self.vehicles[i, 6] == 0:  # Not active
                continue
            for j in range(i + 1, self.max_vehicles):
                if self.vehicles[j, 6] == 0:  # Not active
                    continue
                
                # Check physical proximity (actual collision)
                x1, y1 = self.vehicles[i, 0], self.vehicles[i, 1]
                x2, y2 = self.vehicles[j, 0], self.vehicles[j, 1]
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                
                if distance < 0.5:  # Increased collision detection range
                    # More detailed collision detection
                    dir1, dir2 = int(self.vehicles[i, 4]), int(self.vehicles[j, 4])
                    
                    # Check for actual collisions
                    if distance < 0.3:  # Very close - actual collision
                        actual_collisions.append({
                            'vehicles': [i, j],
                            'position': ((x1 + x2) / 2, (y1 + y2) / 2),
                            'type': 'intersection_collision' if dir1 != dir2 else 'same_lane_collision',
                            'distance': distance
                        })
                        self.collision_count += 1
                        
                        # Emergency stop for both vehicles
                        self.vehicles[i, 2] *= 0.1  # Emergency brake
                        self.vehicles[i, 3] *= 0.1
                        self.vehicles[j, 2] *= 0.1
                        self.vehicles[j, 3] *= 0.1
                        
                        if self.debug:
                            print(f"[COLLISION] Actual collision between vehicles {i} and {j} at distance {distance:.3f}")
                    
                    # Near-miss detection (close but not colliding)
                    elif distance < 0.4 and dir1 != dir2:  # Near miss at intersection
                        # This should have been prevented by 6G
                        if not any(collision['vehicles'] == [i, j] or collision['vehicles'] == [j, i] 
                                 for collision in collisions_prevented):
                            # 6G system failed to prevent this near miss
                            actual_collisions.append({
                                'vehicles': [i, j],
                                'position': ((x1 + x2) / 2, (y1 + y2) / 2),
                                'type': 'near_miss_intersection',
                                'distance': distance
                            })
                            
                            if self.debug:
                                print(f"[NEAR_MISS] 6G failed to prevent near collision {i}-{j} at distance {distance:.3f}")
        
        # Then, handle 6G communication to prevent future collisions
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
                if distance < 2.0 and distance > 0.5:  # Not too close, not too far
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
                    
                    # Enhanced 6G intersection management
                    direction = int(self.vehicles[i, 4])
                    if direction == 0:  # L2R
                        path = (3, 1)  # West to East
                    else:  # T2B
                        path = (0, 2)  # North to South
                    
                    duration = 3.0  # Longer crossing time to create more conflicts
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
                        # 6G prevented a potential collision - be more aggressive in prevention
                        collisions_prevented.append({
                            'vehicle': i,
                            'intersection': intersection['id'],
                            'reason': 'intersection_occupied',
                            'suggested_time': slot,
                            'prevention_type': '6G_V2I_reservation'
                        })
                        
                        # More noticeable collision prevention action
                        current_speed = np.sqrt(vx**2 + vy**2)
                        if current_speed > self.min_speed:
                            # Apply multiple brake actions for stronger effect
                            self._apply_action(i, 2)  # Brake
                            if current_speed > 1.5:
                                self._apply_action(i, 2)  # Double brake for high speeds
                    
                    # Additional V2V collision prediction
                    for j in range(self.max_vehicles):
                        if j != i and self.vehicles[j, 6] == 1:  # Different active vehicle
                            x2, y2 = self.vehicles[j, 0], self.vehicles[j, 1]
                            vx2, vy2 = self.vehicles[j, 2], self.vehicles[j, 3]
                            
                            # Predict future collision at intersection
                            future_time = 3.0  # Look ahead 3 seconds
                            future_x1 = x + vx * future_time
                            future_y1 = y + vy * future_time
                            future_x2 = x2 + vx2 * future_time
                            future_y2 = y2 + vy2 * future_time
                            
                            # Check if both will be at same intersection
                            predicted_distance = np.sqrt((future_x1 - future_x2)**2 + (future_y1 - future_y2)**2)
                            if predicted_distance < 1.5:  # Collision predicted
                                collisions_prevented.append({
                                    'vehicle': i,
                                    'other_vehicle': j,
                                    'intersection': intersection['id'],
                                    'reason': 'v2v_collision_prediction',
                                    'prevention_type': '6G_V2V_coordination'
                                })
                                
                                # Coordinate speeds to avoid collision
                                vehicle_speed = np.sqrt(vx**2 + vy**2)
                                other_speed = np.sqrt(vx2**2 + vy2**2)
                                if vehicle_speed > other_speed:
                                    self._apply_action(i, 2)  # Faster vehicle brakes
                                else:
                                    self._apply_action(j, 2)  # Other vehicle brakes
        
        return {
            'messages_sent': messages_sent,
            'messages_delivered': messages_delivered,
            'intersection_reservations': intersection_reservations,
            'collisions_prevented': collisions_prevented,
            'actual_collisions': actual_collisions,  # NEW: Track real collisions
            'collision_prevention_rate': len(collisions_prevented) / max(messages_sent, 1) * 100,
            '6g_status': 'enabled'
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
                
            # Check traffic light restrictions if enabled
            if self.enable_traffic_lights:
                if not self._can_proceed_through_intersection(i):
                    # Vehicle must stop - add to queue and brake
                    if i not in self.queued_vehicles:
                        self.queued_vehicles.append(i)
                    self._apply_action(i, 2)  # Brake
                    self.total_wait_time += 1
                    continue
                else:
                    # Vehicle can proceed - remove from queue if it was there
                    if i in self.queued_vehicles:
                        self.queued_vehicles.remove(i)
            
            # Update position - ONLY in the direction the vehicle is traveling
            # NO diagonal movement - vehicles move ONLY in X-axis OR Y-axis
            self.vehicles[i, 0] += self.vehicles[i, 2]  # x += vx
            self.vehicles[i, 1] += self.vehicles[i, 3]  # y += vy
            
            # Lane offset is ONLY for visualization in the visualizer
            # Do NOT modify actual vehicle position here
    
    def _update_traffic_lights(self):
        """Update traffic light states based on timing cycle."""
        for intersection in self.intersections:
            # Calculate current phase based on simulation time
            cycle_time = self.sim_time % self.light_cycle_duration
            
            if cycle_time < self.green_duration:
                # Green for horizontal traffic (L2R)
                intersection['traffic_light']['current_state'] = 'green_horizontal'
            elif cycle_time < self.green_duration + self.yellow_duration:
                # Yellow for horizontal traffic
                intersection['traffic_light']['current_state'] = 'yellow_horizontal'
            elif cycle_time < 2 * self.green_duration + self.yellow_duration:
                # Green for vertical traffic (T2B)
                intersection['traffic_light']['current_state'] = 'green_vertical'
            else:
                # Yellow for vertical traffic
                intersection['traffic_light']['current_state'] = 'yellow_vertical'
    
    def _can_proceed_through_intersection(self, vehicle_id):
        """Check if a vehicle can proceed through nearby intersections based on traffic lights."""
        x, y = self.vehicles[vehicle_id, 0], self.vehicles[vehicle_id, 1]
        direction = int(self.vehicles[vehicle_id, 4])
        
        # Check all intersections
        for intersection in self.intersections:
            int_x, int_y = intersection['position']
            distance = np.sqrt((x - int_x)**2 + (y - int_y)**2)
            
            # If vehicle is approaching intersection (within 1.5 units)
            if distance < 1.5:
                light_state = intersection['traffic_light']['current_state']
                
                # Check if vehicle's direction has green light
                if direction == 0:  # L2R (Horizontal)
                    if light_state in ['green_horizontal']:
                        return True
                    elif light_state in ['yellow_horizontal']:
                        # Allow if very close to intersection (already committed)
                        return distance < 0.8
                    else:  # Red or green for opposite direction
                        return False
                        
                else:  # T2B (Vertical)
                    if light_state in ['green_vertical']:
                        return True
                    elif light_state in ['yellow_vertical']:
                        # Allow if very close to intersection (already committed)
                        return distance < 0.8
                    else:  # Red or green for opposite direction
                        return False
        
        # No nearby intersection restrictions
        return True
    
    def get_traffic_light_states(self):
        """Get current traffic light states for all intersections."""
        light_states = {}
        for intersection in self.intersections:
            light_states[intersection['id']] = {
                'state': intersection['traffic_light']['current_state'],
                'position': intersection['position'],
                'cycle_time': self.sim_time % self.light_cycle_duration
            }
        return light_states
    
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
    
    def _calculate_reward(self, vehicle_id=0):
        """Calculate reward for a specific vehicle with proper scaling for RL training."""
        if self.vehicles[vehicle_id, 6] == 0:  # Vehicle not active
            return 0
        
        # Vehicle state
        x, y = self.vehicles[vehicle_id, 0], self.vehicles[vehicle_id, 1]
        vx, vy = self.vehicles[vehicle_id, 2], self.vehicles[vehicle_id, 3]
        dest_x, dest_y = self.vehicles[vehicle_id, 8], self.vehicles[vehicle_id, 9]
        
        # 1. Progress reward (small incremental reward)
        distance_to_dest = np.sqrt((x - dest_x)**2 + (y - dest_y)**2)
        max_distance = np.sqrt(self.grid_size[0]**2 + self.grid_size[1]**2)
        progress_ratio = 1 - (distance_to_dest / max_distance)
        progress_reward = progress_ratio * 5.0  # 0-5 points for progress
        
        # 2. Speed maintenance reward (encourage consistent movement)
        speed = np.sqrt(vx**2 + vy**2)
        if speed > self.min_speed:
            speed_reward = 1.0  # Reward for moving
        else:
            speed_reward = -0.5  # Penalty for being too slow
        
        # 3. Step penalty (encourage efficiency)
        step_penalty = -0.05  # Small penalty per step to encourage faster completion
        
        # 4. Destination reached bonus (only when very close)
        if distance_to_dest < 0.5:
            goal_bonus = 20.0  # Big bonus for actually reaching destination
        else:
            goal_bonus = 0
        
        # 5. Collision penalty (from step info)
        collision_penalty = 0  # Will be added in step function
        
        # Total reward (more conservative scaling)
        total_reward = progress_reward + speed_reward + step_penalty + goal_bonus - collision_penalty
        
        return total_reward
    
    def _get_observation(self, vehicle_id=0):
        """Get observation for a specific vehicle."""
        if self.vehicles[vehicle_id, 6] == 0:  # Vehicle not active
            return np.zeros(20, dtype=np.float32)
        
        # Vehicle state
        vehicle_state = self.vehicles[vehicle_id, :6]  # x, y, vx, vy, direction, lane_offset
        
        # Nearest vehicles (up to 3)
        other_vehicles = []
        vehicle_x, vehicle_y = self.vehicles[vehicle_id, 0], self.vehicles[vehicle_id, 1]
        
        distances = []
        for i in range(self.max_vehicles):
            if i != vehicle_id and self.vehicles[i, 6] == 1:  # Active and not self
                other_x, other_y = self.vehicles[i, 0], self.vehicles[i, 1]
                dist = np.sqrt((vehicle_x - other_x)**2 + (vehicle_y - other_y)**2)
                distances.append((dist, i))
        
        distances.sort()
        for _, i in distances[:3]:  # Take 3 nearest
            other_vehicles.extend(self.vehicles[i, :4])  # x, y, vx, vy
        
        # Pad if fewer than 3 vehicles
        while len(other_vehicles) < 12:
            other_vehicles.extend([0, 0, 0, 0])
        
        # Distance to destination
        dest_x, dest_y = self.vehicles[vehicle_id, 8], self.vehicles[vehicle_id, 9]
        dest_distance = [dest_x - vehicle_x, dest_y - vehicle_y]
        
        observation = np.concatenate([vehicle_state, other_vehicles[:12], dest_distance])
        return observation.astype(np.float32)
    
    def get_statistics(self):
        """Get comprehensive simulation statistics including traffic light data."""
        active_vehicles = int(np.sum(self.vehicles[:, 6]))
        avg_journey_time = np.mean(self.journey_times) if self.journey_times else 0
        
        stats = {
            'simulation_time': self.sim_time,
            'total_spawned': self.total_spawned,
            'total_completed': self.total_completed,
            'active_vehicles': active_vehicles,
            'completion_rate': (self.total_completed / max(self.total_spawned, 1)) * 100,
            'collision_count': self.collision_count,
            'collision_rate': self.collision_count / max(self.sim_time, 1) * 100,
            'avg_journey_time': avg_journey_time,
            'journey_times': self.journey_times.copy(),
            'traffic_light_enabled': self.enable_traffic_lights,
            '6g_enabled': self.enable_6g
        }
        
        # Add traffic light specific statistics
        if self.enable_traffic_lights:
            stats.update({
                'queued_vehicles': len(self.queued_vehicles),
                'total_wait_time': self.total_wait_time,
                'avg_wait_time_per_vehicle': self.total_wait_time / max(self.total_spawned, 1),
                'current_light_states': self.get_traffic_light_states(),
                'traffic_efficiency': max(0, 100 - (self.total_wait_time / max(self.sim_time * active_vehicles, 1) * 100))
            })
        else:
            stats.update({
                'queued_vehicles': 0,
                'total_wait_time': 0,
                'avg_wait_time_per_vehicle': 0,
                'current_light_states': {},
                'traffic_efficiency': 100  # No waiting without traffic lights
            })
            
        return stats
    
    def get_intersection_info(self):
        """Get information about all intersections (for traffic light integration)."""
        return {
            'intersections': self.intersections,
            'count': len(self.intersections),
            'main_intersection': next((i for i in self.intersections if i['type'] == 'main'), None)
        } 