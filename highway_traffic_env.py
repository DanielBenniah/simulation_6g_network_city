"""
Highway-style traffic environment with spawn zones and dedicated lanes.
Vehicles spawn from edges, move in dedicated lanes, and can only accelerate/brake.
Only perpendicular traffic can collide (at intersections).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class HighwayTrafficEnv(gym.Env):
    """
    Highway-style traffic simulation with:
    - Spawn zones at grid edges that continuously create vehicles
    - Dedicated lanes (no head-on collisions)
    - Vehicles can only accelerate/brake in their original direction
    - Collisions only occur between perpendicular traffic at intersections
    - Realistic continuous traffic flow
    """
    
    def __init__(self, grid_size=(12, 12), max_vehicles=16, spawn_rate=0.15, debug=False):
        super(HighwayTrafficEnv, self).__init__()
        self.grid_size = grid_size
        self.max_vehicles = max_vehicles
        self.spawn_rate = spawn_rate  # Probability of spawning new vehicle each step
        self.debug = debug
        
        # Highway parameters
        self.max_speed = 2.0
        self.min_speed = 0.2
        self.accel = 0.4
        self.brake_factor = 0.7
        self.safe_distance = 1.5
        
        # Action space: 0=maintain, 1=accelerate, 2=brake
        self.n_actions = 3
        self.action_space = spaces.Discrete(self.n_actions)
        self.obs_dim = 8  # [x, y, vx, vy, direction, speed, front_distance, side_traffic]
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), dtype=np.float32)
        
        # Vehicle tracking
        self.vehicles = None
        self.vehicle_count = 0
        self.sim_time = 0.0
        
        # Spawn zones (edges of grid)
        self.spawn_zones = {
            'left_to_right': {'start': (0, 'random'), 'direction': (1, 0)},
            'right_to_left': {'start': (grid_size[0]-1, 'random'), 'direction': (-1, 0)},
            'top_to_bottom': {'start': ('random', 0), 'direction': (0, 1)},
            'bottom_to_top': {'start': ('random', grid_size[1]-1), 'direction': (0, -1)}
        }

    def reset(self, seed=None, **kwargs):
        """Reset with initial vehicles from spawn zones."""
        if seed is not None:
            np.random.seed(seed)
        
        # Vehicle array: [x, y, vx, vy, direction_code, active, id, lane_offset]
        # direction_code: 0=L2R, 1=R2L, 2=T2B, 3=B2T
        self.vehicles = np.zeros((self.max_vehicles, 8), dtype=np.float32)
        self.vehicle_count = 0
        self.sim_time = 0.0
        
        # Spawn initial vehicles
        initial_vehicles = min(8, self.max_vehicles // 2)
        for _ in range(initial_vehicles):
            self._spawn_vehicle()
        
        obs = self._get_observation()
        info = {"vehicles_spawned": initial_vehicles, "collisions": 0, "vehicles_exited": 0}
        return obs, info

    def _spawn_vehicle(self):
        """Spawn a new vehicle from a random spawn zone."""
        if self.vehicle_count >= self.max_vehicles:
            return False
        
        # Find empty slot
        vehicle_idx = None
        for i in range(self.max_vehicles):
            if self.vehicles[i, 5] == 0:  # Not active
                vehicle_idx = i
                break
        
        if vehicle_idx is None:
            return False
        
        # Choose random spawn zone
        spawn_type = np.random.choice(['left_to_right', 'right_to_left', 'top_to_bottom', 'bottom_to_top'])
        spawn_info = self.spawn_zones[spawn_type]
        
        # Set position
        start_pos = spawn_info['start']
        if start_pos[0] == 'random':
            x = np.random.uniform(1, self.grid_size[0] - 1)
            y = start_pos[1]
        elif start_pos[1] == 'random':
            x = start_pos[0]
            y = np.random.uniform(1, self.grid_size[1] - 1)
        else:
            x, y = start_pos
        
        # Lane offset for separation
        if spawn_type in ['left_to_right', 'right_to_left']:
            lane_offset = np.random.choice([-0.3, 0.3])  # Horizontal lanes
            y += lane_offset
        else:
            lane_offset = np.random.choice([-0.3, 0.3])  # Vertical lanes  
            x += lane_offset
        
        # Set velocity
        direction = spawn_info['direction']
        speed = np.random.uniform(0.8, 1.4)
        vx = direction[0] * speed
        vy = direction[1] * speed
        
        # Direction codes
        direction_codes = {'left_to_right': 0, 'right_to_left': 1, 'top_to_bottom': 2, 'bottom_to_top': 3}
        direction_code = direction_codes[spawn_type]
        
        # Store vehicle data
        self.vehicles[vehicle_idx] = [x, y, vx, vy, direction_code, 1, self.vehicle_count, lane_offset]
        self.vehicle_count += 1
        
        if self.debug:
            print(f"Spawned vehicle {self.vehicle_count} at ({x:.1f}, {y:.1f}) going {spawn_type}")
        
        return True

    def _get_front_vehicle_distance(self, vehicle_idx):
        """Get distance to vehicle ahead in same lane."""
        if self.vehicles[vehicle_idx, 5] == 0:
            return float('inf')
        
        x, y, vx, vy, direction_code = self.vehicles[vehicle_idx, :5]
        min_distance = float('inf')
        
        # Check vehicles in same direction
        for i in range(self.max_vehicles):
            if i == vehicle_idx or self.vehicles[i, 5] == 0:
                continue
            if self.vehicles[i, 4] != direction_code:  # Different direction
                continue
            
            other_x, other_y = self.vehicles[i, 0], self.vehicles[i, 1]
            
            # Check if vehicle is ahead in same lane
            if direction_code == 0:  # Left to right
                if other_x > x and abs(other_y - y) < 0.6:
                    distance = other_x - x
                    min_distance = min(min_distance, distance)
            elif direction_code == 1:  # Right to left
                if other_x < x and abs(other_y - y) < 0.6:
                    distance = x - other_x
                    min_distance = min(min_distance, distance)
            elif direction_code == 2:  # Top to bottom
                if other_y > y and abs(other_x - x) < 0.6:
                    distance = other_y - y
                    min_distance = min(min_distance, distance)
            elif direction_code == 3:  # Bottom to top
                if other_y < y and abs(other_x - x) < 0.6:
                    distance = y - other_y
                    min_distance = min(min_distance, distance)
        
        return min_distance

    def _check_intersection_collision(self, vehicle_idx):
        """Check for collision with perpendicular traffic."""
        if self.vehicles[vehicle_idx, 5] == 0:
            return False, []
        
        x, y, direction_code = self.vehicles[vehicle_idx, 0], self.vehicles[vehicle_idx, 1], self.vehicles[vehicle_idx, 4]
        collisions = []
        
        # Check vehicles moving in perpendicular directions
        for i in range(self.max_vehicles):
            if i == vehicle_idx or self.vehicles[i, 5] == 0:
                continue
            
            other_x, other_y, other_dir = self.vehicles[i, 0], self.vehicles[i, 1], self.vehicles[i, 4]
            
            # Check if directions are perpendicular
            horizontal_dirs = [0, 1]  # Left-right, right-left
            vertical_dirs = [2, 3]    # Top-bottom, bottom-top
            
            perpendicular = ((direction_code in horizontal_dirs and other_dir in vertical_dirs) or
                           (direction_code in vertical_dirs and other_dir in horizontal_dirs))
            
            if perpendicular:
                # Check if vehicles are close enough to collide
                distance = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                if distance < 0.8:  # Collision threshold
                    collisions.append(i)
        
        return len(collisions) > 0, collisions

    def step(self, action):
        """Step the simulation with highway-style traffic behavior."""
        # Handle actions for learning agent (vehicle 0 if active)
        agent_vehicle_idx = None
        for i in range(self.max_vehicles):
            if self.vehicles[i, 5] == 1:  # Find first active vehicle as agent
                agent_vehicle_idx = i
                break
        
        rewards = 0
        info = {"collisions": [], "vehicles_spawned": 0, "vehicles_exited": 0, "throughput": 0}
        
        # Apply actions
        for i in range(self.max_vehicles):
            if self.vehicles[i, 5] == 0:
                continue
            
            # Get action (agent action for first vehicle, scripted for others)
            if i == agent_vehicle_idx:
                vehicle_action = action
            else:
                vehicle_action = self._highway_policy(i)
            
            # Apply action (only speed control, no direction change)
            current_speed = np.sqrt(self.vehicles[i, 2]**2 + self.vehicles[i, 3]**2)
            direction_code = int(self.vehicles[i, 4])
            
            if vehicle_action == 1:  # Accelerate
                if current_speed < self.max_speed:
                    speed_increase = min(self.accel, self.max_speed - current_speed)
                    if direction_code == 0:  # Left to right
                        self.vehicles[i, 2] += speed_increase
                    elif direction_code == 1:  # Right to left
                        self.vehicles[i, 2] -= speed_increase
                    elif direction_code == 2:  # Top to bottom
                        self.vehicles[i, 3] += speed_increase
                    elif direction_code == 3:  # Bottom to top
                        self.vehicles[i, 3] -= speed_increase
                        
            elif vehicle_action == 2:  # Brake
                self.vehicles[i, 2] *= self.brake_factor
                self.vehicles[i, 3] *= self.brake_factor
                # Ensure minimum speed
                current_speed = np.sqrt(self.vehicles[i, 2]**2 + self.vehicles[i, 3]**2)
                if current_speed < self.min_speed and current_speed > 0.1:
                    scale = self.min_speed / current_speed
                    self.vehicles[i, 2] *= scale
                    self.vehicles[i, 3] *= scale
        
        # Move vehicles
        vehicles_to_remove = []
        for i in range(self.max_vehicles):
            if self.vehicles[i, 5] == 0:
                continue
            
            # Update position
            self.vehicles[i, 0] += self.vehicles[i, 2] * 1.0  # dt = 1.0
            self.vehicles[i, 1] += self.vehicles[i, 3] * 1.0
            
            # Check if vehicle exited grid
            x, y = self.vehicles[i, 0], self.vehicles[i, 1]
            if (x < -1 or x > self.grid_size[0] + 1 or 
                y < -1 or y > self.grid_size[1] + 1):
                vehicles_to_remove.append(i)
                info["vehicles_exited"] += 1
                info["throughput"] += 1
                if i == agent_vehicle_idx:
                    rewards += 1.0  # Reward for successfully traversing
        
        # Remove exited vehicles
        for i in vehicles_to_remove:
            self.vehicles[i, 5] = 0
        
        # Check collisions
        collision_pairs = set()
        for i in range(self.max_vehicles):
            if self.vehicles[i, 5] == 0:
                continue
            
            has_collision, collision_vehicles = self._check_intersection_collision(i)
            if has_collision:
                for j in collision_vehicles:
                    pair = tuple(sorted([i, j]))
                    collision_pairs.add(pair)
        
        # Handle collisions
        for i, j in collision_pairs:
            # Stop both vehicles
            self.vehicles[i, 2] *= 0.1
            self.vehicles[i, 3] *= 0.1
            self.vehicles[j, 2] *= 0.1
            self.vehicles[j, 3] *= 0.1
            info["collisions"].append((i, j))
            if i == agent_vehicle_idx or j == agent_vehicle_idx:
                rewards -= 1.0  # Collision penalty
        
        # Spawn new vehicles
        if np.random.random() < self.spawn_rate:
            if self._spawn_vehicle():
                info["vehicles_spawned"] += 1
        
        # Calculate rewards for agent
        if agent_vehicle_idx is not None:
            # Reward for maintaining good speed
            current_speed = np.sqrt(self.vehicles[agent_vehicle_idx, 2]**2 + 
                                  self.vehicles[agent_vehicle_idx, 3]**2)
            if 0.8 <= current_speed <= 1.5:
                rewards += 0.1
            
            # Penalty for being too slow or too fast
            if current_speed < 0.3:
                rewards -= 0.05
            elif current_speed > 1.8:
                rewards -= 0.02
            
            # Small reward for forward progress
            direction_code = int(self.vehicles[agent_vehicle_idx, 4])
            if direction_code in [0, 2]:  # Moving right or down
                rewards += 0.01
            else:  # Moving left or up
                rewards += 0.01
        
        self.sim_time += 1.0
        
        # Episode doesn't terminate - continuous traffic simulation
        terminated = False
        truncated = False
        
        obs = self._get_observation()
        return obs, float(rewards), terminated, truncated, info

    def _highway_policy(self, vehicle_idx):
        """Scripted policy for highway driving - maintain safe following distance."""
        front_distance = self._get_front_vehicle_distance(vehicle_idx)
        current_speed = np.sqrt(self.vehicles[vehicle_idx, 2]**2 + self.vehicles[vehicle_idx, 3]**2)
        
        # Simple following behavior
        if front_distance < self.safe_distance:
            return 2  # Brake
        elif front_distance > self.safe_distance * 2 and current_speed < self.max_speed * 0.8:
            return 1  # Accelerate
        else:
            return 0  # Maintain

    def _get_observation(self):
        """Get observation for the learning agent (first active vehicle)."""
        # Find first active vehicle as agent
        agent_idx = None
        for i in range(self.max_vehicles):
            if self.vehicles[i, 5] == 1:
                agent_idx = i
                break
        
        if agent_idx is None:
            return np.zeros(self.obs_dim, dtype=np.float32)
        
        x, y, vx, vy, direction_code = self.vehicles[agent_idx, :5]
        speed = np.sqrt(vx**2 + vy**2)
        front_distance = self._get_front_vehicle_distance(agent_idx)
        
        # Count perpendicular traffic nearby
        side_traffic = 0
        for i in range(self.max_vehicles):
            if i == agent_idx or self.vehicles[i, 5] == 0:
                continue
            other_x, other_y, other_dir = self.vehicles[i, 0], self.vehicles[i, 1], self.vehicles[i, 4]
            distance = np.sqrt((x - other_x)**2 + (y - other_y)**2)
            
            # Check if perpendicular and nearby
            horizontal_dirs = [0, 1]
            vertical_dirs = [2, 3]
            perpendicular = ((direction_code in horizontal_dirs and other_dir in vertical_dirs) or
                           (direction_code in vertical_dirs and other_dir in horizontal_dirs))
            
            if perpendicular and distance < 3.0:
                side_traffic += 1
        
        obs = np.array([
            x / self.grid_size[0],  # Normalized position
            y / self.grid_size[1],
            vx / self.max_speed,    # Normalized velocity
            vy / self.max_speed,
            direction_code / 3.0,   # Normalized direction
            speed / self.max_speed, # Normalized speed
            min(front_distance / 5.0, 1.0),  # Normalized front distance
            min(side_traffic / 3.0, 1.0)     # Normalized side traffic
        ], dtype=np.float32)
        
        return obs

    def get_active_vehicles(self):
        """Get list of active vehicles for visualization."""
        active_vehicles = []
        for i in range(self.max_vehicles):
            if self.vehicles[i, 5] == 1:
                x, y, vx, vy, direction_code, _, vehicle_id, lane_offset = self.vehicles[i]
                active_vehicles.append({
                    'id': int(vehicle_id),
                    'position': (x, y),
                    'velocity': (vx, vy),
                    'direction_code': int(direction_code),
                    'speed': np.sqrt(vx**2 + vy**2),
                    'lane_offset': lane_offset
                })
        return active_vehicles 