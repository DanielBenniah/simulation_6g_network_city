"""
Realistic traffic environment where vehicles actually navigate to destinations
with proper pathfinding, traffic rules, and realistic behavior.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.intersection_manager import IntersectionManager
from utils.comm_module import CommModule

class RealisticTrafficEnv(gym.Env):
    """
    Realistic traffic simulation where vehicles:
    - Navigate intelligently toward destinations
    - Follow traffic rules and respect intersections
    - Show realistic speed variations based on traffic conditions
    - Actually reach their intended destinations
    """
    
    def __init__(self, grid_size=(10, 10), max_vehicles=8, multi_agent=False, debug=False):
        super(RealisticTrafficEnv, self).__init__()
        self.grid_size = grid_size
        self.max_vehicles = max_vehicles
        self.n_actions = 5  # 0: stay, 1: accelerate, 2: brake, 3: turn left, 4: turn right
        self.dt = 1.0
        self.multi_agent = multi_agent
        self.debug = debug
        
        # Realistic parameters
        self.max_speed = 1.5      # Realistic city speed
        self.accel = 0.3          # Gradual acceleration
        self.brake_factor = 0.6   # Effective braking
        self.min_following_distance = 1.0  # Safe following distance
        
        self.action_space = spaces.Discrete(self.n_actions)
        self.obs_dim = 7 + 1 + 4*3
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), dtype=np.float32)
        
        self.vehicles = None
        self.num_vehicles = None
        self.intersection = IntersectionManager()
        self.intersection_cell = (grid_size[0] // 2, grid_size[1] // 2)
        self.comm = CommModule()
        self.sim_time = 0.0
        self.pending_requests = {}
        self.intersection_responses = {}
        self.agent_ids = None
        
        # Track destinations reached
        self.destinations_reached = 0

    def reset(self, num_vehicles=None, seed=None, **kwargs):
        """Reset with realistic initial conditions."""
        if seed is not None:
            np.random.seed(seed)
        if num_vehicles is None:
            self.num_vehicles = self.max_vehicles
        else:
            self.num_vehicles = min(num_vehicles, self.max_vehicles)
        
        self.agent_ids = [f"agent_{i}" for i in range(self.num_vehicles)]
        self.vehicles = np.zeros((self.max_vehicles, 7), dtype=np.float32)
        self.destinations_reached = 0
        
        # Realistic initial positioning
        for i in range(self.num_vehicles):
            # Place vehicles at reasonable starting positions
            attempts = 0
            while attempts < 50:  # Prevent infinite loop
                start_x = np.random.randint(0, self.grid_size[0])
                start_y = np.random.randint(0, self.grid_size[1])
                
                # Check if position is clear
                position_clear = True
                for j in range(i):
                    if (abs(self.vehicles[j, 0] - start_x) < 1.5 and 
                        abs(self.vehicles[j, 1] - start_y) < 1.5):
                        position_clear = False
                        break
                
                if position_clear:
                    self.vehicles[i, 0], self.vehicles[i, 1] = start_x, start_y
                    break
                attempts += 1
            
            # Set destination that's reasonably far away
            while True:
                dest_x = np.random.randint(0, self.grid_size[0])
                dest_y = np.random.randint(0, self.grid_size[1])
                distance = np.sqrt((dest_x - start_x)**2 + (dest_y - start_y)**2)
                if distance > 3:  # Ensure meaningful journey
                    self.vehicles[i, 4], self.vehicles[i, 5] = dest_x, dest_y
                    break
            
            # Initial velocity toward destination (realistic)
            dx = self.vehicles[i, 4] - self.vehicles[i, 0]
            dy = self.vehicles[i, 5] - self.vehicles[i, 1]
            
            # Choose primary direction toward destination
            if abs(dx) > abs(dy):
                # Move horizontally first
                initial_speed = np.random.uniform(0.3, 0.7)
                self.vehicles[i, 2] = initial_speed if dx > 0 else -initial_speed
                self.vehicles[i, 3] = 0
            else:
                # Move vertically first
                initial_speed = np.random.uniform(0.3, 0.7)
                self.vehicles[i, 2] = 0
                self.vehicles[i, 3] = initial_speed if dy > 0 else -initial_speed
            
            self.vehicles[i, 6] = 1  # Active
        
        self.intersection = IntersectionManager()
        self.comm = CommModule()
        self.sim_time = 0.0
        self.pending_requests = {}
        self.intersection_responses = {}
        
        obs = self._get_all_obs()
        info = {"destinations_reached": 0}
        if self.multi_agent:
            return {aid: obs[i] for i, aid in enumerate(self.agent_ids)}, info
        else:
            return obs[0], info

    def _get_next_direction_toward_destination(self, vehicle_idx):
        """Calculate the best next direction to reach destination."""
        x, y = self.vehicles[vehicle_idx, 0], self.vehicles[vehicle_idx, 1]
        dest_x, dest_y = self.vehicles[vehicle_idx, 4], self.vehicles[vehicle_idx, 5]
        
        dx = dest_x - x
        dy = dest_y - y
        
        # If very close to destination, we've arrived
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return None  # Arrived
        
        # Choose direction based on larger distance component
        if abs(dx) > abs(dy):
            return 1 if dx > 0 else 3  # East or West
        else:
            return 2 if dy > 0 else 0  # South or North

    def _check_path_clear(self, vehicle_idx, direction):
        """Check if the path ahead is clear for safe movement."""
        x, y = self.vehicles[vehicle_idx, 0], self.vehicles[vehicle_idx, 1]
        
        # Check positions ahead based on direction
        check_positions = []
        if direction == 0:  # North
            check_positions = [(x, y-1), (x, y-2)]
        elif direction == 1:  # East
            check_positions = [(x+1, y), (x+2, y)]
        elif direction == 2:  # South
            check_positions = [(x, y+1), (x, y+2)]
        elif direction == 3:  # West
            check_positions = [(x-1, y), (x-2, y)]
        
        # Check if any other vehicle is in these positions
        for check_x, check_y in check_positions:
            for j in range(self.num_vehicles):
                if j == vehicle_idx or self.vehicles[j, 6] == 0:
                    continue
                other_x, other_y = self.vehicles[j, 0], self.vehicles[j, 1]
                if abs(other_x - check_x) < 0.8 and abs(other_y - check_y) < 0.8:
                    return False
        return True

    def _realistic_scripted_policy(self, vehicle_idx):
        """Realistic navigation policy that actually tries to reach destinations."""
        x, y, vx, vy, dest_x, dest_y, active = self.vehicles[vehicle_idx]
        current_speed = np.sqrt(vx**2 + vy**2)
        
        # Get desired direction toward destination
        desired_direction = self._get_next_direction_toward_destination(vehicle_idx)
        
        if desired_direction is None:
            # Arrived at destination - brake
            return 2 if current_speed > 0.1 else 0
        
        # Current direction
        if abs(vx) > abs(vy):
            current_direction = 1 if vx > 0 else 3  # East or West
        else:
            current_direction = 2 if vy > 0 else 0  # South or North
        
        # Check if path ahead is clear
        path_clear = self._check_path_clear(vehicle_idx, desired_direction)
        
        # Decision logic
        if current_direction != desired_direction:
            # Need to turn toward destination
            if desired_direction == (current_direction + 1) % 4:
                return 4  # Turn right
            elif desired_direction == (current_direction - 1) % 4:
                return 3  # Turn left
            else:
                # Need to turn around - choose shortest turn
                return 3  # Turn left (arbitrary choice)
        
        elif not path_clear:
            # Path blocked - slow down or stop
            if current_speed > 0.3:
                return 2  # Brake
            else:
                return 0  # Stay (wait)
        
        elif current_speed < 0.8:
            # Path clear and moving in right direction - accelerate
            return 1
        
        else:
            # Cruising at good speed
            return 0

    def step(self, actions):
        """Step with realistic traffic behavior."""
        if self.multi_agent:
            acts = [actions.get(aid, 0) for aid in self.agent_ids]
        else:
            acts = [actions] + [self._realistic_scripted_policy(i) for i in range(1, self.num_vehicles)]
        
        rewards = np.zeros(self.max_vehicles, dtype=np.float32)
        terminated = False
        truncated = False
        info = {"collisions": [], "intersection_denials": [], "messages_sent": 0, 
                "messages_delivered": 0, "destinations_reached": 0}
        
        # Apply actions with realistic physics
        for i in range(self.num_vehicles):
            if self.vehicles[i, 6] == 0:
                continue
                
            action = acts[i]
            vx, vy = self.vehicles[i, 2], self.vehicles[i, 3]
            current_speed = np.sqrt(vx**2 + vy**2)
            
            # Determine current direction
            if abs(vx) > abs(vy):
                direction = 1 if vx > 0 else 3  # East or West
            else:
                direction = 2 if vy > 0 else 0  # South or North
            
            # Apply action with realistic constraints
            if action == 1:  # Accelerate
                if current_speed < self.max_speed:
                    if direction == 0:  # North
                        self.vehicles[i, 3] = max(vy - self.accel, -self.max_speed)
                    elif direction == 1:  # East
                        self.vehicles[i, 2] = min(vx + self.accel, self.max_speed)
                    elif direction == 2:  # South
                        self.vehicles[i, 3] = min(vy + self.accel, self.max_speed)
                    else:  # West
                        self.vehicles[i, 2] = max(vx - self.accel, -self.max_speed)
                        
            elif action == 2:  # Brake
                self.vehicles[i, 2] *= self.brake_factor
                self.vehicles[i, 3] *= self.brake_factor
                
            elif action == 3:  # Turn left
                turn_speed = min(current_speed, 0.8)  # Slow down for turns
                if direction == 0:  # North -> West
                    self.vehicles[i, 2], self.vehicles[i, 3] = -turn_speed, 0
                elif direction == 1:  # East -> North
                    self.vehicles[i, 2], self.vehicles[i, 3] = 0, -turn_speed
                elif direction == 2:  # South -> East
                    self.vehicles[i, 2], self.vehicles[i, 3] = turn_speed, 0
                else:  # West -> South
                    self.vehicles[i, 2], self.vehicles[i, 3] = 0, turn_speed
                    
            elif action == 4:  # Turn right
                turn_speed = min(current_speed, 0.8)  # Slow down for turns
                if direction == 0:  # North -> East
                    self.vehicles[i, 2], self.vehicles[i, 3] = turn_speed, 0
                elif direction == 1:  # East -> South
                    self.vehicles[i, 2], self.vehicles[i, 3] = 0, turn_speed
                elif direction == 2:  # South -> West
                    self.vehicles[i, 2], self.vehicles[i, 3] = -turn_speed, 0
                else:  # West -> North
                    self.vehicles[i, 2], self.vehicles[i, 3] = 0, -turn_speed

        # Move vehicles with realistic boundary handling
        positions = {}
        for i in range(self.num_vehicles):
            if self.vehicles[i, 6] == 0:
                continue
                
            old_x, old_y = self.vehicles[i, 0], self.vehicles[i, 1]
            vx, vy = self.vehicles[i, 2], self.vehicles[i, 3]
            new_x = old_x + vx * self.dt
            new_y = old_y + vy * self.dt
            
            # Realistic boundary handling - stop at edges
            if new_x < 0 or new_x >= self.grid_size[0] or new_y < 0 or new_y >= self.grid_size[1]:
                # Stop at boundary and reverse direction
                self.vehicles[i, 2] *= -0.5
                self.vehicles[i, 3] *= -0.5
                new_x = max(0, min(new_x, self.grid_size[0] - 1))
                new_y = max(0, min(new_y, self.grid_size[1] - 1))
                rewards[i] -= 0.2  # Penalty for hitting boundary
            
            self.vehicles[i, 0] = new_x
            self.vehicles[i, 1] = new_y
            
            # Collision detection
            pos = (int(round(self.vehicles[i, 0])), int(round(self.vehicles[i, 1])))
            if pos in positions:
                other = positions[pos]
                # Realistic collision response - both stop
                self.vehicles[i, 2] = 0
                self.vehicles[i, 3] = 0
                self.vehicles[other, 2] = 0
                self.vehicles[other, 3] = 0
                rewards[i] -= 1.0
                rewards[other] -= 1.0
                info["collisions"].append((i, other, pos))
            else:
                positions[pos] = i
            
            # Realistic reward structure
            route_x, route_y = self.vehicles[i, 4], self.vehicles[i, 5]
            dist_to_dest = np.sqrt((self.vehicles[i, 0] - route_x)**2 + (self.vehicles[i, 1] - route_y)**2)
            
            # Reward for getting closer to destination
            rewards[i] += -0.01 * dist_to_dest
            
            # Reward for appropriate speed
            current_speed = np.sqrt(self.vehicles[i, 2]**2 + self.vehicles[i, 3]**2)
            if 0.3 <= current_speed <= 1.2:
                rewards[i] += 0.02  # Good speed
            elif current_speed < 0.1:
                rewards[i] -= 0.05  # Penalty for being stopped too long
            
            # Check if reached destination
            if dist_to_dest < 0.7:
                rewards[i] += 2.0  # Big reward for reaching destination
                info["destinations_reached"] += 1
                self.destinations_reached += 1
                
                # Set new destination
                while True:
                    new_dest_x = np.random.randint(0, self.grid_size[0])
                    new_dest_y = np.random.randint(0, self.grid_size[1])
                    new_dist = np.sqrt((self.vehicles[i, 0] - new_dest_x)**2 + (self.vehicles[i, 1] - new_dest_y)**2)
                    if new_dist > 3:  # Ensure meaningful new journey
                        self.vehicles[i, 4] = new_dest_x
                        self.vehicles[i, 5] = new_dest_y
                        break

        self.sim_time += self.dt
        
        # Episode continues - realistic traffic doesn't "end"
        terminated = False
        
        obs = self._get_all_obs()
        if self.multi_agent:
            rew = {aid: rewards[i] for i, aid in enumerate(self.agent_ids)}
            obs_dict = {aid: obs[i] for i, aid in enumerate(self.agent_ids)}
            terminated_dict = {aid: terminated for aid in self.agent_ids}
            truncated_dict = {aid: truncated for aid in self.agent_ids}
            info_dict = {aid: info for aid in self.agent_ids}
            return obs_dict, rew, terminated_dict, truncated_dict, info_dict
        else:
            return obs[0], float(rewards[0]), bool(terminated), bool(truncated), info

    def _get_all_obs(self):
        """Get observations for all vehicles."""
        if self.num_vehicles == 0:
            return np.zeros((0, self.obs_dim), dtype=np.float32)
        
        obs = np.zeros((self.num_vehicles, self.obs_dim), dtype=np.float32)
        for i in range(self.num_vehicles):
            x, y, vx, vy, rx, ry, active = self.vehicles[i]
            dist_to_inter = np.linalg.norm([x - self.intersection_cell[0], y - self.intersection_cell[1]])
            stopped = float(np.sqrt(vx**2 + vy**2) < 0.1)
            obs[i, :7] = [x, y, vx, vy, rx, ry, dist_to_inter]
            obs[i, 7] = stopped
            
            # Find nearest vehicles
            dists = []
            for j in range(self.num_vehicles):
                if j == i or self.vehicles[j, 6] == 0:
                    continue
                dx, dy = self.vehicles[j, 0] - x, self.vehicles[j, 1] - y
                dist = np.hypot(dx, dy)
                dists.append((dist, j))
            dists.sort()
            
            for k in range(4):
                if k < len(dists):
                    _, j = dists[k]
                    obs[i, 8 + k*3:8 + (k+1)*3] = self.vehicles[j, :3]
                else:
                    obs[i, 8 + k*3:8 + (k+1)*3] = 0
        return obs 