"""
Demo version of CityTrafficEnv optimized for visualization and demonstration.
This version has modified parameters to keep vehicles active longer and show
more interesting traffic behavior including speed variations.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from intersection_manager import IntersectionManager
from comm_module import CommModule

class DemoTrafficEnv(gym.Env):
    """
    Demo version of traffic environment optimized for visualization.
    
    Key differences from original:
    - Lower maximum speeds to prevent vehicles from leaving grid quickly
    - Boundary wrapping instead of deactivation
    - Better reward structure for longer episodes
    - More conservative movement to show speed variations
    """
    
    def __init__(self, grid_size=(10, 10), max_vehicles=8, multi_agent=False, debug=False):
        super(DemoTrafficEnv, self).__init__()
        self.grid_size = grid_size
        self.max_vehicles = max_vehicles
        self.n_actions = 5  # 0: stay, 1: accelerate, 2: brake, 3: turn left, 4: turn right
        self.dt = 1.0
        self.multi_agent = multi_agent
        self.debug = debug
        
        # Demo-specific parameters
        self.max_speed = 2.0  # Reduced from 5.0 to keep vehicles on grid longer
        self.accel = 0.5      # Reduced from 1.0 for smoother acceleration
        self.brake_factor = 0.7  # More gradual braking
        
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

    def reset(self, num_vehicles=None, seed=None, **kwargs):
        """Reset with better initial conditions for demo."""
        if seed is not None:
            np.random.seed(seed)
        if num_vehicles is None:
            self.num_vehicles = self.max_vehicles
        else:
            self.num_vehicles = min(num_vehicles, self.max_vehicles)
        
        self.agent_ids = [f"agent_{i}" for i in range(self.num_vehicles)]
        self.vehicles = np.zeros((self.max_vehicles, 7), dtype=np.float32)
        
        # Better initial positioning - spread vehicles around the grid
        for i in range(self.num_vehicles):
            # Position vehicles around the perimeter to avoid immediate collisions
            if i < 4:
                # Place first 4 vehicles on edges
                if i == 0:
                    self.vehicles[i, 0], self.vehicles[i, 1] = 0, self.grid_size[1] // 2
                elif i == 1:
                    self.vehicles[i, 0], self.vehicles[i, 1] = self.grid_size[0] - 1, self.grid_size[1] // 2
                elif i == 2:
                    self.vehicles[i, 0], self.vehicles[i, 1] = self.grid_size[0] // 2, 0
                else:
                    self.vehicles[i, 0], self.vehicles[i, 1] = self.grid_size[0] // 2, self.grid_size[1] - 1
            else:
                # Place remaining vehicles randomly but away from center
                while True:
                    x = np.random.randint(0, self.grid_size[0])
                    y = np.random.randint(0, self.grid_size[1])
                    # Avoid center intersection area
                    if abs(x - self.intersection_cell[0]) > 1 or abs(y - self.intersection_cell[1]) > 1:
                        self.vehicles[i, 0], self.vehicles[i, 1] = x, y
                        break
            
            # Initial velocity - start slow
            direction = np.random.randint(0, 4)
            speed = np.random.uniform(0.2, 0.8)  # Start with low speeds
            if direction == 0:  # North
                self.vehicles[i, 2], self.vehicles[i, 3] = 0, -speed
            elif direction == 1:  # East
                self.vehicles[i, 2], self.vehicles[i, 3] = speed, 0
            elif direction == 2:  # South
                self.vehicles[i, 2], self.vehicles[i, 3] = 0, speed
            else:  # West
                self.vehicles[i, 2], self.vehicles[i, 3] = -speed, 0
            
            # Random destination
            self.vehicles[i, 4] = np.random.randint(0, self.grid_size[0])
            self.vehicles[i, 5] = np.random.randint(0, self.grid_size[1])
            self.vehicles[i, 6] = 1  # Active
        
        self.intersection = IntersectionManager()
        self.comm = CommModule()
        self.sim_time = 0.0
        self.pending_requests = {}
        self.intersection_responses = {}
        
        obs = self._get_all_obs()
        info = {}
        if self.multi_agent:
            return {aid: obs[i] for i, aid in enumerate(self.agent_ids)}, info
        else:
            return obs[0], info

    def step(self, actions):
        """Step with demo-friendly modifications."""
        if self.multi_agent:
            acts = [actions.get(aid, 0) for aid in self.agent_ids]
        else:
            acts = [actions] + [self._demo_scripted_policy(i) for i in range(1, self.num_vehicles)]
        
        rewards = np.zeros(self.max_vehicles, dtype=np.float32)
        terminated = False
        truncated = False
        info = {"collisions": [], "intersection_denials": [], "messages_sent": 0, "messages_delivered": 0}
        
        # Apply actions with demo-friendly physics
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
            
            # Apply action
            if action == 1:  # Accelerate
                if current_speed < self.max_speed:
                    scale = min(1.0, (self.max_speed - current_speed) / self.max_speed)
                    if direction == 0:  # North
                        self.vehicles[i, 3] = max(vy - self.accel * scale, -self.max_speed)
                    elif direction == 1:  # East
                        self.vehicles[i, 2] = min(vx + self.accel * scale, self.max_speed)
                    elif direction == 2:  # South
                        self.vehicles[i, 3] = min(vy + self.accel * scale, self.max_speed)
                    else:  # West
                        self.vehicles[i, 2] = max(vx - self.accel * scale, -self.max_speed)
                        
            elif action == 2:  # Brake
                self.vehicles[i, 2] *= self.brake_factor
                self.vehicles[i, 3] *= self.brake_factor
                
            elif action == 3:  # Turn left
                # Preserve speed but change direction
                if direction == 0:  # North -> West
                    self.vehicles[i, 2], self.vehicles[i, 3] = -current_speed, 0
                elif direction == 1:  # East -> North
                    self.vehicles[i, 2], self.vehicles[i, 3] = 0, -current_speed
                elif direction == 2:  # South -> East
                    self.vehicles[i, 2], self.vehicles[i, 3] = current_speed, 0
                else:  # West -> South
                    self.vehicles[i, 2], self.vehicles[i, 3] = 0, current_speed
                    
            elif action == 4:  # Turn right
                # Preserve speed but change direction
                if direction == 0:  # North -> East
                    self.vehicles[i, 2], self.vehicles[i, 3] = current_speed, 0
                elif direction == 1:  # East -> South
                    self.vehicles[i, 2], self.vehicles[i, 3] = 0, current_speed
                elif direction == 2:  # South -> West
                    self.vehicles[i, 2], self.vehicles[i, 3] = -current_speed, 0
                else:  # West -> North
                    self.vehicles[i, 2], self.vehicles[i, 3] = 0, -current_speed

        # Move vehicles with boundary wrapping instead of deactivation
        positions = {}
        for i in range(self.num_vehicles):
            if self.vehicles[i, 6] == 0:
                continue
                
            old_x, old_y = self.vehicles[i, 0], self.vehicles[i, 1]
            vx, vy = self.vehicles[i, 2], self.vehicles[i, 3]
            new_x = old_x + vx * self.dt
            new_y = old_y + vy * self.dt
            
            # Wrap around boundaries instead of deactivating
            new_x = new_x % self.grid_size[0]
            new_y = new_y % self.grid_size[1]
            
            self.vehicles[i, 0] = new_x
            self.vehicles[i, 1] = new_y
            
            # Check for collisions
            pos = (int(round(self.vehicles[i, 0])), int(round(self.vehicles[i, 1])))
            if pos in positions:
                other = positions[pos]
                # Instead of deactivating, just stop both vehicles temporarily
                self.vehicles[i, 2] *= 0.1
                self.vehicles[i, 3] *= 0.1
                self.vehicles[other, 2] *= 0.1
                self.vehicles[other, 3] *= 0.1
                rewards[i] -= 0.5
                rewards[other] -= 0.5
                info["collisions"].append((i, other, pos))
            else:
                positions[pos] = i
            
            # Reward structure for demo
            route_x, route_y = self.vehicles[i, 4], self.vehicles[i, 5]
            dist = np.linalg.norm([self.vehicles[i, 0] - route_x, self.vehicles[i, 1] - route_y])
            rewards[i] += -0.001 * dist  # Small penalty for distance
            
            current_speed = np.sqrt(self.vehicles[i, 2]**2 + self.vehicles[i, 3]**2)
            if current_speed < 0.1:
                rewards[i] -= 0.01  # Small penalty for being stopped
            else:
                rewards[i] += 0.01  # Small reward for moving
            
            # Check if reached destination
            if abs(self.vehicles[i, 0] - route_x) < 0.5 and abs(self.vehicles[i, 1] - route_y) < 0.5:
                rewards[i] += 0.5  # Reward for reaching destination
                # Set new destination
                self.vehicles[i, 4] = np.random.randint(0, self.grid_size[0])
                self.vehicles[i, 5] = np.random.randint(0, self.grid_size[1])

        self.sim_time += self.dt
        
        # Don't terminate episodes early - let them run longer
        # Only terminate if explicitly requested or after very long time
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

    def _demo_scripted_policy(self, i):
        """Demo-friendly scripted policy that creates interesting behavior."""
        x, y, vx, vy, rx, ry, active = self.vehicles[i]
        current_speed = np.sqrt(vx**2 + vy**2)
        
        # More interesting scripted behavior
        if current_speed < 0.2:
            return 1  # Accelerate if too slow
        elif current_speed > 1.5:
            return 2  # Brake if too fast
        elif np.random.random() < 0.1:
            return np.random.choice([3, 4])  # Occasionally turn
        else:
            return 0  # Stay course

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