import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.intersection_manager import IntersectionManager
from utils.comm_module import CommModule
import matplotlib.pyplot as plt

class CityTrafficEnv(gym.Env):
    """
    Multi-agent Gym environment for simulating city traffic with autonomous vehicles.

    Supports both single-agent (ego vs scripted) and multi-agent (all learning) modes.
    Vehicles move on a grid, interact at intersections via a reservation system, and communicate using a simulated 6G V2V/V2I network.

    Parameters:
        grid_size (tuple): Size of the city grid (rows, cols).
        max_vehicles (int): Maximum number of vehicles in the environment.
        multi_agent (bool): If True, all vehicles are learning agents; if False, only agent 0 is learning.
        debug (bool): If True, prints detailed debug information each step.
        continuous_spawn (bool): Enable continuous vehicle spawning for long episodes.

    Observation:
        [x, y, vx, vy, route_x, route_y, dist_to_inter, stopped, ...nearest_vehicles]
        We'll use 7 + 1 + 4*3 = 20 dims: self state (7), stopped (1), up to 4 nearest vehicles (x, y, vx)

    Action:
        Discrete(5): 0=stay, 1=accelerate, 2=brake, 3=turn left, 4=turn right

    Notes:
        - Time step is 1.0 second per step.
        - Communications are simulated with delay and drop probability (see CommModule).
        - Intersection reservations are managed by IntersectionManager.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=(10, 10), max_vehicles=10, multi_agent=True, debug=False, continuous_spawn=False):
        """
        Initialize the CityTrafficEnv.

        Args:
            grid_size (tuple): Size of the city grid (rows, cols).
            max_vehicles (int): Maximum number of vehicles.
            multi_agent (bool): Multi-agent mode flag.
            debug (bool): Enable debug printouts if True.
            continuous_spawn (bool): Enable continuous vehicle spawning for long episodes.
        """
        super(CityTrafficEnv, self).__init__()
        self.grid_size = grid_size
        self.max_vehicles = max_vehicles
        self.n_actions = 5  # 0: stay, 1: accelerate, 2: brake, 3: turn left, 4: turn right
        self.dt = 1.0
        self.multi_agent = multi_agent
        self.debug = debug
        self.continuous_spawn = continuous_spawn
        self.spawn_rate = 0.3 if continuous_spawn else 0.0  # Spawn rate for continuous mode
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
        
        # Journey tracking for continuous mode
        self.vehicle_spawn_times = {}
        self.journey_times = []
        self.total_vehicles_spawned = 0
        self.total_vehicles_completed = 0

    def reset(self, num_vehicles=None, seed=None, **kwargs):
        """
        Reset the environment to an initial state.

        Args:
            num_vehicles (int, optional): Number of vehicles to initialize. Defaults to max_vehicles.
            seed (int, optional): Random seed for reproducibility.
            **kwargs: Additional arguments for compatibility.

        Returns:
            tuple: (obs, info) where obs is the initial observation(s) and info is a dict.
        """
        if seed is not None:
            np.random.seed(seed)
        if num_vehicles is None:
            self.num_vehicles = self.max_vehicles
        else:
            self.num_vehicles = min(num_vehicles, self.max_vehicles)
        self.agent_ids = [f"agent_{i}" for i in range(self.num_vehicles)]
        self.vehicles = np.zeros((self.max_vehicles, 7), dtype=np.float32)
        for i in range(self.num_vehicles):
            self._spawn_vehicle_at_index(i)
            
        # Initialize journey tracking
        self.vehicle_spawn_times = {}
        self.journey_times = []
        self.total_vehicles_spawned = self.num_vehicles
        self.total_vehicles_completed = 0
        
        # Track spawn times for initial vehicles
        for i in range(self.num_vehicles):
            self.vehicle_spawn_times[i] = self.sim_time
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
        """
        Advance the simulation by one time step.

        Args:
            actions (dict or int): Actions for each agent (dict) or single agent (int).

        Returns:
            tuple: (obs, reward, terminated, truncated, info) where:
                obs: dict or np.ndarray, observations for each agent or single agent.
                reward: dict or float, rewards for each agent or single agent.
                terminated: dict or bool, terminated flags for each agent or single agent.
                truncated: dict or bool, truncated flags for each agent or single agent.
                info: dict, additional info for each agent or single agent.
        Notes:
            - Handles vehicle kinematics, intersection reservations, collisions, and communication.
            - Rewards are shaped for safety and efficiency.
        """
        if self.multi_agent:
            acts = [actions.get(aid, 0) for aid in self.agent_ids]
        else:
            acts = [actions] + [self._scripted_policy(i) for i in range(1, self.num_vehicles)]
        rewards = np.zeros(self.max_vehicles, dtype=np.float32)
        terminated = False
        truncated = False
        info = {"collisions": [], "intersection_denials": [], "messages_sent": 0, "messages_delivered": 0}
        max_speed = 5.0
        accel = 1.0
        # V2V broadcast
        for i in range(self.num_vehicles):
            if self.vehicles[i, 6] == 0:
                continue
            pos_msg = {"type": "position", "vehicle_id": i, "pos": (self.vehicles[i, 0], self.vehicles[i, 1])}
            self.comm.broadcast(pos_msg, sender=i, recipients=range(self.num_vehicles), now=self.sim_time)
            info["messages_sent"] += self.num_vehicles - 1
        # Action application
        for i in range(self.num_vehicles):
            if self.vehicles[i, 6] == 0:
                continue
            action = acts[i]
            vx, vy = self.vehicles[i, 2], self.vehicles[i, 3]
            if vx == 0 and vy == 0:
                direction = np.random.randint(0, 4)
            elif abs(vx) > abs(vy):
                direction = 2 if vx > 0 else 0
            else:
                direction = 1 if vy > 0 else 3
            if self.debug:
                print(f"[DEBUG] Vehicle {i}: pos=({self.vehicles[i,0]:.1f},{self.vehicles[i,1]:.1f}) action={action} vel=({vx:.1f},{vy:.1f})")
            if action == 1:
                if direction == 0:
                    self.vehicles[i, 2] = max(vx - accel, -max_speed)
                elif direction == 1:
                    self.vehicles[i, 3] = min(vy + accel, max_speed)
                elif direction == 2:
                    self.vehicles[i, 2] = min(vx + accel, max_speed)
                elif direction == 3:
                    self.vehicles[i, 3] = max(vy - accel, -max_speed)
            elif action == 2:
                if direction == 0 or direction == 2:
                    self.vehicles[i, 2] *= 0.5
                else:
                    self.vehicles[i, 3] *= 0.5
            elif action == 3:
                if direction == 0:
                    self.vehicles[i, 2], self.vehicles[i, 3] = 0, -abs(vx if vx != 0 else 1)
                elif direction == 1:
                    self.vehicles[i, 2], self.vehicles[i, 3] = -abs(vy if vy != 0 else 1), 0
                elif direction == 2:
                    self.vehicles[i, 2], self.vehicles[i, 3] = 0, abs(vx if vx != 0 else 1)
                elif direction == 3:
                    self.vehicles[i, 2], self.vehicles[i, 3] = abs(vy if vy != 0 else 1), 0
            elif action == 4:
                if direction == 0:
                    self.vehicles[i, 2], self.vehicles[i, 3] = 0, abs(vx if vx != 0 else 1)
                elif direction == 1:
                    self.vehicles[i, 2], self.vehicles[i, 3] = abs(vy if vy != 0 else 1), 0
                elif direction == 2:
                    self.vehicles[i, 2], self.vehicles[i, 3] = 0, -abs(vx if vx != 0 else 1)
                elif direction == 3:
                    self.vehicles[i, 2], self.vehicles[i, 3] = -abs(vy if vy != 0 else 1), 0
        # Intersection requests
        for i in range(self.num_vehicles):
            if self.vehicles[i, 6] == 0:
                continue
            old_x, old_y = self.vehicles[i, 0], self.vehicles[i, 1]
            vx, vy = self.vehicles[i, 2], self.vehicles[i, 3]
            new_x = old_x + vx * self.dt
            new_y = old_y + vy * self.dt
            intersection_x, intersection_y = self.intersection_cell
            will_enter = (int(round(new_x)), int(round(new_y))) == self.intersection_cell
            currently_in = (int(old_x), int(old_y)) == self.intersection_cell
            if will_enter and not currently_in and i not in self.pending_requests:
                from_dir = self._get_direction((old_x, old_y), self.intersection_cell)
                to_dir = self._get_direction(self.intersection_cell, (self.vehicles[i, 4], self.vehicles[i, 5]))
                arrival_time = self.sim_time + self.dt
                duration = 1.0
                if self.debug:
                    print(f"[DEBUG] Vehicle {i} requests intersection: from {from_dir} to {to_dir} at t={arrival_time:.1f}")
                req_msg = {"type": "reservation_request", "vehicle_id": i, "from_dir": from_dir, "to_dir": to_dir, "arrival_time": arrival_time, "duration": duration}
                self.comm.send(req_msg, recipient="intersection", now=self.sim_time)
                self.pending_requests[i] = (from_dir, to_dir, arrival_time, duration)
                info["messages_sent"] += 1
        # Deliver messages
        delivered = self.comm.deliver_messages(self.sim_time)
        info["messages_delivered"] = len(delivered)
        for recipient, message in delivered:
            if recipient == "intersection":
                if message["type"] == "reservation_request":
                    vehicle_id = message["vehicle_id"]
                    from_dir = message["from_dir"]
                    to_dir = message["to_dir"]
                    arrival_time = message["arrival_time"]
                    duration = message["duration"]
                    granted, slot = self.intersection.request_reservation(vehicle_id, arrival_time, duration, (from_dir, to_dir))
                    if self.debug:
                        print(f"[DEBUG] IntersectionManager: Vehicle {vehicle_id} reservation {'GRANTED' if granted else 'DENIED'} for t={arrival_time:.1f}")
                    resp_msg = {"type": "reservation_response", "vehicle_id": vehicle_id, "granted": granted, "slot": slot}
                    self.comm.send(resp_msg, recipient=vehicle_id, now=self.sim_time)
                    info["messages_sent"] += 1
            elif isinstance(recipient, int):
                if message["type"] == "reservation_response":
                    self.intersection_responses[recipient] = (message["granted"], message["slot"])
        # Move vehicles
        positions = {}
        for i in range(self.num_vehicles):
            if self.vehicles[i, 6] == 0:
                continue
            old_x, old_y = self.vehicles[i, 0], self.vehicles[i, 1]
            vx, vy = self.vehicles[i, 2], self.vehicles[i, 3]
            new_x = old_x + vx * self.dt
            new_y = old_y + vy * self.dt
            intersection_x, intersection_y = self.intersection_cell
            will_enter = (int(round(new_x)), int(round(new_y))) == self.intersection_cell
            currently_in = (int(old_x), int(old_y)) == self.intersection_cell
            stopped = (vx == 0 and vy == 0)
            if will_enter and not currently_in:
                granted = False
                if i in self.intersection_responses:
                    granted, slot = self.intersection_responses[i]
                    if granted:
                        self.pending_requests.pop(i, None)
                        self.intersection_responses.pop(i, None)
                if not granted:
                    self.vehicles[i, 2] = 0
                    self.vehicles[i, 3] = 0
                    rewards[i] -= 0.1  # Penalty for waiting
                    info["intersection_denials"].append(i)
                    continue
            if (new_x < 0 or new_x >= self.grid_size[0] or new_y < 0 or new_y >= self.grid_size[1]):
                # Track completion if vehicle exits in continuous mode
                if self.continuous_spawn and i in self.vehicle_spawn_times:
                    journey_time = self.sim_time - self.vehicle_spawn_times[i]
                    self.journey_times.append(journey_time)
                    self.total_vehicles_completed += 1
                    del self.vehicle_spawn_times[i]
                
                self.vehicles[i, 6] = 0
                rewards[i] = -1  # Penalty for leaving grid
                continue
            self.vehicles[i, 0] = int(round(new_x))
            self.vehicles[i, 1] = int(round(new_y))
            pos = (int(self.vehicles[i, 0]), int(self.vehicles[i, 1]))
            if pos in positions:
                other = positions[pos]
                self.vehicles[i, 6] = 0
                self.vehicles[other, 6] = 0
                rewards[i] = -1  # Collision penalty
                rewards[other] = -1
                info["collisions"].append((i, other, pos))
            else:
                positions[pos] = i
            route_x, route_y = self.vehicles[i, 4], self.vehicles[i, 5]
            dist = np.linalg.norm([self.vehicles[i, 0] - route_x, self.vehicles[i, 1] - route_y])
            rewards[i] += -0.01 * dist
            if stopped:
                rewards[i] -= 0.1  # Penalty for being stopped
            if self.vehicles[i, 0] == route_x and self.vehicles[i, 1] == route_y:
                rewards[i] += 1  # Reward for reaching destination
                
                # Track completion in continuous mode
                if self.continuous_spawn and i in self.vehicle_spawn_times:
                    journey_time = self.sim_time - self.vehicle_spawn_times[i]
                    self.journey_times.append(journey_time)
                    self.total_vehicles_completed += 1
                    del self.vehicle_spawn_times[i]
                
                self.vehicles[i, 4] = np.random.randint(0, self.grid_size[0])
                self.vehicles[i, 5] = np.random.randint(0, self.grid_size[1])
        self.intersection.cleanup(self.sim_time + self.dt)
        self.sim_time += self.dt
        # Spawn new vehicles in continuous mode
        if self.continuous_spawn and np.random.random() < self.spawn_rate:
            self._try_spawn_new_vehicle()
        
        # Check if episode is done (all vehicles inactive)
        if self.continuous_spawn:
            terminated = False  # Never terminate in continuous mode
        else:
            terminated = np.sum(self.vehicles[:, 6]) == 0
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
        """
        Compute observations for all vehicles.

        Returns:
            np.ndarray: Array of shape (num_vehicles, obs_dim) with each agent's observation.
        """
        # Handle edge case where no vehicles exist
        if self.num_vehicles == 0:
            return np.zeros((0, self.obs_dim), dtype=np.float32)
        
        # For each vehicle, return [x, y, vx, vy, route_x, route_y, dist_to_inter, stopped, ...nearest_vehicles]
        obs = np.zeros((self.num_vehicles, self.obs_dim), dtype=np.float32)
        for i in range(self.num_vehicles):
            x, y, vx, vy, rx, ry, active = self.vehicles[i]
            dist_to_inter = np.linalg.norm([x - self.intersection_cell[0], y - self.intersection_cell[1]])
            stopped = float(vx == 0 and vy == 0)
            obs[i, :7] = [x, y, vx, vy, rx, ry, dist_to_inter]
            obs[i, 7] = stopped
            # Find up to 4 nearest vehicles (excluding self)
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
                    obs[i, 8 + k*3:8 + (k+1)*3] = self.vehicles[j, :3]  # x, y, vx
                else:
                    obs[i, 8 + k*3:8 + (k+1)*3] = 0
        return obs

    def _scripted_policy(self, i):
        """
        Simple scripted policy for non-learning vehicles.
        Moves toward the route target if stopped, otherwise stays.

        Args:
            i (int): Vehicle index.
        Returns:
            int: Action to take (0-4).
        """
        x, y, vx, vy, rx, ry, active = self.vehicles[i]
        if vx == 0 and vy == 0:
            if rx > x:
                return 1  # accelerate south
            elif rx < x:
                return 1  # accelerate north
            elif ry > y:
                return 1  # accelerate east
            elif ry < y:
                return 1  # accelerate west
            else:
                return 0  # stay
        return 0  # stay

    def _get_direction(self, from_pos, to_pos):
        """
        Compute the direction index from one position to another.

        Args:
            from_pos (tuple): (x, y) start position.
            to_pos (tuple): (x, y) target position.
        Returns:
            int: Direction index (0=N, 1=E, 2=S, 3=W).
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        if abs(dx) > abs(dy):
            return 2 if dx > 0 else 0
        elif abs(dy) > 0:
            return 1 if dy > 0 else 3
        return 0

    def render(self, mode='human'):
        """
        Visualize the grid and vehicles.

        Args:
            mode (str): 'human' or 'plot' for Matplotlib, 'text' for ASCII grid.
        Notes:
            - Vehicles are shown as colored circles, intersection is highlighted.
            - For animation, call repeatedly in a loop.
        """
        if mode == 'text':
            grid = np.full(self.grid_size, '.', dtype=str)
            for i in range(self.num_vehicles):
                if self.vehicles[i, 6] == 1:
                    x, y = int(self.vehicles[i, 0]), int(self.vehicles[i, 1])
                    grid[x, y] = str(i)
            print("\n".join([" ".join(row) for row in grid]))
            return
        # Matplotlib visualization
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        # Draw grid lines
        for x in range(self.grid_size[0]+1):
            ax.plot([x-0.5, x-0.5], [-0.5, self.grid_size[1]-0.5], color='gray', linewidth=0.5)
        for y in range(self.grid_size[1]+1):
            ax.plot([-0.5, self.grid_size[0]-0.5], [y-0.5, y-0.5], color='gray', linewidth=0.5)
        # Draw intersection cell
        ix, iy = self.intersection_cell
        ax.add_patch(plt.Rectangle((iy-0.5, ix-0.5), 1, 1, color='yellow', alpha=0.3, zorder=0, label='Intersection'))
        # Draw vehicles
        colors = plt.cm.tab10.colors
        for i in range(self.num_vehicles):
            if self.vehicles[i, 6] == 1:
                x, y = self.vehicles[i, 0], self.vehicles[i, 1]
                ax.scatter(y, x, s=200, color=colors[i % 10], label=f'Vehicle {i}' if i < 10 else None, edgecolor='k', zorder=2)
        # Legend (only for first 10 vehicles)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small')
        ax.set_xlim(-0.5, self.grid_size[1]-0.5)
        ax.set_ylim(self.grid_size[0]-0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_xticks(range(self.grid_size[1]))
        ax.set_yticks(range(self.grid_size[0]))
        ax.set_xlabel('Y (East-West)')
        ax.set_ylabel('X (North-South)')
        ax.set_title('City Traffic Simulation')
        plt.tight_layout()
        plt.show()
    
    def _spawn_vehicle_at_index(self, i):
        """Spawn a vehicle at the given index."""
        self.vehicles[i, 0] = np.random.randint(0, self.grid_size[0])
        self.vehicles[i, 1] = np.random.randint(0, self.grid_size[1])
        direction = np.random.randint(0, 4)
        speed = np.random.randint(0, 2)
        if direction == 0:
            self.vehicles[i, 2] = -speed
            self.vehicles[i, 3] = 0
        elif direction == 1:
            self.vehicles[i, 2] = 0
            self.vehicles[i, 3] = speed
        elif direction == 2:
            self.vehicles[i, 2] = speed
            self.vehicles[i, 3] = 0
        elif direction == 3:
            self.vehicles[i, 2] = 0
            self.vehicles[i, 3] = -speed
        self.vehicles[i, 4] = np.random.randint(0, self.grid_size[0])
        self.vehicles[i, 5] = np.random.randint(0, self.grid_size[1])
        self.vehicles[i, 6] = 1
    
    def _try_spawn_new_vehicle(self):
        """Try to spawn a new vehicle in an empty slot."""
        # Find empty slot
        for i in range(self.max_vehicles):
            if self.vehicles[i, 6] == 0:  # Not active
                self._spawn_vehicle_at_index(i)
                self.vehicle_spawn_times[i] = self.sim_time
                self.total_vehicles_spawned += 1
                if self.debug:
                    print(f"[6G_CITY] Spawned new vehicle {i} at ({self.vehicles[i, 0]}, {self.vehicles[i, 1]})")
                return True
        return False
    
    def get_journey_statistics(self):
        """Get journey time statistics for continuous mode."""
        if not self.continuous_spawn or not self.journey_times:
            return {
                'count': 0,
                'average_time': 0,
                'min_time': 0,
                'max_time': 0,
                'total_completed': self.total_vehicles_completed,
                'total_spawned': self.total_vehicles_spawned
            }
        
        return {
            'count': len(self.journey_times),
            'average_time': np.mean(self.journey_times),
            'min_time': np.min(self.journey_times),
            'max_time': np.max(self.journey_times),
            'total_completed': self.total_vehicles_completed,
            'total_spawned': self.total_vehicles_spawned
        } 