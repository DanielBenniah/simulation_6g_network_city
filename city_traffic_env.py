import gym
from gym import spaces
import numpy as np
from .intersection_manager import IntersectionManager
from .comm_module import CommModule

class CityTrafficEnv(gym.Env):
    """
    A Gym environment simulating a city traffic grid with autonomous vehicles.
    Supports a variable number of vehicles and multi-agent actions.
    Implements simple kinematics, collision/out-of-bounds checks, intersection reservation, and 6G V2V/V2I communication.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=(10, 10), max_vehicles=10):
        super(CityTrafficEnv, self).__init__()
        self.grid_size = grid_size  # (rows, cols) of the city grid
        self.max_vehicles = max_vehicles  # Maximum number of vehicles supported
        self.n_actions = 5  # 0: stay, 1: accelerate, 2: brake, 3: turn left, 4: turn right
        self.dt = 1.0  # Time step in seconds

        # Action space: one discrete action per vehicle
        self.action_space = spaces.MultiDiscrete([self.n_actions] * self.max_vehicles)

        # Observation space: for each vehicle: (x, y, vx, vy, route_x, route_y, active)
        # x, y: position in grid; vx, vy: velocity; route_x, route_y: next route target; active: 0/1
        obs_low = np.array([0, 0, -5, -5, 0, 0, 0] * self.max_vehicles, dtype=np.float32)
        obs_high = np.array([
            grid_size[0]-1, grid_size[1]-1, 5, 5, grid_size[0]-1, grid_size[1]-1, 1
        ] * self.max_vehicles, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.vehicles = None  # Will hold vehicle states
        self.num_vehicles = None

        # Intersection manager for the central intersection
        self.intersection = IntersectionManager()
        self.intersection_cell = (grid_size[0] // 2, grid_size[1] // 2)

        # Communication module for V2V and V2I
        self.comm = CommModule()
        self.sim_time = 0.0

        # Track pending intersection requests: vehicle_id -> (from_dir, to_dir, arrival_time, duration)
        self.pending_requests = {}
        # Track intersection responses: vehicle_id -> (granted, slot)
        self.intersection_responses = {}

    def reset(self, num_vehicles=None):
        """
        Reset the environment to an initial state.
        Optionally specify the number of vehicles (default: max_vehicles).
        Returns the initial observation.
        """
        if num_vehicles is None:
            self.num_vehicles = self.max_vehicles
        else:
            self.num_vehicles = min(num_vehicles, self.max_vehicles)

        # Vehicle state: [x, y, vx, vy, route_x, route_y, active]
        self.vehicles = np.zeros((self.max_vehicles, 7), dtype=np.float32)
        for i in range(self.num_vehicles):
            # Random initial position
            self.vehicles[i, 0] = np.random.randint(0, self.grid_size[0])  # x
            self.vehicles[i, 1] = np.random.randint(0, self.grid_size[1])  # y
            # Random initial velocity (0 or 1 in a random direction)
            direction = np.random.randint(0, 4)
            speed = np.random.randint(0, 2)
            if direction == 0:  # North
                self.vehicles[i, 2] = -speed
                self.vehicles[i, 3] = 0
            elif direction == 1:  # East
                self.vehicles[i, 2] = 0
                self.vehicles[i, 3] = speed
            elif direction == 2:  # South
                self.vehicles[i, 2] = speed
                self.vehicles[i, 3] = 0
            elif direction == 3:  # West
                self.vehicles[i, 2] = 0
                self.vehicles[i, 3] = -speed
            # Random route target (for now, just a random grid cell)
            self.vehicles[i, 4] = np.random.randint(0, self.grid_size[0])  # route_x
            self.vehicles[i, 5] = np.random.randint(0, self.grid_size[1])  # route_y
            self.vehicles[i, 6] = 1  # active
        # Inactive vehicles remain at zero state
        self.intersection = IntersectionManager()  # Reset intersection reservations
        self.comm = CommModule()  # Reset communication
        self.sim_time = 0.0
        self.pending_requests = {}
        self.intersection_responses = {}
        return self._get_obs()

    def step(self, actions):
        """
        Take a step in the environment using the provided actions for each vehicle.
        actions: array-like of length max_vehicles (only first num_vehicles are used)
        Returns: obs, reward, done, info
        """
        rewards = np.zeros(self.max_vehicles, dtype=np.float32)
        done = False
        info = {"collisions": [], "intersection_denials": [], "messages_sent": 0, "messages_delivered": 0}

        # Parameters for kinematics
        max_speed = 5.0
        accel = 1.0  # acceleration per action
        # Directions: 0=N, 1=E, 2=S, 3=W

        # 1. Vehicles broadcast their positions to all others (V2V)
        for i in range(self.num_vehicles):
            if self.vehicles[i, 6] == 0:
                continue
            pos_msg = {"type": "position", "vehicle_id": i, "pos": (self.vehicles[i, 0], self.vehicles[i, 1])}
            self.comm.broadcast(pos_msg, sender=i, recipients=range(self.num_vehicles), now=self.sim_time)
            info["messages_sent"] += self.num_vehicles - 1

        # 2. Vehicles process actions and prepare intersection requests if needed
        for i in range(self.num_vehicles):
            if self.vehicles[i, 6] == 0:
                continue
            action = actions[i]
            vx, vy = self.vehicles[i, 2], self.vehicles[i, 3]
            # Determine current direction
            if vx == 0 and vy == 0:
                direction = np.random.randint(0, 4)
            elif abs(vx) > abs(vy):
                direction = 2 if vx > 0 else 0
            else:
                direction = 1 if vy > 0 else 3
            # Action effects
            if action == 1:  # Accelerate
                if direction == 0:
                    self.vehicles[i, 2] = max(vx - accel, -max_speed)
                elif direction == 1:
                    self.vehicles[i, 3] = min(vy + accel, max_speed)
                elif direction == 2:
                    self.vehicles[i, 2] = min(vx + accel, max_speed)
                elif direction == 3:
                    self.vehicles[i, 3] = max(vy - accel, -max_speed)
            elif action == 2:  # Brake
                if direction == 0 or direction == 2:
                    self.vehicles[i, 2] *= 0.5
                else:
                    self.vehicles[i, 3] *= 0.5
            elif action == 3:  # Turn left (change direction, keep speed)
                if direction == 0:  # N -> W
                    self.vehicles[i, 2], self.vehicles[i, 3] = 0, -abs(vx if vx != 0 else 1)
                elif direction == 1:  # E -> N
                    self.vehicles[i, 2], self.vehicles[i, 3] = -abs(vy if vy != 0 else 1), 0
                elif direction == 2:  # S -> E
                    self.vehicles[i, 2], self.vehicles[i, 3] = 0, abs(vx if vx != 0 else 1)
                elif direction == 3:  # W -> S
                    self.vehicles[i, 2], self.vehicles[i, 3] = abs(vy if vy != 0 else 1), 0
            elif action == 4:  # Turn right (change direction, keep speed)
                if direction == 0:  # N -> E
                    self.vehicles[i, 2], self.vehicles[i, 3] = 0, abs(vx if vx != 0 else 1)
                elif direction == 1:  # E -> S
                    self.vehicles[i, 2], self.vehicles[i, 3] = abs(vy if vy != 0 else 1), 0
                elif direction == 2:  # S -> W
                    self.vehicles[i, 2], self.vehicles[i, 3] = 0, -abs(vx if vx != 0 else 1)
                elif direction == 3:  # W -> N
                    self.vehicles[i, 2], self.vehicles[i, 3] = -abs(vy if vy != 0 else 1), 0
            # 0: stay (no change)

        # 3. Vehicles that want to enter the intersection send reservation requests (V2I)
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
                arrival_time = self.sim_time + self.dt  # Predict arrival next step
                duration = 1.0
                req_msg = {"type": "reservation_request", "vehicle_id": i, "from_dir": from_dir, "to_dir": to_dir, "arrival_time": arrival_time, "duration": duration}
                self.comm.send(req_msg, recipient="intersection", now=self.sim_time)
                self.pending_requests[i] = (from_dir, to_dir, arrival_time, duration)
                info["messages_sent"] += 1

        # 4. Deliver all messages whose delay has elapsed
        delivered = self.comm.deliver_messages(self.sim_time)
        info["messages_delivered"] = len(delivered)
        for recipient, message in delivered:
            if recipient == "intersection":
                # Intersection manager processes reservation requests
                if message["type"] == "reservation_request":
                    vehicle_id = message["vehicle_id"]
                    from_dir = message["from_dir"]
                    to_dir = message["to_dir"]
                    arrival_time = message["arrival_time"]
                    duration = message["duration"]
                    granted, slot = self.intersection.request_reservation(vehicle_id, arrival_time, duration, (from_dir, to_dir))
                    resp_msg = {"type": "reservation_response", "vehicle_id": vehicle_id, "granted": granted, "slot": slot}
                    self.comm.send(resp_msg, recipient=vehicle_id, now=self.sim_time)
                    info["messages_sent"] += 1
            elif isinstance(recipient, int):
                # Vehicle receives a message
                if message["type"] == "reservation_response":
                    self.intersection_responses[recipient] = (message["granted"], message["slot"])
                # V2V position messages could be used for cooperative driving, etc.
                # (Not used in this simple example)

        # 5. Move vehicles, considering intersection permissions
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
            if will_enter and not currently_in:
                # Only move if reservation granted
                granted = False
                if i in self.intersection_responses:
                    granted, slot = self.intersection_responses[i]
                    if granted:
                        # Remove request/response after use
                        self.pending_requests.pop(i, None)
                        self.intersection_responses.pop(i, None)
                if not granted:
                    # Denied or no response yet: vehicle must wait
                    self.vehicles[i, 2] = 0
                    self.vehicles[i, 3] = 0
                    rewards[i] -= 0.5
                    info["intersection_denials"].append(i)
                    continue
            # Out-of-bounds check
            if (new_x < 0 or new_x >= self.grid_size[0] or
                new_y < 0 or new_y >= self.grid_size[1]):
                # Out of bounds: deactivate vehicle, large penalty
                self.vehicles[i, 6] = 0
                rewards[i] = -100
                continue
            # Snap to grid (discrete positions)
            self.vehicles[i, 0] = int(round(new_x))
            self.vehicles[i, 1] = int(round(new_y))
            # Collision check
            pos = (int(self.vehicles[i, 0]), int(self.vehicles[i, 1]))
            if pos in positions:
                # Collision: deactivate both vehicles, penalty
                other = positions[pos]
                self.vehicles[i, 6] = 0
                self.vehicles[other, 6] = 0
                rewards[i] = -50
                rewards[other] = -50
                info["collisions"].append((i, other, pos))
            else:
                positions[pos] = i
            # Reward for moving closer to route target
            route_x, route_y = self.vehicles[i, 4], self.vehicles[i, 5]
            dist = np.linalg.norm([self.vehicles[i, 0] - route_x, self.vehicles[i, 1] - route_y])
            rewards[i] += -dist * 0.1
            # Reward for reaching route target
            if self.vehicles[i, 0] == route_x and self.vehicles[i, 1] == route_y:
                rewards[i] += 20
                # Assign new random route
                self.vehicles[i, 4] = np.random.randint(0, self.grid_size[0])
                self.vehicles[i, 5] = np.random.randint(0, self.grid_size[1])
        self.intersection.cleanup(self.sim_time + self.dt)  # Clean up old reservations
        self.sim_time += self.dt
        # Done if all vehicles inactive
        if np.sum(self.vehicles[:, 6]) == 0:
            done = True
        return self._get_obs(), rewards, done, info

    def _get_direction(self, from_pos, to_pos):
        """
        Returns a direction index (0=N, 1=E, 2=S, 3=W) from from_pos to to_pos.
        """
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        if abs(dx) > abs(dy):
            return 2 if dx > 0 else 0
        elif abs(dy) > 0:
            return 1 if dy > 0 else 3
        return 0  # Default North

    def _get_obs(self):
        """
        Returns the flattened observation for all vehicles (including inactive).
        """
        return self.vehicles.flatten()

    def render(self, mode='human'):
        """
        Render the environment (optional, simple text output).
        """
        grid = np.full(self.grid_size, '.', dtype=str)
        for i in range(self.num_vehicles):
            if self.vehicles[i, 6] == 1:
                x, y = int(self.vehicles[i, 0]), int(self.vehicles[i, 1])
                grid[x, y] = str(i)
        print("\n".join([" ".join(row) for row in grid])) 