import gym
from gym import spaces
import numpy as np

class CityTrafficEnv(gym.Env):
    """
    A Gym environment simulating a city traffic grid with autonomous vehicles.
    Supports a variable number of vehicles and multi-agent actions.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=(10, 10), max_vehicles=10):
        super(CityTrafficEnv, self).__init__()
        self.grid_size = grid_size  # (rows, cols) of the city grid
        self.max_vehicles = max_vehicles  # Maximum number of vehicles supported
        self.n_actions = 5  # 0: stay, 1: accelerate, 2: decelerate, 3: turn left, 4: turn right

        # Action space: one discrete action per vehicle
        self.action_space = spaces.MultiDiscrete([self.n_actions] * self.max_vehicles)

        # Observation space: for each vehicle: (x, y, speed, direction, active)
        # x, y: position in grid; speed: 0-5; direction: 0=N,1=E,2=S,3=W; active: 0/1
        obs_low = np.array([0, 0, 0, 0, 0] * self.max_vehicles, dtype=np.float32)
        obs_high = np.array([
            grid_size[0]-1, grid_size[1]-1, 5, 3, 1
        ] * self.max_vehicles, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.vehicles = None  # Will hold vehicle states
        self.num_vehicles = None

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

        # Initialize vehicle states: [x, y, speed, direction, active]
        self.vehicles = np.zeros((self.max_vehicles, 5), dtype=np.float32)
        for i in range(self.num_vehicles):
            self.vehicles[i, 0] = np.random.randint(0, self.grid_size[0])  # x
            self.vehicles[i, 1] = np.random.randint(0, self.grid_size[1])  # y
            self.vehicles[i, 2] = np.random.randint(1, 3)  # speed
            self.vehicles[i, 3] = np.random.randint(0, 4)  # direction
            self.vehicles[i, 4] = 1  # active
        # Inactive vehicles remain at zero state
        return self._get_obs()

    def step(self, actions):
        """
        Take a step in the environment using the provided actions for each vehicle.
        actions: array-like of length max_vehicles (only first num_vehicles are used)
        Returns: obs, reward, done, info
        """
        rewards = np.zeros(self.max_vehicles, dtype=np.float32)
        done = False
        info = {}

        for i in range(self.num_vehicles):
            if self.vehicles[i, 4] == 0:
                continue  # Skip inactive vehicles
            action = actions[i]
            # 0: stay, 1: accelerate, 2: decelerate, 3: turn left, 4: turn right
            if action == 1:
                self.vehicles[i, 2] = min(self.vehicles[i, 2] + 1, 5)
            elif action == 2:
                self.vehicles[i, 2] = max(self.vehicles[i, 2] - 1, 0)
            elif action == 3:
                self.vehicles[i, 3] = (self.vehicles[i, 3] - 1) % 4
            elif action == 4:
                self.vehicles[i, 3] = (self.vehicles[i, 3] + 1) % 4
            # Move vehicle according to speed and direction
            speed = int(self.vehicles[i, 2])
            direction = int(self.vehicles[i, 3])
            if speed > 0:
                if direction == 0:  # North
                    self.vehicles[i, 0] = max(self.vehicles[i, 0] - speed, 0)
                elif direction == 1:  # East
                    self.vehicles[i, 1] = min(self.vehicles[i, 1] + speed, self.grid_size[1] - 1)
                elif direction == 2:  # South
                    self.vehicles[i, 0] = min(self.vehicles[i, 0] + speed, self.grid_size[0] - 1)
                elif direction == 3:  # West
                    self.vehicles[i, 1] = max(self.vehicles[i, 1] - speed, 0)
            # Simple reward: -1 per step, -10 for collision, +10 for reaching edge
            rewards[i] = -1
        # Check for collisions
        positions = set()
        for i in range(self.num_vehicles):
            if self.vehicles[i, 4] == 0:
                continue
            pos = (int(self.vehicles[i, 0]), int(self.vehicles[i, 1]))
            if pos in positions:
                rewards[i] -= 10  # Collision penalty
            else:
                positions.add(pos)
            # Reward for reaching edge
            if (self.vehicles[i, 0] == 0 or self.vehicles[i, 0] == self.grid_size[0]-1 or
                self.vehicles[i, 1] == 0 or self.vehicles[i, 1] == self.grid_size[1]-1):
                rewards[i] += 10
                self.vehicles[i, 4] = 0  # Deactivate vehicle
        # Done if all vehicles inactive
        if np.sum(self.vehicles[:, 4]) == 0:
            done = True
        return self._get_obs(), rewards, done, info

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
            if self.vehicles[i, 4] == 1:
                x, y = int(self.vehicles[i, 0]), int(self.vehicles[i, 1])
                grid[x, y] = str(i)
        print("\n".join([" ".join(row) for row in grid])) 