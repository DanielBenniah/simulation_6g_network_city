import unittest
import numpy as np
from city_traffic_env import CityTrafficEnv
from intersection_manager import IntersectionManager

class TestIntersectionManager(unittest.TestCase):
    def test_conflicting_reservations(self):
        im = IntersectionManager()
        # Vehicle 1: North to South, arrives at t=1
        granted1, slot1 = im.request_reservation(1, arrival_time=1.0, duration=1.0, path=(0,2))
        self.assertTrue(granted1)
        # Vehicle 2: East to West, arrives at t=1 (conflict)
        granted2, slot2 = im.request_reservation(2, arrival_time=1.0, duration=1.0, path=(1,3))
        self.assertFalse(granted2)
        # Vehicle 3: North to South, arrives at t=2 (no conflict)
        granted3, slot3 = im.request_reservation(3, arrival_time=2.0, duration=1.0, path=(0,2))
        self.assertTrue(granted3)

    def test_single_vehicle_approved(self):
        im = IntersectionManager()
        granted, slot = im.request_reservation(1, arrival_time=0.0, duration=1.0, path=(0,2))
        self.assertTrue(granted)

class TestCityTrafficEnv(unittest.TestCase):
    def test_reset_no_out_of_bounds(self):
        env = CityTrafficEnv(grid_size=(5,5), max_vehicles=3, multi_agent=False)
        obs = env.reset()
        for i in range(env.num_vehicles):
            x, y = env.vehicles[i, 0], env.vehicles[i, 1]
            self.assertGreaterEqual(x, 0)
            self.assertLess(x, env.grid_size[0])
            self.assertGreaterEqual(y, 0)
            self.assertLess(y, env.grid_size[1])

    def test_step_no_vehicles(self):
        env = CityTrafficEnv(grid_size=(5,5), max_vehicles=0, multi_agent=False)
        obs = env.reset(num_vehicles=0)
        obs2, reward, done, info = env.step(0)
        np.testing.assert_array_equal(obs, obs2)
        self.assertEqual(reward, 0)
        self.assertFalse(done)

    def test_step_single_vehicle(self):
        env = CityTrafficEnv(grid_size=(5,5), max_vehicles=1, multi_agent=False)
        obs = env.reset(num_vehicles=1)
        obs2, reward, done, info = env.step(0)  # Stay action
        # Vehicle should still be in bounds
        x, y = env.vehicles[0, 0], env.vehicles[0, 1]
        self.assertGreaterEqual(x, 0)
        self.assertLess(x, env.grid_size[0])
        self.assertGreaterEqual(y, 0)
        self.assertLess(y, env.grid_size[1])

if __name__ == '__main__':
    unittest.main() 