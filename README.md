# City Traffic Simulation with Multi-Agent RL and 6G Communication

## Overview
This project simulates a city traffic grid with autonomous vehicles, focusing on advanced features for research and development in intelligent transportation systems. The simulation includes:

- **Multi-agent traffic environment**: Vehicles move on a grid, follow routes, and interact at intersections.
- **Reservation-based intersection management**: Vehicles must request and receive time slots to safely cross intersections, preventing collisions.
- **6G V2V and V2I communication**: All vehicle-to-vehicle and vehicle-to-infrastructure messages are sent over a simulated 6G network, with configurable delay and reliability.
- **Multi-agent reinforcement learning (RL)**: Supports both single-agent (ego vs. scripted) and multi-agent (parameter sharing or per-agent) RL training, compatible with Stable Baselines3 and PettingZoo-style APIs.
- **Visualization and logging**: Includes Matplotlib-based grid visualization, training metric plots, and detailed logging/debugging options.

## Installation & Setup
1. **Clone the repository** (or open the folder in Cursor):
   ```bash
   git clone <your-repo-url>
   cd simulation_city_grid
   ```
2. **Install dependencies** (Python 3.8+ recommended):
   ```bash
   pip install -r requirements.txt
   # Or, for a minimal setup:
   pip install gym numpy matplotlib stable-baselines3 torch
   ```
   (For TensorBoard logging: `pip install tensorboard`)

3. **(Optional) Install Cursor**
   - Download from [cursor.so](https://www.cursor.so/) and follow the setup instructions.

## Running the Simulation
- **Unit tests**:
  ```bash
  python test_simulation.py
  ```
- **Visualize a random episode**:
  ```python
  from city_traffic_env import CityTrafficEnv
  env = CityTrafficEnv(grid_size=(10,10), max_vehicles=5, debug=True)
  obs = env.reset()
  for _ in range(20):
      env.render()  # Shows the grid and vehicles
      obs, reward, done, info = env.step(0)  # All vehicles stay
      if done:
          break
  ```

## Training Agents
- **Single-agent RL (ego vehicle, others scripted):**
  ```bash
  python train_city_traffic.py
  ```
  This will train a PPO agent using Stable Baselines3, log metrics, and plot training progress.

- **Multi-agent RL:**
  - Set `multi_agent=True` in `CityTrafficEnv`.
  - Use parameter sharing (one policy for all agents) or train agents one at a time.
  - For advanced multi-agent RL, consider PettingZoo wrappers or MARL libraries.

## Project Structure
- `city_traffic_env.py` — Main environment class (Gym-compatible, multi-agent ready, visualization, debug).
- `intersection_manager.py` — Reservation-based intersection logic.
- `comm_module.py` — 6G V2V/V2I communication simulation (delay, drop, broadcast).
- `train_city_traffic.py` — Training script using Stable Baselines3 PPO, with logging and plotting.
- `test_simulation.py` — Unit tests for environment and intersection logic.
- `README.md` — This documentation.

## Using Cursor for Iterative Development
Cursor is an AI-powered code editor that helps you rapidly prototype, debug, and extend this simulation. To use Cursor effectively:

- **Open the project folder in Cursor.**
- **Use the provided prompt pack** to:
  - Add new features (e.g., new intersection types, more realistic vehicle dynamics).
  - Refactor or debug code with AI assistance.
  - Run and inspect tests, visualize outputs, and iterate quickly.
- **Leverage inline code suggestions and chat** to ask for explanations, generate docstrings, or optimize RL training.

Cursor is especially useful for:
- Rapid prototyping of new RL environments or agent behaviors.
- Debugging complex multi-agent interactions.
- Visualizing and iterating on simulation logic in real time.

---

**Happy simulating and experimenting!**

For questions or contributions, open an issue or PR, or use Cursor's chat to get help and suggestions. 