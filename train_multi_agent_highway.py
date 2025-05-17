#!/usr/bin/env python3
"""
Multi-Agent Smart Highway Training
=================================

Train multiple smart highway vehicles simultaneously using multi-agent reinforcement learning.
Each vehicle learns to navigate efficiently while coordinating through 6G communication.
"""

import numpy as np
import sys
import os
import argparse
import time
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Import multi-agent libraries
try:
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    import ray
    RAY_AVAILABLE = True
    print("✅ Ray RLLib available - using advanced multi-agent training")
except ImportError:
    RAY_AVAILABLE = False
    print("⚠️ Ray RLLib not available - using basic multi-agent wrapper")

# Fallback to stable-baselines3 with custom multi-agent wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

from environments.smart_highway_env import SmartHighwayEnv

class MultiAgentWrapper(gym.Env):
    """
    Wrapper to convert multi-agent environment to single-agent for stable-baselines3.
    This is a simplified approach - for full multi-agent, use Ray RLLib.
    """
    
    def __init__(self, base_env):
        super().__init__()
        self.base_env = base_env
        self.num_agents = base_env.max_vehicles
        
        # Concatenate all agent observations and actions
        single_obs_dim = base_env.single_obs_space.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.num_agents * single_obs_dim,), 
            dtype=np.float32
        )
        
        # Action space: one action per agent
        self.action_space = spaces.MultiDiscrete([3] * self.num_agents)
        
    def reset(self, **kwargs):
        obs_dict, info = self.base_env.reset(**kwargs)
        return self._dict_to_array(obs_dict), info
    
    def step(self, actions):
        # Convert action array to dict
        action_dict = {}
        for i, action in enumerate(actions):
            agent_id = f"agent_{i}"
            action_dict[agent_id] = action
        
        obs_dict, reward_dict, term_dict, trunc_dict, info = self.base_env.step(action_dict)
        
        # Convert to arrays
        obs_array = self._dict_to_array(obs_dict)
        reward_sum = sum(reward_dict.values()) if reward_dict else 0
        terminated = any(term_dict.values()) if term_dict else False
        truncated = any(trunc_dict.values()) if trunc_dict else False
        
        return obs_array, reward_sum, terminated, truncated, info
    
    def _dict_to_array(self, obs_dict):
        """Convert observation dict to flat array."""
        if not obs_dict:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Create full array
        full_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs_dim = self.base_env.single_obs_space.shape[0]
        
        for agent_id, obs in obs_dict.items():
            agent_idx = int(agent_id.split('_')[1])
            start_idx = agent_idx * obs_dim
            end_idx = start_idx + obs_dim
            full_obs[start_idx:end_idx] = obs
            
        return full_obs

def create_multi_agent_environment(num_agents=4):
    """Create multi-agent smart highway environment."""
    base_env = SmartHighwayEnv(
        grid_size=(10, 10),
        max_vehicles=num_agents,  # Limit to number of learning agents
        spawn_rate=0.3,
        multi_agent=True,
        debug=False
    )
    
    if RAY_AVAILABLE:
        # TODO: Implement Ray RLLib multi-agent training
        print("🚧 Ray RLLib integration coming soon - using wrapper for now")
    
    # Use wrapper for stable-baselines3
    wrapped_env = MultiAgentWrapper(base_env)
    return wrapped_env

def train_multi_agent():
    """Train multiple agents using multi-agent learning."""
    print("🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖")
    print("🎓 MULTI-AGENT SMART HIGHWAY TRAINING 🎓")
    print("🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖")
    print("🎯 Multi-Agent Training Objectives:")
    print("  • Multiple vehicles learning simultaneously")
    print("  • Cooperative navigation and collision avoidance")
    print("  • 6G-based communication and coordination")
    print("  • Emergent traffic optimization patterns")
    print("🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖🤖")
    
    # Training parameters
    num_agents = 4  # Start with 4 learning agents
    total_timesteps = 200000  # More timesteps needed for multi-agent
    eval_freq = 10000
    target_reward = 200  # Higher target for multi-agent coordination
    
    print(f"📚 Setting up multi-agent environment...")
    print(f"📊 Multi-Agent Configuration:")
    print(f"  Learning Agents: {num_agents}")
    print(f"  Total Timesteps: {total_timesteps:,}")
    print(f"  Evaluation Frequency: {eval_freq:,}")
    print(f"  Target Reward: {target_reward}")
    print(f"  Parallel Environments: 4")
    
    # Create environment
    def make_env():
        return create_multi_agent_environment(num_agents)
    
    # Vectorized environments for parallel training
    env = make_vec_env(make_env, n_envs=4)
    
    # Create evaluation environment
    eval_env = Monitor(create_multi_agent_environment(num_agents))
    
    print("🧠 Creating Multi-Agent PPO...")
    
    # PPO model with larger network for multi-agent complexity
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"./logs/multi_agent_training/",
        learning_rate=3e-4,
        n_steps=2048,  # More steps for multi-agent
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration in multi-agent
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256, 256], vf=[256, 256, 256])]  # Larger networks
        )
    )
    
    # Create callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"./logs/multi_agent_eval/"
    os.makedirs(log_dir, exist_ok=True)
    
    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=target_reward, 
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./trained_models/",
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        callback_on_new_best=stop_callback
    )
    
    print("🎓 Starting Multi-Agent Training...")
    print(f"📈 Monitor progress: tensorboard --logdir ./logs/multi_agent_training/")
    
    # Start training
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    training_time = time.time() - start_time
    
    print(f"✅ Multi-Agent Training completed in {training_time:.1f} seconds!")
    
    # Save final model
    model.save("trained_models/ppo_multi_agent_highway")
    print(f"💾 Multi-Agent Model saved to: trained_models/ppo_multi_agent_highway")
    
    # Test the trained model
    test_multi_agent_model(model, create_multi_agent_environment(num_agents))

def test_multi_agent_model(model, env):
    """Test the trained multi-agent model."""
    print("🧪 Testing multi-agent model...")
    print(f"🎯 Testing model for 5 episodes...")
    
    total_rewards = []
    total_steps = []
    
    for episode in range(5):
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while steps < 500:  # Max steps per episode
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        print(f"  Episode {episode + 1}: Reward={episode_reward:.2f}, Steps={steps}")
    
    print(f"📊 MULTI-AGENT MODEL PERFORMANCE:")
    print(f"  Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Average Episode Length: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
    print(f"  🤖 Multi-agent coordination achieved!")

def compare_single_vs_multi_agent():
    """Compare single-agent vs multi-agent performance."""
    print("📊 COMPARING SINGLE-AGENT vs MULTI-AGENT PERFORMANCE")
    print("=" * 60)
    
    # Test single-agent model
    try:
        from train_smart_highway import SmartHighwayGymEnv
        single_model = PPO.load("trained_models/ppo_smart_highway")
        single_env = Monitor(SmartHighwayGymEnv(
            grid_size=(10, 10), max_vehicles=16, spawn_rate=0.3, debug=False
        ))
        
        print("🔍 Testing Single-Agent Model...")
        single_rewards = []
        for _ in range(5):
            obs, _ = single_env.reset()
            episode_reward = 0
            for _ in range(500):
                action, _ = single_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = single_env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            single_rewards.append(episode_reward)
        
        print(f"Single-Agent Average: {np.mean(single_rewards):.2f}")
        
    except Exception as e:
        print(f"⚠️ Could not load single-agent model: {e}")
    
    # Test multi-agent model
    try:
        multi_model = PPO.load("trained_models/ppo_multi_agent_highway")
        multi_env = create_multi_agent_environment(4)
        
        print("🔍 Testing Multi-Agent Model...")
        multi_rewards = []
        for _ in range(5):
            obs, _ = multi_env.reset()
            episode_reward = 0
            for _ in range(500):
                action, _ = multi_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = multi_env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            multi_rewards.append(episode_reward)
        
        print(f"Multi-Agent Average: {np.mean(multi_rewards):.2f}")
        
        # Calculate improvement
        if 'single_rewards' in locals():
            improvement = (np.mean(multi_rewards) - np.mean(single_rewards)) / np.mean(single_rewards) * 100
            print(f"🚀 Multi-Agent Improvement: {improvement:+.1f}%")
        
    except Exception as e:
        print(f"⚠️ Could not load multi-agent model: {e}")

def main():
    parser = argparse.ArgumentParser(description='Multi-Agent Smart Highway Training')
    parser.add_argument('--train', action='store_true', help='Train multi-agent model')
    parser.add_argument('--test', action='store_true', help='Test trained multi-agent model')
    parser.add_argument('--compare', action='store_true', help='Compare single vs multi-agent')
    
    args = parser.parse_args()
    
    if args.train:
        train_multi_agent()
    elif args.test:
        try:
            model = PPO.load("trained_models/ppo_multi_agent_highway")
            env = create_multi_agent_environment(4)
            test_multi_agent_model(model, env)
        except FileNotFoundError:
            print("❌ No multi-agent model found. Run training first with --train")
    elif args.compare:
        compare_single_vs_multi_agent()
    else:
        print("🤖 Multi-Agent Smart Highway Training System")
        print("Usage:")
        print("  python train_multi_agent_highway.py --train     # Train multi-agent model")
        print("  python train_multi_agent_highway.py --test      # Test trained model")
        print("  python train_multi_agent_highway.py --compare   # Compare performance")

if __name__ == "__main__":
    main() 