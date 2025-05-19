#!/usr/bin/env python3
"""
Multi-Agent Smart Highway Training WITHOUT 6G Communication
===========================================================

Train multiple smart highway vehicles simultaneously WITHOUT 6G communication.
This is for comparison purposes to demonstrate the benefit of 6G coordination.
"""

import numpy as np
import sys
import os
import argparse
import time
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

from environments.smart_highway_env import SmartHighwayEnv

class MultiAgentNoCommWrapper(gym.Env):
    """
    Wrapper to convert multi-agent environment to single-agent for stable-baselines3.
    This version DISABLES 6G communication to test performance without coordination.
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

def create_multi_agent_no_comm_environment(num_agents=4):
    """Create multi-agent smart highway environment WITHOUT 6G communication."""
    base_env = SmartHighwayEnv(
        grid_size=(10, 10),
        max_vehicles=num_agents,  # Limit to number of learning agents
        spawn_rate=0.3,
        multi_agent=True,
        debug=False,
        enable_6g=False  # DISABLE 6G communication for comparison
    )
    
    # Use wrapper for stable-baselines3
    wrapped_env = MultiAgentNoCommWrapper(base_env)
    return wrapped_env

def train_multi_agent_no_comm():
    """Train multiple agents WITHOUT 6G communication."""
    print("🚫📡🚫📡🚫📡🚫📡🚫📡🚫📡🚫📡🚫📡🚫📡🚫📡🚫📡🚫📡")
    print("🎓 MULTI-AGENT TRAINING WITHOUT 6G COMMUNICATION 🎓")
    print("🚫📡🚫📡🚫📡🚫📡🚫📡🚫📡🚫📡🚫📡🚫📡🚫📡🚫📡🚫📡")
    print("🎯 Training Objectives (NO 6G):")
    print("  • Multiple vehicles learning simultaneously")
    print("  • NO 6G communication or coordination")
    print("  • Pure individual learning and collision avoidance")
    print("  • Baseline for comparison with 6G systems")
    print("🚫📡🚫📡🚫📡🚫📡🚫📡🚫📡🚫📡🚫📡🚫📡🚫📡🚫📡🚫📡")
    
    # Training parameters
    num_agents = 4  # Same as 6G version for fair comparison
    total_timesteps = 200000  # Same training time
    eval_freq = 10000
    target_reward = 150  # Lower target expected without 6G coordination
    
    print(f"📚 Setting up multi-agent environment WITHOUT 6G...")
    print(f"📊 Multi-Agent Configuration (NO COMMUNICATION):")
    print(f"  Learning Agents: {num_agents}")
    print(f"  Total Timesteps: {total_timesteps:,}")
    print(f"  Evaluation Frequency: {eval_freq:,}")
    print(f"  Target Reward: {target_reward}")
    print(f"  6G Communication: DISABLED")
    print(f"  Parallel Environments: 4")
    
    # Create environment
    def make_env():
        return create_multi_agent_no_comm_environment(num_agents)
    
    # Vectorized environments for parallel training
    env = make_vec_env(make_env, n_envs=4)
    
    # Create evaluation environment
    eval_env = Monitor(create_multi_agent_no_comm_environment(num_agents))
    
    print("🧠 Creating Multi-Agent PPO (NO 6G)...")
    
    # PPO model with larger network for multi-agent complexity
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"./logs/multi_agent_no_comm_training/",
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
    log_dir = f"./logs/multi_agent_no_comm_eval/"
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
    
    print("🎓 Starting Multi-Agent Training (NO 6G)...")
    print(f"📈 Monitor progress: tensorboard --logdir ./logs/multi_agent_no_comm_training/")
    
    # Start training
    start_time = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    training_time = time.time() - start_time
    
    print(f"✅ Multi-Agent Training (NO 6G) completed in {training_time:.1f} seconds!")
    
    # Save final model
    model.save("trained_models/ppo_multi_agent_no_comm")
    print(f"💾 Multi-Agent No-Comm Model saved to: trained_models/ppo_multi_agent_no_comm")
    
    # Test the trained model
    test_multi_agent_no_comm_model(model, create_multi_agent_no_comm_environment(num_agents))

def test_multi_agent_no_comm_model(model, env):
    """Test the trained multi-agent model without 6G."""
    print("🧪 Testing multi-agent model (NO 6G)...")
    print(f"🎯 Testing model for 5 episodes...")
    
    total_rewards = []
    total_steps = []
    total_collisions = []
    
    for episode in range(5):
        obs, _ = env.reset()
        episode_reward = 0
        episode_collisions = 0
        steps = 0
        
        while steps < 500:  # Max steps per episode
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_collisions += len(info.get('actual_collisions', []))
            steps += 1
            
            if terminated or truncated:
                break
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        total_collisions.append(episode_collisions)
        print(f"  Episode {episode + 1}: Reward={episode_reward:.2f}, Steps={steps}, Collisions={episode_collisions}")
    
    print(f"📊 MULTI-AGENT MODEL (NO 6G) PERFORMANCE:")
    print(f"  Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"  Average Episode Length: {np.mean(total_steps):.1f} ± {np.std(total_steps):.1f}")
    print(f"  Average Collisions: {np.mean(total_collisions):.2f} ± {np.std(total_collisions):.2f}")
    print(f"  🚫 No 6G coordination - pure individual learning!")

def main():
    parser = argparse.ArgumentParser(description='Multi-Agent Smart Highway Training WITHOUT 6G')
    parser.add_argument('--train', action='store_true', help='Train multi-agent model (no 6G)')
    parser.add_argument('--test', action='store_true', help='Test trained multi-agent model (no 6G)')
    
    args = parser.parse_args()
    
    if args.train:
        train_multi_agent_no_comm()
    elif args.test:
        try:
            model = PPO.load("trained_models/ppo_multi_agent_no_comm")
            env = create_multi_agent_no_comm_environment(4)
            test_multi_agent_no_comm_model(model, env)
        except FileNotFoundError:
            print("❌ No multi-agent no-comm model found. Run training first with --train")
    else:
        print("🚫📡 Multi-Agent Smart Highway Training System (NO 6G COMMUNICATION)")
        print("Usage:")
        print("  python train_multi_agent_no_comm.py --train     # Train multi-agent model (no 6G)")
        print("  python train_multi_agent_no_comm.py --test      # Test trained model (no 6G)")

if __name__ == "__main__":
    main() 