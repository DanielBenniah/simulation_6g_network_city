# Smart Highway Training & Logs Guide 🚗📊

## 📁 **Logs Folder Structure**

When you run training, the `logs/` folder will contain:

```
logs/
├── smart_highway_training/          # TensorBoard training logs
│   ├── smart_highway_ppo_1/         # Training run 1
│   │   ├── events.out.tfevents.*    # TensorBoard event files
│   │   └── ...
│   ├── smart_highway_ppo_2/         # Training run 2 (if you retrain)
│   └── ...
└── smart_highway_eval/              # Evaluation logs
    ├── evaluations.npz              # Numerical evaluation results
    ├── monitor.csv                  # Episode statistics
    └── ...
```

## 📈 **What Each Log Contains**

### **1. TensorBoard Training Logs** (`smart_highway_training/`)
- **Reward progression** over training steps
- **Policy loss** and **value loss** during learning
- **Learning rate** adjustments
- **Episode length** statistics
- **Entropy** (exploration vs exploitation balance)
- **6G communication efficiency** metrics

### **2. Evaluation Logs** (`smart_highway_eval/`)
- **Best model checkpoints** (automatically saved)
- **Episode rewards** during evaluation
- **Collision statistics** (actual vs prevented)
- **Training progress** measurements

## 🔧 **How to Use the Results**

### **Method 1: TensorBoard (Recommended)**
```bash
# Install TensorBoard if not already installed
pip install tensorboard

# View training progress in real-time or after training
tensorboard --logdir logs/smart_highway_training/

# Then open your browser to: http://localhost:6006
```

**What you'll see in TensorBoard:**
- 📈 **Reward curves** - Is the agent learning?
- 📉 **Loss curves** - Is training stable?
- 🎯 **Episode length** - Are episodes getting longer (better performance)?
- 🛡️ **Custom metrics** - 6G prevention rates, collision counts

### **Method 2: Python Analysis**
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load evaluation results
eval_data = np.load('logs/smart_highway_eval/evaluations.npz')
rewards = eval_data['results']
timesteps = eval_data['timesteps']

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(timesteps, rewards.mean(axis=1))
plt.fill_between(timesteps, 
                 rewards.mean(axis=1) - rewards.std(axis=1),
                 rewards.mean(axis=1) + rewards.std(axis=1), 
                 alpha=0.3)
plt.xlabel('Training Steps')
plt.ylabel('Average Reward')
plt.title('Smart Highway Agent Learning Progress')
plt.grid(True)
plt.show()

# Load episode statistics
monitor_data = pd.read_csv('logs/smart_highway_eval/monitor.csv')
print("Training Statistics:")
print(f"Best Episode Reward: {monitor_data['r'].max():.2f}")
print(f"Average Episode Length: {monitor_data['l'].mean():.1f}")
```

### **Method 3: Compare Training Runs**
```bash
# Compare multiple training sessions
tensorboard --logdir logs/smart_highway_training/ --reload_interval=1
```

## 🎯 **Key Metrics to Monitor**

### **During Training (TensorBoard):**
1. **📈 Reward Trend**
   - Should generally increase over time
   - Plateaus indicate convergence
   - Drops might indicate overfitting

2. **🎲 Policy Loss**
   - Should decrease and stabilize
   - Spikes indicate learning instability

3. **📊 Episode Length**
   - Longer episodes = better performance
   - Consistent length = stable policy

4. **🛡️ 6G Metrics** (Custom)
   - Collision prevention rate
   - Communication efficiency
   - Intersection management success

### **After Training (Evaluation):**
1. **Final Performance**
   - Average reward per episode
   - Collision rate (should be low)
   - Journey completion rate

2. **Consistency**
   - Standard deviation of rewards
   - Performance across different scenarios

## 🚀 **Training Commands & Log Generation**

### **Start Training** (Generates logs)
```bash
python train_smart_highway.py --train
```

### **Monitor Live Training**
```bash
# In a separate terminal while training:
tensorboard --logdir logs/smart_highway_training/
```

### **Evaluate Trained Model**
```bash
python train_smart_highway.py --test
```

### **Compare Random vs Trained**
```bash
python train_smart_highway.py --compare
```

## 📊 **Expected Results**

### **Good Training Signs:**
- ✅ Reward increases from ~10 to 150+ over time
- ✅ Collision rate decreases to <1 per episode
- ✅ 6G prevention rate >80%
- ✅ Episode length increases (agent survives longer)
- ✅ Policy loss stabilizes

### **Warning Signs:**
- ⚠️ Reward oscillates wildly or decreases
- ⚠️ High collision rate (>5 per episode)
- ⚠️ Very short episodes consistently
- ⚠️ Policy loss keeps increasing

## 🔍 **Troubleshooting with Logs**

### **Poor Performance?**
1. Check TensorBoard reward curve - is it increasing?
2. Look at episode length - are episodes too short?
3. Check collision metrics - too many actual collisions?

### **Training Instability?**
1. Monitor policy loss - should decrease smoothly
2. Check learning rate - might need adjustment
3. Look at entropy - balance exploration vs exploitation

### **Model Not Learning?**
1. Verify reward signal is meaningful
2. Check if environment is too complex
3. Consider adjusting hyperparameters

## 💾 **Best Model Storage**

The training automatically saves:
- **Best model** → `trained_models/best_model.zip`
- **Final model** → `trained_models/ppo_smart_highway.zip`
- **Evaluation checkpoints** → `logs/smart_highway_eval/`

## 🎯 **Next Steps After Training**

1. **Analyze Results**: Use TensorBoard and Python analysis
2. **Test Performance**: Run evaluation and comparison scripts
3. **Visualize**: Use the trained model in the visualizer
4. **Iterate**: Adjust hyperparameters based on log analysis
5. **Deploy**: Use the best model for real simulations

---

**Pro Tip:** Keep TensorBoard running during training to monitor progress in real-time and catch issues early! 