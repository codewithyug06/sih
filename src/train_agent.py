# train_agent.py
from train_env import TrainTrafficEnv
from stable_baselines3 import PPO
import os

# Create directories to save logs and models
models_dir = "models/PPO"
logdir = "logs"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Instantiate the environment
env = TrainTrafficEnv()
env.reset()

# Define the PPO model
# 'MlpPolicy' is a standard feedforward neural network policy.
# verbose=1 will print out training progress.
# tensorboard_log allows you to view learning curves with TensorBoard.
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

# Train the model for a specified number of timesteps
# We train in loops to save a checkpoint model periodically.
TIMESTEPS = 20000 
for i in range(1, 11): # Train for a total of 200,000 steps
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    # Save the model
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    print(f"Training loop {i} complete. Model saved.")

env.close()
print("Training finished.")