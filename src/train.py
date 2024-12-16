import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Create vectorized environment for parallel training
env = make_vec_env("BipedalWalker-v3", n_envs=4)

# Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100000)

# Save the model
model.save("bipedal_walker_ppo")

# Test the trained model
test_env = gym.make("BipedalWalker-v3")  # Explicitly create a test environment
state = test_env.reset()

for _ in range(1000):
    test_env.render()
    action, _ = model.predict(state)
    state, reward, done, info = test_env.step(action)
    if done:
        print("Episode finished!")
        break

test_env.close()  # Close the test environment
