import gym
import numpy as np
from cartpole_random_weights import run_episode
import matplotlib.pyplot as plt

def hill_climbing(env):

	best_params = None
	best_reward = 0

	noise_scaling = 0.1
	params = np.random.rand(4) * 2 - 1

	for episode in range(1000):
		new_params = params + (np.random.rand(4) * 2 - 1) * noise_scaling
		reward = run_episode(env, params, print_observation = False)
		if reward > best_reward:
			best_reward = reward
			params = new_params
			if reward == 200:
				break

	return episode

def hill_climbing_increase_noise(env):

	best_params = None
	best_reward = 0

	noise_scaling = 0.1
	params = np.random.rand(4) * 2 - 1

	for episode in range(1000):
		new_params = params + (np.random.rand(4) * 2 - 1) * noise_scaling
		reward = run_episode(env, params, print_observation = False)
		if reward > best_reward:
			best_reward = reward
			params = new_params
			if reward == 200:
				break
		else:
			# Agent going worse increase noise factor
			noise_scaling += 0.01

	return episode

def hill_climbing_cummulative_reward(env):

	best_params = None
	best_reward = 0

	noise_scaling = 0.1
	params = np.random.rand(4) * 2 - 1
	episodes_per_update = 5
	for episode in range(1000):
		new_params = params + (np.random.rand(4) * 2 - 1) * noise_scaling	
		reward = 0
		for _ in range(episodes_per_update):
			r = run_episode(env, params, print_observation = False)
			reward += r
		if reward > best_reward:
			best_reward = reward
			params = new_params
			if reward == 200:
				break

	return episode


if __name__ == '__main__':
	env = gym.make('CartPole-v0')

	episodes = []

	for _ in range(1000):
		# episodes.append(hill_climbing(env))
		# episodes.append(hill_climbing_increase_noise(env))
		# episodes.append(hill_climbing_cummulative_reward(env))

		if _ % 100 == 0:
			print(_)

	print(np.average(episodes))

	plt.hist(episodes,bins = 30)
	plt.show()

