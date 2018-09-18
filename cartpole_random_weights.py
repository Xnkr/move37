import gym
import numpy as np

def run_episode(env, parameters, print_observation = True):
	observation = env.reset()
	total_reward = 0
	for _ in range(200):
		if print_observation:
			print(observation)
		weighted_vector = np.matmul(parameters,observation)
		if weighted_vector < 0: # The pole is moving left
			action = 0
		else:
			action = 1
		observation, reward, done, info = env.step(action)
		total_reward += reward
		if done:
			break
	return total_reward

if __name__ == '__main__':
	
	env = gym.make('CartPole-v0')
	# Random weights
	parameters = np.random.rand(4) * 2 - 1 # [-1,1]

	print(run_episode(env,parameters))


