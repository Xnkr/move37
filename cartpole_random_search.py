import gym 
import numpy as np
from cartpole_random_weights import run_episode
import matplotlib.pyplot as plt

def random_search(env):
	best_parameter = None
	best_reward = 0

	for episode in range(10000):
		parameters = np.random.rand(4) * 2 - 1
		reward = run_episode(env,parameters,print_observation = False)

		if reward > best_reward:
			best_reward = reward
			best_parameter = parameters

			if reward == 200:
				break

	return best_reward, episode

if __name__ == '__main__':
	
	env = gym.make('CartPole-v0')
	episodes = []
	for _ in range(1000):
		reward,episode = random_search(env)
		episodes.append(episode)
		if _ % 100 == 0:
			print(_)

	
	print(np.average(episodes))

	plt.hist(episodes,bins = 30)
	plt.show()




