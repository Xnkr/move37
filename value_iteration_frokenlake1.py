import gym
import numpy as np

env_name  = 'FrozenLake8x8-v0'
env = gym.make(env_name).unwrapped

theta = 1e-20
gamma = 0.99


def best_action_value(env,s,V):

	best_action = float('-inf')
	best_value = float('-inf')

	for a in range(env.nA):
		T = env.P[s][a]
		e_reward = 0
		e_value = 0
		for (prob, s_prime, r, done) in T:
			e_reward += prob * r
			e_value += prob * V[s_prime]

		value = e_reward + gamma * e_value

		if value > best_value:
			best_value = value
			best_action = a

	return best_action, best_value



def calculate_values(env):
	V = {}

	for s in range(env.nS):
		V[s] = 0

	diff = 1
	i = 0
	while diff > theta:
		diff = 0
		for s in range(env.nS):
			old_value = V[s]
			best_action, best_value = best_action_value(env,s,V)
			V[s] = best_value

			diff = max(np.abs(best_value-old_value),diff)
		i += 1
		if i % 50 == 0:
			print(i,diff)

	return V


def evaluate_policy(env,V):
	print('Extracting policy')
	policy = {}
	for s in range(env.nS):
		policy[s] = 0

	for s in range(env.nS):

		best_action, best_value = best_action_value(env,s,V)
		policy[s] = best_action

	return policy

def run_episode(env,policy):
	print('Running episode')
	total = []
	n = 1000
	for i in range(n):
		t = 0
		obs = env.reset()
		step = 0
		while True:
			# env.render()
			obs, reward, done, info = env.step(policy[obs])
			t += gamma ** step * reward
			step += 1
			if done:
				break
		
		if i %100 == 0:
			print(i)

		total.append(t)

	print(np.mean(total))



V = calculate_values(env)
print(V)
optimal_policy = evaluate_policy(env,V)
print(optimal_policy)
run_episode(env,optimal_policy)



