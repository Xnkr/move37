import gym
import numpy as np
import matplotlib.pyplot as plt

THETA = 1e-20
GAMMA = 1.0

def calculate_values(env):
	V = np.zeros(env.nS)

	max_iterations = 100000
	for i in range(max_iterations):
		old_value = np.copy(V)
		for state in range(env.nS):
			op_q_sa = []
			for action in range(env.nA):
				T = env.P[state][action]
				q_sa = []
				for prob, s_prime, reward, _ in T:
					q_sa.append(prob * (reward + GAMMA * old_value[s_prime]))

				op_q_sa.append(sum(q_sa))
			V[state] = max(op_q_sa)

		if np.sum(np.fabs(old_value - V)) < THETA:
			print(f'Converged at {i}')
			break
	return V


def extract_policy(env,V):
	
	op_policy = np.zeros(env.nS)

	for state in range(env.nS):
		op_q_sa = np.zeros(env.nA)
		for action in range(env.nA):
			T = env.P[state][action]
			q_sa = []
			for p, s_, r_, _ in T:
				q_sa.append(p * (r_ + GAMMA * V[s_]))
			op_q_sa[action] = sum(q_sa)
				

		op_policy[state] = np.argmax(op_q_sa)

	return op_policy



def run_episode(env,policy, render = False, GAMMA = 0.9):
	scores = []
	n = 1000
	for i in range(n):
		step_idx = 0
		obs = env.reset()
		total_reward = 0
		while True:
			obs, r_, done, _ = env.step(int(policy[obs]))

			total_reward += GAMMA ** step_idx * r_
			step_idx += 1

			if done:
				break
		if i % 100 == 0: 
			print(f'Episode {i}')
		scores.append(total_reward)

	print(np.mean(scores))
	plt.plot(scores)
	plt.show()


if __name__ == '__main__':
	env_name  = 'FrozenLake8x8-v0'
	env = gym.make(env_name).unwrapped

	V = calculate_values(env)
	print(V)

	op_policy = extract_policy(env,V)
	print(op_policy)

	run_episode(env, op_policy)

