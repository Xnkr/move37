import gym
import numpy as np


def extract_policy(env,V, gamma = 0.9):
    policy = np.zeros(env.nS)

    for state in range(env.nS):
        q_sa = np.zeros(env.nA)
        for action in range(env.nA):
            T = env.P[state][action]
            for prob, s_, r_, _ in T:
                q_sa[action] += prob * (r_ + gamma * V[s_])
        policy[state] = np.argmax(q_sa)
    return policy
    
def compute_value_for_policy(env,policy,gamma=0.9):
    print(f'Gamma: {gamma}')
    V = np.zeros(env.nS)
    theta = 1e-10
    
    while True:
        prev_value = np.copy(V)
        for state in range(env.nS):
            action = policy[state]
            q_sa = []
            T = env.P[state][action]
            for prob, s_, r_, _ in T:
                q_sa.append(prob*(r_ + gamma * V[s_]))
            V[state] = sum(q_sa)

        if np.sum(np.fabs(prev_value - V)) < theta:
            break

    return V


def policy_iteration(env,gamma=0.9):

    policy = np.random.choice(env.nA, size=(env.nS))
    max_iteration = 100000

    print(f'Random policy {policy}')

    for _ in range(max_iteration):
        old_policy_value = compute_value_for_policy(env,policy,gamma)
        new_policy = extract_policy(env,old_policy_value,gamma)

        if np.all(policy == new_policy):
            print('Policy Converged')
            break

        if _ % 1000 == 0:
            print(f'Policy iteration {_}')

        policy = new_policy
    return policy


def run_episode(env,policy,render=False,n=1000,gamma=0.9):
    score = []
    for i in range(n):
        obs = env.reset()
        total_reward = 0
        step_idx = 0
        while True:
            if render:
                env.render()
            obs, reward, done, _ = env.step(int(policy[obs]))
            total_reward += gamma ** step_idx * reward
            if done:
                break

        if i % 100 == 0:
            print(f'Episode {i} {total_reward}')

        score.append(total_reward)
    print(f'Discounted Reward mean {np.mean(score)}')    

gamma = 0.9

env = gym.make('FrozenLake8x8-v0')
env = env.unwrapped

policy = policy_iteration(env,gamma)
print(f'Optimal policy {policy}')

run_episode(env,policy,render=True,n=1,gamma=gamma)
run_episode(env,policy,render=False,n=1000,gamma=gamma)