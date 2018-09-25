from grid_world import Grid, standard_grid
from utils import print_policy,print_values
import numpy as np


theta = 1e-3
GAMMA = 0.9
ALL_ACTIONS = ('U','D','L','R')

def calculate_best_action_value(grid,V,s):

	best_action = 1e-6
	best_value = 1e-6
	grid.set_state(s)
	for action in ALL_ACTIONS:

		T = grid.get_transition_probs(action)
		e_value = 0
		e_reward = 0

		for (p,r,s_prime) in T:
			e_reward += p * r
			e_value += p * V[s_prime]

		v = e_reward + GAMMA * e_value

		if v > best_value:
			best_value = v
			best_action = action

	return best_value, best_action




def calculate_values(grid):
	V = {}
	# Initialize all values to 0
	for s in grid.all_states():
		V[s] = 0

	i = 0
	diff = 1
	while diff > theta:
		diff = 0
		for s in grid.non_terminal_states():
			old_value = V[s]
			best_value, best_action = calculate_best_action_value(grid,V,s)
			V[s] = best_value
			diff = max(np.abs(old_value-best_value), diff)
		i += 1
		print(i, diff) 

	return V

def initialize_random_policy(grid):
	policies = {}
	for s in grid.non_terminal_states():
		policies[s] = np.random.choice(ALL_ACTIONS)
	return policies

def calculate_optimum_policy(grid,V):
	policies = initialize_random_policy(grid)
	for s in policies.keys():
		grid.set_state(s)
		best_value, best_action = calculate_best_action_value(grid,V,s)
		policies[s] = best_action

	return policies



grid = standard_grid(0.5,-0.2)
print_values(grid.rewards,grid)

V = calculate_values(grid)

print_values(V,grid)
policies = calculate_optimum_policy(grid,V)
print_policy(policies,grid)


