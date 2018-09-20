''' Value iteration for Sports betting '''
import matplotlib.pyplot as plt

# State denotes the amount of capital we have
num_of_states = 100
states = [i for i in range(num_of_states+1)]

# Probability of home team winning is 40%
p_home_winning = 0.4


# Discount factor (No effect)
gamma = 1

# $100 is goal 
rewards = [0] * 101
rewards[100] = 1

values = [0] * 101
policies = [0] * 101

# For difference comparison (Convergence)
theta = 0.00000001 

def main():
	delta = 1
	while delta > theta:
		delta = 0

		for i in range(1,num_of_states):
			old_value = values[i]
			bellman_equation(i)
			difference = abs(old_value - values[i])
			delta = max(difference,delta)

	print(values)
	plt.scatter(states,values)
	plt.title('States vs Values')
	plt.show()

	plt.plot(states,policies)
	plt.title('States vs final policies')
	plt.show()


def bellman_equation(s):

	optimal_value = 0

	for bet in range(0,min(s,num_of_states-s)+1):
		# Bet is the action
		win = s + bet
		loss = s - bet

		sigma = p_home_winning * (rewards[win] + gamma * values[win]) + (1 - p_home_winning) * (rewards[loss] + gamma * values[loss])

		if sigma > optimal_value:
			optimal_value = sigma
			# Update value for the given state
			values[s] = sigma
			# Update action for the given state 
			policies[s] = bet



main()