import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ModelFreeRL:
    def __init__(self, alpha=0.1, beta=5, theta = 0.2, gamma=0.9, epsilon=0.2):

        self.alpha = alpha # Learning rate of stage 1
        self.beta = beta    # Inverse temperature for softmax
        self.gamma = gamma  # Discount factor
        self.theta = theta  # Choice perseveration
        self.epsilon = epsilon # Exploration probability
        self.explore = False
        self.prev_action = None # Needed for choice perservation
        self.action_history = []
        self.reward_history = []
        self.q_value_history = []

        # Fixed probabilities for actions
        self.reward_probs = [0.3, 0.7]  
        self.q_table = np.zeros(2)

    def get_action_probabilities(self):

        q_values = self.q_table.copy()
        if self.prev_action is not None and self.theta != 0:
            q_values[self.prev_action] += self.theta
        
        """ Calculate action probabilities using softmax function """
        exp_values = np.exp(self.beta * q_values)

        return exp_values / np.sum(exp_values)

    
    def choose_action(self):
        """ Select an action based on predefined probabilities with including exploration """

        # If the exploration is True and the random number is smaller than the exploration factor, we get a random action.
        if self.epsilon is not None and np.random.rand() < self.epsilon:
            action = np.random.randint(2)
            return action
            
        action_probs = self.get_action_probabilities()

        # Ensure probabilities remain valid
        #action_probs = np.maximum(action_probs, 0)  # Ensure no negative values

        #action_probs /= np.sum(action_probs)  # Normalize to sum to 1
        
        return np.random.choice([0, 1], p=action_probs)

    
    def update(self, action, reward):
        """ Update the values using TD learning """


        self.action_history.append(action)
        self.reward_history.append(reward)
        self.q_value_history.append(self.q_table.copy())

        rpe = reward -  self.q_table[action]
        self.q_table[action] += self.alpha * rpe
        

        self.prev_action = action


def simulate_participant_TD(trials=100, alpha=0.1, beta=5, gamma=0.9, theta=0.2, epsilon=None, reward_probs=[0.3, 0.7]):
    """
    Simulate participants decisions in the two-armed bandit task
    Args:
        trials (int): Number of trials to simulate
        alpha (float): Learning rate for the agent
        beta (float): Inverse temperature for softmax action selection
        gamma (float): Discount factor
        theta (float): Choice preservation

    Returns:
        actions (list): Sequence of chosen actions
        rewards (list): Sequence of rewards received
    """
   
    model = ModelFreeRL(alpha, beta, gamma, theta, epsilon)
    rewards = []
    actions = []

 
    for t in range(trials):
        action = model.choose_action()
        actions.append(action)
        #print(action)

        reward = np.random.choice([0, 1], p=[1 - reward_probs[action], reward_probs[action]]) 
        rewards.append(reward)
  

        model.update(action, reward)

    return actions, rewards, model
    
def plot_simulated_behavior_TD(actions, rewards):
    """
    Plots reward outcomes (0 or 1) and colors them based on chosen action.
    
    Args:
        actions (list): Sequence of chosen actions (0 or 1)
        rewards (list): Sequence of rewards received (0 or 1)
    """

    trials = np.arange(len(actions))
    colors = ['blue' if action == 0 else 'red' for action in actions]

    plt.figure(figsize=(10, 5))
     
     # Scatter plot where the reward is 0 or 1, colored by action
    plt.scatter(trials, rewards, c=colors, edgecolors='black', marker='o', s=100, alpha=0.75)
    
    plt.xlabel('Trial')
    plt.ylabel('Reward (0 or 1)')
    plt.title('Simulated Rewards Over Trials (Colored by Choice)')
    
    # Custom legend for choices
    plt.scatter([], [], color='blue', label='Choice 0')
    plt.scatter([], [], color='red', label='Choice 1')
    plt.legend()

    plt.show()
 




