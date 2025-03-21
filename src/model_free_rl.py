import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ModelFreeRL:
    def __init__(self, alpha=0.1, beta=5, theta = 0.2, epsilon=0.2):
        """
        Initializes the model-free reinforcement learning agent.

        Args:
            alpha (float): Learning rate.
            beta (float): Inverse temperature for softmax decision rule.
            theta (float): Choice perseveration parameter (bias toward repeating previous actions).
            epsilon (float): Probability of exploration (random action).
        """
        self.alpha = alpha 
        self.beta = beta   
        self.theta = theta  
        self.epsilon = epsilon
        self.explore = False
        self.prev_action = None   # Keeps track of last action for choice perseveration
        self.action_history = []
        self.reward_history = []
        self.q_value_history = []

        # Fixed probabilities for actions
        self.reward_probs = [0.3, 0.7]  
        self.q_table = np.zeros(2)

    def get_action_probabilities(self):
        """ 
        Calculate action probabilities using softmax function 
        Returns: 
            np.ndarray: Probability of choosing each action.
        """
        q_values = self.q_table.copy()
        if self.prev_action is not None and self.theta != 0:
            q_values[self.prev_action] += self.theta
    
        exp_values = np.exp(self.beta * q_values)

        return exp_values / np.sum(exp_values)

    
    def choose_action(self):
        """ Chooses an action based on softmax probabilities with epsilon-greedy exploration.

        Returns:
            int: Selected action (0 or 1).
        """
        # If the exploration is True and the random number is smaller than the exploration factor, we get a random action.
        if self.epsilon is not None and np.random.rand() < self.epsilon:
            action = np.random.randint(2)
            return action
        
        # Otherwise, choose action based on softmax probabilities 
        action_probs = self.get_action_probabilities() 

        return np.random.choice([0, 1], p=action_probs)

    
    def update(self, action, reward):
        """
        Updates Q-values based on the received reward using Temporal Difference (TD) learning.

        Args:
            action (int): Action taken.
            reward (int): Reward received (0 or 1).
        """
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.q_value_history.append(self.q_table.copy())

        rpe = reward -  self.q_table[action]
        self.q_table[action] += self.alpha * rpe
        

        self.prev_action = action


def simulate_participant_TD(trials=100, alpha=0.1, beta=5, theta=-1.2, epsilon=None, reward_probs=[0.3, 0.7]):
    """
    Simulate participants decisions in the two-armed bandit task
    Args:
        trials (int): Number of trials to simulate
        alpha (float): Learning rate for the agent
        beta (float): Inverse temperature for softmax action selection
        theta (float): Choice preservation

    Returns:
        actions (list): Sequence of chosen actions
        rewards (list): Sequence of rewards received
    """
   
    model = ModelFreeRL(alpha, beta, theta, epsilon)
    rewards = []
    actions = []

 
    for t in range(trials):
        action = model.choose_action()
        actions.append(action)
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
 




