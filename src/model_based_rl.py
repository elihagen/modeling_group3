import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ModelBasedRL:
    def __init__(self, alpha = 0.1, beta = 5, gamma = 0.9, theta = 0.2):
        """
        Initialize the model-based RL model
        We consider:
        - Q-table is of size (2, 2) - two states and two actions
        - transition_probs: gives the probabilities of outcomes for each action.
        
        Args: 
            alpha: Learning rate
            beta: Inverse temperature for softmax (exploration-exploitation)
            gamma: Discount factor 
            theta: Stickiness parameter (choice perseveration)
        """
        self.alpha = alpha  # Learning rate
        self.beta = beta  # exploration-exploitation balance (inverse temperature)
        self.gamma = gamma  # Discount factor
        self.theta = theta # choice preservation
        self.q_table = np.zeros((2,2)) # two bandits with 2 possible Q-values
        self.prev_choice = None # needed for choice preservation
        self.transition_probs = {
            0: (0.5, 0.5), 
            1: (0.5, 0.5)
        }

    def get_action_probabilities(self, state):
        """
        Compute softmax probs for actions in the state
        Args:
            state (int): The current state we are in
        Return:
            softmax prob (array): Probability for choosing the action in the state
        """
        q_values = self.q_table[state, :].copy() # get Q-value for both actions
        
        # bias towards repeating actions 
        if self.prev_choice is not None:
            q_values[self.prev_choice] += self.theta 

        # softmax prob for action selection 
        exp_values = np.exp(self.beta * q_values)
        # normlize
        return exp_values / np.sum(exp_values)

    def policy(self, state):
        """
        Sample an action using softmax probabilities
        Args:
            state (int): The current state
        Returns:
            int: The chosen action (0,1)
        """
        action_probs = self.get_action_probabilities(state)
        # sample action based on softmax
        action = np.random.choice([0, 1], p=action_probs)
        return action

    def update_q_table(self, state, action, reward):
        """
        Update the Q-value for the given state and action using the RPE.

        Args:
            state (int): The current state
            action (int): The action taken
            reward (float): The reward received
            terminal (bool): Whether the outcome is terminal
        """
  
        prob_s1, prob_s2 = self.transition_probs[action]
        expected_q = prob_s1 * np.max(self.q_table[0, :]) + prob_s2 * np.max(self.q_table[1, :])
        rpe = reward + self.gamma * expected_q - self.q_table[state, action]

        self.q_table[state, action] += self.alpha * rpe
        self.prev_choice = action
        



def simulate_participant(trials=100, alpha=0.1, beta=5, gamma=0.9, theta=0.2):
    """
    Simulate participants decisions in the two-armed bandit task
    Args:
        trials (int): Number of trials to simulate
        alpha (float): Learning rate for the agent
        beta (float): Inverse temperature for softmax action selection
        gamma (float): Discount factor
        theta (float): Choice preservation

    Returns:
        choices (list): Sequence of chosen actions
        rewards (list): Sequence of rewards received
        agent (ModelBasedRL): The trained agent after simulation
    """
    model = ModelBasedRL(alpha, beta, gamma, theta)
    choices = []
    rewards = []
    for t in range(trials):
        state = 0
        action = model.policy(state=0)
        choices.append(action)
        prob_s1, prob_s2 = model.transition_probs[action]
        outcome_state = np.random.choice([1, 2], p=[prob_s1, prob_s2])

        if outcome_state == 1:
            reward = np.random.choice([0, 1], p=[0.2, 0.8])
        else:
            reward = np.random.choice([0, 1], p=[0.8, 0.2])
        rewards.append(reward)

        model.update_q_table(state=state, action=action, reward=reward)

    return choices, rewards, model

def plot_simulated_behavior(choices, rewards):
    """
    Plots reward outcomes (0 or 1) and colors them based on chosen action.
    
    Args:
        choices (list): Sequence of chosen actions (0 or 1)
        rewards (list): Sequence of rewards received (0 or 1)
    """
    trials = np.arange(len(choices))

    # Define colors based on the chosen action (e.g., 0 = blue, 1 = red)
    colors = ['blue' if choice == 0 else 'red' for choice in choices]
    print(colors)
    plt.figure(figsize=(10, 5))
    
    # Scatter plot where the reward is 0 or 1, colored by choice
    plt.scatter(trials, rewards, c=colors, edgecolors='black', marker='o', s=100, alpha=0.75)
    
    plt.xlabel('Trial')
    plt.ylabel('Reward (0 or 1)')
    plt.title('Simulated Rewards Over Trials (Colored by Choice)')
    
    # Custom legend for choices
    plt.scatter([], [], color='blue', label='Choice 0')
    plt.scatter([], [], color='red', label='Choice 1')
    plt.legend()

    plt.show()
    