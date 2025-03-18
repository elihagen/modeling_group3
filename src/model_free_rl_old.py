# Code written with the help of ChatGPT and some functions are from model_based_rl.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def softmax(q_values, beta):
    """ Compute action probabilities using softmax """
    exp_q = np.exp(beta * q_values)
    return exp_q / np.sum(exp_q)


class ModelFreeRL:
    def __init__(self, alpha=0.1, beta=5, theta = 0.2, gamma=0.9, epsilon=0.2, explore=False):

        self.alpha = alpha # Learning rate of stage 1
        self.beta = beta    # Inverse temperature for softmax
        self.gamma = gamma  # Discount factor
        self.theta = theta  # Choice perseveration
        self.epsilon = epsilon # Exploration probability
        self.explore = explore
        self.prev_choice = None # Needed for choice perservation
        # Fixed probabilities for actions
        self.reward_probs = [0.2, 0.8]  
        self.q_table = np.zeros((2,2))


    
    def choose_action(self):
        """ Select an action based on predefined probabilities, incorporating choice perseveration and exploration """
        probs = self.reward_probs.copy()
   
        # Apply choice perseveration
        if self.prev_choice is not None:
            probs[self.prev_choice] += self.theta
        
        if np.random.rand() < self.epsilon:
            return np.random.randint(2) 
        else:
            if self.prev_choice is not None:
                probs[self.prev_choice] += self.theta

            # Ensure probabilities remain valid
            probs = np.maximum(probs, 0)  # Ensure no negative values
            probs /= np.sum(probs)  # Normalize to sum to 1
            
            return np.random.choice(len(self.q_table), p=probs)

    
    def update(self, choice, reward):
        """ Update the values using TD learning """

        delta = reward - self.q_table[choice]
        self.q_table[choice] += self.alpha * delta

        self.prev_choice = choice


    def simulate_trial(self):
        """ Simulate one trial """
        choice = self.choose_action()
        
        
        reward = np.random.choice([0, 1], p = [1 - self.reward_probs[choice], self.reward_probs[choice]])  # Random binary reward with the probabilities assigned
        
        self.update(choice, reward)
        
        return choice, reward

def simulate_participant_TD(n_trials=100, alpha=0.1, beta=5, gamma=0.9, theta=0.2):
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
    """
   
    model = ModelFreeRL()
    n_trials = 100
    choices = []
    rewards = []
 
    for t in range(n_trials):
        choice, reward = model.simulate_trial()
        choices.append(choice)
        rewards.append(reward)

    return choices, rewards
    
def plot_simulated_behavior_TD(choices, rewards):
    """
    Plots reward outcomes (0 or 1) and colors them based on chosen action.
    
    Args:
        choices (list): Sequence of chosen actions (0 or 1)
        rewards (list): Sequence of rewards received (0 or 1)
    """

    trials = np.arange(len(choices))
    colors = ['blue' if choice == 0 else 'red' for choice in choices]

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
 




