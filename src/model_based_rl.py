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
            gamma: Discount factor (not needed in bandit task)
            theta: Stickiness parameter (choice perseveration)
        """
        self.alpha = alpha  # Learning rate
        self.beta = beta  # exploration-exploitation balance (inverse temperature)
        self.gamma = gamma  # Discount factor
        self.theta = theta # choice preservation
        self.q_table = np.zeros((2,2))
        self.prev_choice = None
        self.transition_probs = {
            0: (0.7, 0.3),
            1: (0.3, 0.7)
        }

    def get_action_probabilities(self, state):
        """
        Compute softmax probs for actions in the state
        Args:
            state (int): The current state we are in
        Return:
            softmax prob (array): Probability for choosing the action in the state
        """
        q_values = self.q_table[state, :].copy()
        if self.prev_choice is not None:
            q_values[self.prev_choice] += self.theta 

        exp_values = np.exp(self.beta * q_values)
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
        return np.random.choice([0, 1], p=action_probs)

    def update_q_table(self, state, action, reward,terminal):
        """
        Update the Q-value for the given state and action using the RPE.

        Args:
            state (int): The current state
            action (int): The action taken
            reward (float): The reward received
            terminal (bool): Whether the outcome is terminal
        """
        if terminal:
            rpe = reward - self.q_table[state, action]
        else:
            prob_s1, prob_s2 = self.transition_model[action]
            expected_q = prob_s1 * np.max(self.q_table[1, :]) + prob_s2 * np.max(self.q_table[2, :])
            rpe = reward + self.gamma * expected_q - self.q_table[state, action]

        self.q_table[state, action] += self.alpha * rpe
        self.prev_choice = action

