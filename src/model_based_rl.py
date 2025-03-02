import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ModelBasedRL:
    def __init__(self, alpha = 0.1, beta = 5, gamma = 0.9):
        """
        Initialize the model-based RL model
        We consider:
        - Q-table is of size (2, 2) - two states and two actions
        - transition_probs: gives the probabilities of outcomes for each action.
        """
        self.alpha = alpha  # Learning rate
        self.beta = beta  # exploration-exploitation balance
        self.gamma = gamma  # Discount factor
        self.q_table = np.zeros((2,2))
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
        q_values = self.q_table[state, :]
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


# YOUR MODEL SIMULATION CODE GOES HERE

def simulate_participant(trials=100, alpha=0.1, beta=5, gamma=0.9):
    """
    Simulate participants decisions in the two-armed bandit task
    Args:
        trials (int): Number of trials to simulate
        alpha (float): Learning rate for the agent
        beta (float): Inverse temperature for softmax action selection
        gamma (float): Discount factor (unused in terminal updates here)

    Returns:
        choices (list): Sequence of chosen actions
        rewards (list): Sequence of rewards received
        agent (ModelBasedRL): The trained agent after simulation
    """
    model = ModelBasedRL(alpha, beta, gamma)
    choices = []
    rewards = []
    for t in range(trials):
        state = 0
        action = model.policy(state)
        choices.append(action)
        prob_s1, prob_s2 = model.transition_probs[action]
        outcome_state = np.random.choice([1, 2], p=[prob_s1, prob_s2])

        if outcome_state == 1:
            reward = np.random.choice([0, 1], p=[0.2, 0.8])
        else:
            reward = np.random.choice([0, 1], p=[0.8, 0.2])
        rewards.append(reward)

        model.update_q_table(state=state, action=action, reward=reward, terminal=True)

    return choices, rewards, model

def plot_simulated_behavior(choices, rewards):

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(choices, label='Choices (0 or 1)', marker='o', linestyle='')
    ax.plot(rewards, label='Rewards (0 or 1)', marker='x', linestyle='')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Action / Reward')
    ax.set_title('Simulated Behavior Over Trials')
    ax.legend()
    plt.show()

def analyze_exploration_exploitation(choices):
    """
    Args:
        choices (list): Sequence of chosen actions.

    Returns:
        tuple: (exploration_count, exploitation_count)
            exploration_count (int): Number of trials where the action changed
            exploitation_count (int): Number of trials where the action remained the same
    """
    exploration_counts = sum([1 for i in range(1, len(choices)) if choices[i] != choices[i-1]])
    exploitation_counts = len(choices) - exploration_counts
    return exploration_counts, exploitation_counts


def analyze_choice_perseveration(choices, rewards):
    """
    Analyze choice perseveration using switch-after-loss and stay-after-win

    Args:
        choices (list): List of chosen actions
        rewards (list): List of rewards received

    Returns:
        tuple: (switch_after_loss, stay_after_win)
            switch_after_loss (int): Count of switches following a loss
            stay_after_win (int): Count of stays following a win
    """
    switch_after_loss = 0
    stay_after_win = 0
    for i in range(1, len(choices)):
        if rewards[i-1] == 0 and choices[i] != choices[i-1]:
            switch_after_loss += 1
        if rewards[i-1] == 1 and choices[i] == choices[i-1]:
            stay_after_win += 1
    return switch_after_loss, stay_after_win
