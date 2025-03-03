import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model_based_rl import *

def plot_choices_over_time(df):

    choice_props = df.groupby('trial_index')['response'].mean()

    plt.plot(choice_props.index, choice_props.values, marker='o')
    plt.xlabel('Block of 10 trials')
    plt.ylabel('Mean Choice (proportion of right bandit)')
    plt.title('Choice Proportions Over Time')
    plt.show()
    

def plot_reward_over_time(df): 
    df['block'] = df['trial_index'] #// 10

    reward_rate = df.groupby('block')['value'].mean()
    plt.plot(reward_rate.index, reward_rate.values, marker='o')
    plt.xlabel('Block of 10 trials')
    plt.ylabel('Average Reward')
    plt.title('Reward Rate Over Time')
    plt.show()
    
    
    
def switch_win_loss(df): 
    df['prev_choice'] = df['response'].shift(1)
    df = df.loc[:, ~df.columns.duplicated()]
    df['prev_reward'] = df['value'].shift(1)

    df['switch'] = (df['value'] != df['prev_choice']).astype(int)

    # Switch rate after win:
    switch_after_win = df.loc[df['prev_reward']==1, 'switch'].mean()
    # Switch rate after loss:
    switch_after_loss = df.loc[df['prev_reward']==0, 'switch'].mean()

    print("Switch after Win:", switch_after_win)
    print("Switch after Loss:", switch_after_loss)
    
def plot_rt(df): 
    plt.hist(df['rt'], bins=30)
    plt.xlabel('Reaction Time (ms)')
    plt.ylabel('Frequency')
    plt.title('Distribution of RTs')
    plt.show()
    
    
 
def simulate_participant(trials=100, alpha=0.1, beta=5, gamma=0.9, theta=0.2):
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

        model.update_q_table(state=state, action=action, reward=reward, terminal=True)

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
