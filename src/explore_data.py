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
    df['block'] = df['trial_index'] #// 10  # do we want to plot averaged reward?

    reward_rate = df.groupby('block')['value'].mean()
    plt.plot(reward_rate.index, reward_rate.values, marker='o')
    plt.xlabel('Block of 10 trials')
    plt.ylabel('Average Reward')
    plt.title('Reward Rate Over Time')
    plt.show()
    
    
    
def switch_win_loss(df): 
    df = df.dropna(axis=1, how='all') 
    df = df.loc[:, ~df.columns.duplicated()]
    df['prev_choice'] = df['response'].shift(1)
    df['prev_reward'] = df['value'].shift(1)

    df['switch'] = (df['response'] != df['prev_choice']).astype(int)

    # Switch rate after win:
    switch_after_win = df.loc[df['prev_reward']==1, 'switch'].mean()
    # Switch rate after loss:
    switch_after_loss = df.loc[df['prev_reward']==0, 'switch'].mean()

    print("Switch after Win:", switch_after_win)
    print("Switch after Loss:", switch_after_loss)
    
def plot_rt(df): 
    # plot the response time as histogram 
    plt.hist(df['rt'], bins=30)
    plt.xlabel('Reaction Time (ms)')
    plt.ylabel('Frequency')
    plt.title('Distribution of RTs')
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
