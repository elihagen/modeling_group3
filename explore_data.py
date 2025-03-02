import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_choices_over_time(df):

    choice_props = df.groupby('trial_index')['response'].mean()

    plt.plot(choice_props.index, choice_props.values, marker='o')
    plt.xlabel('Block of 10 trials')
    plt.ylabel('Mean Choice (proportion of right bandit)')
    plt.title('Choice Proportions Over Time')
    plt.show()
    

def plot_reward_over_time(df): 
    df['block'] = df['trial_index'] // 10

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