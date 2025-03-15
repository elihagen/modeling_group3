import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from model_based_rl import ModelBasedRL
from model_free_rl import ModelFreeRL, softmax

def log_likelihood(agent, data):
    """
    Compute the log-likelihood of a dataset given an agent.

    Args:
        agent: our model (here: model-based)
        data (pd.DataFrame): DataFrame with columns 'choice' and 'reward'

    Returns:
        float:Log-likelihood of the data
    """
    ll_sum = 0
    for idx, trial_data in data.iterrows():
        chosen_action = trial_data['choice']
        state = 0
        received_reward = int(trial_data['reward'])

        action_probs = agent.get_action_probabilities(state)
        chosen_action_prob = action_probs[chosen_action]
        agent.update_q_table(state, chosen_action, received_reward)

        ll_sum += np.log(chosen_action_prob)

    return ll_sum

def grid_search_parameter_fit(data, alpha_range=np.linspace(0, 1, 10), beta_range=np.linspace(0.1, 10, 10), theta_range=np.linspace(-1, 1, 10)):
    """
    Perform a grid search over alpha and beta values to find the best-fitting parameters.

    Args:
        data (pd.DataFrame): DataFrame with columns 'choice' and 'reward'
        alpha_range (np.array): Array of alpha values to search
        beta_range (np.array): Array of beta values to search
        theta_range (np.array): Array of theta values to search

    Returns:
        tuple: (best_params, best_likelihood)
            best_params (tuple): Best (alpha, beta, theta) pair
            best_likelihood (float): Corresponding log-likelihood
    """
    best_params = None
    best_likelihood = float('-inf')

    for alpha in alpha_range:
        for beta in beta_range:
            for theta in theta_range: 
                agent = ModelBasedRL(alpha=alpha, beta=beta, theta=theta)
                likelihood = log_likelihood(agent, data)
                if likelihood > best_likelihood:
                    best_likelihood = likelihood
                    best_params = (alpha, beta, theta)

    return best_params, best_likelihood

def simulate_data_for_fitting(trials=100, alpha=0.1, beta=5, gamma=0.9, theta=0.2):
    """
    Generate synthetic data using a model with known parameters.
    Args:
        trials (int): Number of trials to simulate
        alpha (float): True learning rate
        beta (float): True inverse temperature
        gamma (float): Discount factor

    Returns:
        pd.DataFrame: DataFrame with columns 'choice' and 'reward'
    """
    model = ModelBasedRL(alpha, beta, gamma, theta)
    data = []
    for t in range(trials):
        state = 0
        action_probs = model.get_action_probabilities(state)
        action = np.random.choice([0, 1], p=action_probs)
        reward = np.random.choice([0, 1], p=[0.3, 0.7]) if action == 0 else np.random.choice([0, 1], p=[0.7, 0.3])
        model.update_q_table(state, action, reward)
        data.append({'choice': action, 'reward': reward})
    return pd.DataFrame(data)

def simulate_mfrl_trials(trials=100, alpha=0.1, beta=5, gamma=0.9, theta=0.2):
    agent = ModelFreeRL(alpha, beta, theta, gamma)
    data = []
    
    for _ in range(trials):
        action = agent.choose_action()
        reward = np.random.choice([0, 1], p=[0.3, 0.7] if action == 0 else [0.7, 0.3])
        agent.update(action, reward)
        data.append({'choice': action, 'reward': reward})
    
    return pd.DataFrame(data)


def log_likelihood_mf(agent, data):
    """
    Compute the log-likelihood of a dataset given a Model-Free RL agent.

    Args:
        agent (ModelFreeRL): Model-Free RL agent
        data (pd.DataFrame): DataFrame with columns 'choice' and 'reward'

    Returns:
        float: Log-likelihood of the data
    """
    ll_sum = 0
    for _, trial_data in data.iterrows():
        chosen_action = trial_data['choice']
        received_reward = int(trial_data['reward'])

        # Get action probabilities using softmax
        action_probs = softmax(agent.q_table, agent.beta)
        chosen_action_prob = action_probs[chosen_action]

        # Update the agent
        agent.update(chosen_action, received_reward)

        # Accumulate log-likelihood
        ll_sum += np.log(chosen_action_prob)

    return ll_sum

def grid_search_parameter_fit_mf(data, alpha_range=np.linspace(0, 1, 10), beta_range=np.linspace(0.1, 10, 10), theta_range=np.linspace(-1, 1, 10)):
    """
    Perform a grid search over alpha, beta, and theta values to find the best-fitting parameters
    for the Model-Free RL agent.

    Args:
        data (pd.DataFrame): DataFrame with columns 'choice' and 'reward'
        alpha_range (np.array): Array of alpha values to search
        beta_range (np.array): Array of beta values to search
        theta_range (np.array): Array of theta values to search

    Returns:
        tuple: (best_params, best_likelihood)
            best_params (tuple): Best (alpha, beta, theta) pair
            best_likelihood (float): Corresponding log-likelihood
    """
    best_params = None
    best_likelihood = float('-inf')

    for alpha in alpha_range:
        for beta in beta_range:
            for theta in theta_range:
                agent = ModelFreeRL(alpha=alpha, beta=beta, theta=theta)
                likelihood = log_likelihood_mf(agent, data)
                print(likelihood, best_likelihood)
                if likelihood > best_likelihood:
                    best_likelihood = likelihood
                    best_params = (alpha, beta, theta)

    return best_params, best_likelihood
