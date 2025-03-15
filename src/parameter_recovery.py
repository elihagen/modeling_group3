from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from parameter_fitting import simulate_data_for_fitting, grid_search_parameter_fit, simulate_mfrl_trials, grid_search_parameter_fit_mf

def parameter_recovery(num_samples=20):
    """
    Perform parameter recovery by simulating data from randomly sampled parameters
    and then recovering them using grid search.

    Args:
        num_samples (int): Number of parameter sets to sample

    Returns:
        tuple: (true_params, recovered_params)
            true_params (dict): True 'alpha' and 'beta' values
            recovered_params (dict): Recovered 'alpha' and 'beta' values
    """
    true_params = {
        'alpha': np.random.uniform(0, 1, num_samples),
        'beta': np.random.uniform(0.1, 10, num_samples), 
        'theta': np.random.uniform(0, 1, num_samples)
    }
    recovered_params = {'alpha': [], 'beta': [], 'theta': []}

    for i in range(num_samples):
        alpha, beta, theta = true_params['alpha'][i], true_params['beta'][i], true_params['theta'][i]
        simulated_data = simulate_data_for_fitting(alpha=alpha, beta=beta, theta=theta)
        estimated_params, _ = grid_search_parameter_fit(simulated_data)
        recovered_params['alpha'].append(estimated_params[0])
        recovered_params['beta'].append(estimated_params[1])
        recovered_params['theta'].append(estimated_params[2])

    return true_params, recovered_params

def evaluate_parameter_fit(true_params, recovered_params):
    """
    Evaluate the fit of recovered parameters using correlation and mean absolute error.
    Args:
        true_params (dict): Dictionary with true 'alpha' and 'beta' arrays
        recovered_params (dict): Dictionary with recovered 'alpha' and 'beta' lists

    Returns:
        tuple: ((alpha_corr, beta_corr), (alpha_mae, beta_mae))
            alpha_corr (float): Pearson correlation for alpha
            beta_corr (float): Pearson correlation for beta
            alpha_mae (float): Mean absolute error for alpha
            beta_mae (float): Mean absolute error for beta
    """
    alpha_corr, _ = pearsonr(true_params['alpha'], recovered_params['alpha'])
    beta_corr, _ = pearsonr(true_params['beta'], recovered_params['beta'])
    theta_corr, _ = pearsonr(true_params['theta'], recovered_params['theta'])

    alpha_mae = np.mean(np.abs(np.array(true_params['alpha']) - np.array(recovered_params['alpha'])))
    beta_mae = np.mean(np.abs(np.array(true_params['beta']) - np.array(recovered_params['beta'])))
    theta_mae = np.mean(np.abs(np.array(true_params['theta']) - np.array(recovered_params['theta'])))
    print(f"Alpha Correlation: {alpha_corr:.3f}, Alpha MAE: {alpha_mae:.3f}")
    print(f"Beta Correlation: {beta_corr:.3f}, Beta MAE: {beta_mae:.3f}")
    print(f"Theta Correlation: {theta_corr:.3f}, Theta MAE: {theta_mae:.3f}")

    return (alpha_corr, beta_corr, theta_corr), (alpha_mae, beta_mae, theta_mae)

def plot_param_recovery(true_params, recovered_params):
    """
    Plot true vs. recovered parameters.
    Args:
        true_params (dict): Dictionary with true parameter arrays
        recovered_params (dict): Dictionary with recovered parameter lists
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    axes[0].scatter(true_params['alpha'], recovered_params['alpha'], alpha=0.7)
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='red')
    axes[0].set_xlabel('True Alpha')
    axes[0].set_ylabel('Recovered Alpha')
    axes[0].set_title('Alpha Parameter Recovery')

    axes[1].scatter(true_params['beta'], recovered_params['beta'], alpha=0.7)
    axes[1].plot([0, 10], [0, 10], linestyle='--', color='red')
    axes[1].set_xlabel('True Beta')
    axes[1].set_ylabel('Recovered Beta')
    axes[1].set_title('Beta Parameter Recovery')
    
    axes[2].scatter(true_params['theta'], recovered_params['theta'], alpha=0.7)
    axes[2].plot([0, 10], [0, 10], linestyle='--', color='red')
    axes[2].set_xlabel('True theta')
    axes[2].set_ylabel('Recovered theta')
    axes[2].set_title('theta Parameter Recovery')

    plt.show()

def parameter_recovery_mfrl(num_samples=20):
    """
    Perform parameter recovery for Model-Free RL by simulating data from randomly sampled parameters
    and then recovering them using grid search.

    Args:
        num_samples (int): Number of parameter sets to sample

    Returns:
        tuple: (true_params, recovered_params)
            true_params (dict): True 'alpha', 'beta', and 'theta' values
            recovered_params (dict): Recovered 'alpha', 'beta', and 'theta' values
    """
    true_params = {
        'alpha': np.random.uniform(0, 1, num_samples),
        'beta': np.random.uniform(0.1, 10, num_samples),
        'theta': np.random.uniform(0, 1, num_samples)
    }
    recovered_params = {'alpha': [], 'beta': [], 'theta': []}

    for i in range(num_samples):
        alpha, beta, theta = true_params['alpha'][i], true_params['beta'][i], true_params['theta'][i]
        simulated_data = simulate_mfrl_trials(trials=100, alpha=alpha, beta=beta, theta=theta)
        estimated_params, _ = grid_search_parameter_fit_mf(simulated_data)
        recovered_params['alpha'].append(estimated_params[0])
        recovered_params['beta'].append(estimated_params[1])
        recovered_params['theta'].append(estimated_params[2])

    return true_params, recovered_params

def evaluate_parameter_fit_mfrl(true_params, recovered_params):
    """
    Evaluate the fit of recovered parameters for Model-Free RL using correlation and mean absolute error.
    
    Args:
        true_params (dict): Dictionary with true 'alpha', 'beta', and 'theta' arrays
        recovered_params (dict): Dictionary with recovered 'alpha', 'beta', and 'theta' lists

    Returns:
        tuple: ((alpha_corr, beta_corr, theta_corr), (alpha_mae, beta_mae, theta_mae))
    """
    alpha_corr, _ = pearsonr(true_params['alpha'], recovered_params['alpha'])
    beta_corr, _ = pearsonr(true_params['beta'], recovered_params['beta'])
    theta_corr, _ = pearsonr(true_params['theta'], recovered_params['theta'])

    alpha_mae = np.mean(np.abs(np.array(true_params['alpha']) - np.array(recovered_params['alpha'])))
    beta_mae = np.mean(np.abs(np.array(true_params['beta']) - np.array(recovered_params['beta'])))
    theta_mae = np.mean(np.abs(np.array(true_params['theta']) - np.array(recovered_params['theta'])))

    print(f"Alpha Correlation: {alpha_corr:.3f}, Alpha MAE: {alpha_mae:.3f}")
    print(f"Beta Correlation: {beta_corr:.3f}, Beta MAE: {beta_mae:.3f}")
    print(f"Theta Correlation: {theta_corr:.3f}, Theta MAE: {theta_mae:.3f}")

    return (alpha_corr, beta_corr, theta_corr), (alpha_mae, beta_mae, theta_mae)

def plot_param_recovery_mfrl(true_params, recovered_params):
    """
    Plot true vs. recovered parameters for Model-Free RL.
    
    Args:
        true_params (dict): Dictionary with true parameter arrays
        recovered_params (dict): Dictionary with recovered parameter lists
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    axes[0].scatter(true_params['alpha'], recovered_params['alpha'], alpha=0.7)
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='red')
    axes[0].set_xlabel('True Alpha')
    axes[0].set_ylabel('Recovered Alpha')
    axes[0].set_title('Alpha Parameter Recovery')

    axes[1].scatter(true_params['beta'], recovered_params['beta'], alpha=0.7)
    axes[1].plot([0, 10], [0, 10], linestyle='--', color='red')
    axes[1].set_xlabel('True Beta')
    axes[1].set_ylabel('Recovered Beta')
    axes[1].set_title('Beta Parameter Recovery')
    
    axes[2].scatter(true_params['theta'], recovered_params['theta'], alpha=0.7)
    axes[2].plot([0, 1], [0, 1], linestyle='--', color='red')
    axes[2].set_xlabel('True Theta')
    axes[2].set_ylabel('Recovered Theta')
    axes[2].set_title('Theta Parameter Recovery')

    plt.show()