from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from parameter_fitting import simulate_data_for_fitting, grid_search_parameter_fit, simulate_mfrl_trials, grid_search_parameter_fit_mf

def parameter_recovery(num_samples=20):
    true_params = {
        'alpha': np.random.uniform(0, 1, num_samples),
        'beta': np.random.uniform(0.1, 10, num_samples), 
        'theta': np.random.uniform(0, 1, num_samples)
    }
    recovered_params = {'alpha': [], 'beta': [], 'theta': []}

    for i in range(num_samples):
        alpha, beta, theta = true_params['alpha'][i], true_params['beta'][i], true_params['theta'][i]
        simulated_data = simulate_data_for_fitting(alpha=alpha, beta=beta, theta=theta)
        simulated_data += np.random.normal(0, 0.05, simulated_data.shape)  # Adding noise
        estimated_params, _ = grid_search_parameter_fit(simulated_data)
        recovered_params['alpha'].append(estimated_params[0])
        recovered_params['beta'].append(estimated_params[1])
        recovered_params['theta'].append(estimated_params[2])

    return true_params, recovered_params

def plot_density_recovery(true_params, recovered_params):
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    for i, param in enumerate(['alpha', 'beta', 'theta']):
        sns.kdeplot(true_params[param], label='True', ax=axes[i], fill=True)
        sns.kdeplot(recovered_params[param], label='Recovered', ax=axes[i], fill=True)
        axes[i].set_xlabel(param)
        axes[i].set_title(f'{param} Distribution')
        axes[i].legend()

    plt.show()

def parameter_recovery_mfrl(num_samples=20):
    true_params = {
        'alpha': np.random.uniform(0, 1, num_samples),
        'beta': np.random.uniform(0.1, 10, num_samples),
        'theta': np.random.uniform(0, 1, num_samples)
    }
    recovered_params = {'alpha': [], 'beta': [], 'theta': []}

    for i in range(num_samples):
        alpha, beta, theta = true_params['alpha'][i], true_params['beta'][i], true_params['theta'][i]
        simulated_data = simulate_mfrl_trials(trials=100, alpha=alpha, beta=beta, theta=theta)
        simulated_data += np.random.normal(0, 0.05, simulated_data.shape)  # Adding noise
        estimated_params, _ = grid_search_parameter_fit_mf(simulated_data)
        recovered_params['alpha'].append(estimated_params[0])
        recovered_params['beta'].append(estimated_params[1])
        recovered_params['theta'].append(estimated_params[2])

    return true_params, recovered_params
