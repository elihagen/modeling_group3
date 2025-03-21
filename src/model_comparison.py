from data_loading import *
from parameter_fitting import *
from parameter_recovery import *
from model_based_rl import *
from model_free_rl import *
import matplotlib.pyplot as ptl


# Compute Bayesian Information Criterion (BIC)
def calculate_bic(num_params, num_data_points, ll):
    """
    Calculate Bayesian Information Criterion (BIC).
     Args:
        num_params (int): Number of parameters in the model.
        num_data_points (int): Number of data points (e.g., trials).
        ll (float): Log-likelihood of the fitted model.
    Returns:
        float: The BIC value.
    """
    return num_params * np.log(num_data_points) - 2 * ll


def run_model_comparison(num_participants):
    """
    Runs a model comparison between Model-Free and Model-Based reinforcement learning models.
    - Loads and preprocesses behavioral data.
    - Fits both Model-Free and Model-Based RL models using grid search.
    - Compares models based on log-likelihood and Bayesian Information Criterion (BIC).
    - Simulates behavior using the best-fitting model.
    - Visualizes model predictions vs. actual choices.

    Args:
        num_participants (str): 'merged' for aggregated data or 'single' for individual participant data.
    """
       
    # Load and preprocess the experimental data from our experiment 
    data = load_data(num_participants)
    data = processing_data(data)
    data = data.loc[:, ~data.columns.duplicated()]

    # Select relevant columns (choice and reward)
    if 'reward' in data.columns and data.columns.duplicated().sum() > 0:
        data = data.loc[:, ~data.columns.duplicated()]
    # Select relevant columns (choice and reward)
    data = data[['response', 'value']].rename(columns={'response': 'choice', 'value': 'reward'})

    data['choice'] = pd.to_numeric(data['choice'], errors='coerce')
    data['reward'] = pd.to_numeric(data['reward'], errors='coerce')

    # Fit Model-Based RL
    best_params_mb, best_likelihood_mb = grid_search_parameter_fit(data)
    model_based = ModelBasedRL(alpha=best_params_mb[0], beta=best_params_mb[1], gamma=best_params_mb[2], theta=best_params_mb[3])

    # Fit Model-Free RL
    best_params_mf, best_likelihood_mf = grid_search_parameter_fit_mf(data)
    model_free = ModelFreeRL(alpha=best_params_mf[0], beta=best_params_mf[1], theta=best_params_mf[2])

    # Evaluate which model performs better
    print(f"Model-Based Log-Likelihood: {best_likelihood_mb}")
    print(f"Model-Free Log-Likelihood: {best_likelihood_mf}")

    # Compare model fit
    if best_likelihood_mb > best_likelihood_mf:
        print("Model-Based RL performs better")
        best_model = model_based
        best_params = best_params_mb
    else:
        print("Model-Free RL performs better")
        best_model = model_free
        best_params = best_params_mf

    # Simulate behavior using the best model
    choices, rewards = [], []
    for _ in range(len(data)):
        choice = best_model.choose_action() if isinstance(best_model, ModelFreeRL) else best_model.policy(state=0)
        reward = np.random.choice([0, 1], p=[0.7, 0.3] if choice == 0 else [0.3, 0.7])
        try: 
            best_model.update(choice, reward)
        except: 
            best_model.update_q_table(action = choice, reward = reward)
        choices.append(choice)
        rewards.append(reward)
        
    # Number of parameters in each model
    n_params_mf = 3  # Alpha, Beta, Theta
    n_params_mb = 4  # Alpha, Beta, Gamma, Theta

    bic_mf = calculate_bic(n_params_mf, len(data), best_likelihood_mf)
    bic_mb = calculate_bic(n_params_mb, len(data), best_likelihood_mb)

    # Compare Results
    results = pd.DataFrame({
        'Model': ['Model-Free', 'Model-Based'],
        'BIC': [bic_mf, bic_mb]
    })

    from IPython.display import display
    display(results)

    # Plot the behavior of the best model against the data
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(data)), data['choice'], label='Actual Choices', alpha=0.5)
    plt.scatter(range(len(choices)), choices, label='Predicted Choices', alpha=0.5)
    plt.xlabel('Trial')
    plt.ylabel('Choice')
    plt.legend()
    plt.title('Best Model Behavior vs. Actual Data')
    plt.show()
