# modeling_group3
# Model-Based vs. Model-Free Reinforcement Learning

This project explores the differences between **model-based** and **model-free** reinforcement learning (RL) using simulations, parameter fitting, recovery, and model comparison. The task is a two-armed bandit with probabilistic reward switching.

---

## Project Structure
├── Group3_Experiment_1.ipynb # Main notebook (run everything from here) 
└── src/
    ├── data_loading.py # Loads participant data from .json files 
    ├── explore_data.py # Plots and behavioral metrics 
    ├── model_based_rl.py # Model-Based RL implementation 
    ├── model_free_rl.py # Model-Free RL implementation
    ├── parameter_fitting.py # Grid search parameter fitting 
    ├── parameter_recovery.py # Parameter recovery scripts
    ├── model_comparison.py # Log-likelihood and BIC comparison 
    ├── model_recovery.py # Model identity recovery 
└── data/ # Folder for .json data files

##Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scipy tqdm
```

##How to Use: 
# Local
1. Place all .py files and .json data files in the same folder.
2. Edit the file path in data_loading.py:
```bash
folder_path = "C:/your_path/modeling_group3/data"
```

# Google Colab
1. Upload all .py files and .json files via the file browser
2. Uncomment the first load_data function in data_loading.py

```bash
# from google.colab import files
# def load_data(num_participants): 
#     uploaded = files.upload()
```
