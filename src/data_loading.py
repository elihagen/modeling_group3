import pandas as pd
import json
# from google.colab import files
import os
import glob


#### uncomment if you run this code in colab 
# def load_data(num_participants): 
    # Upload the file manually
    # uploaded = files.upload()
    # file_name = list(uploaded.keys())[0]
    # for file_name in uploaded.keys():
    #     with open(file_name, "r") as f:
    #         data = json.load(f)
        
    #     df = pd.json_normalize(data)

    #     # Append processed DataFrame to the list
    #     df_list.append(df)

    # # Concatenate all DataFrames into a single DataFrame
    # final_df = pd.concat(df_list, ignore_index=True)
    # return final_df
    
    
def load_data(num_participants): 
    """
    Loads and parses participant data from JSON files.
    Args:
        num_participants (str): Either 'merged' to load all JSON files or 'single' to load one.
    Returns:
        pd.DataFrame: A dataframe containing participant(s) data.
    """
    # Upload the file manually
    # uploaded = files.upload()
    folder_path = "/Users/pelinkomurluoglu/Desktop/safeSafeProject/modeling_group3/data"#"C:\Arbeitsordner\Semester3\modeling_group3\data"

    # Get the filename from the uploaded dictionary
    # file_name = list(uploaded.keys())[0]
    df_list = []
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    #print(json_files)
    # for file_name in uploaded.keys():
    if num_participants == "merged": 
        for file_name in json_files:
            with open(file_name, "r") as f:
                data = json.load(f)
            
            df = pd.json_normalize(data)

            # Append processed DataFrame to the list
            df_list.append(df)

        # Concatenate all DataFrames into a single DataFrame
        final_df = pd.concat(df_list, ignore_index=True)

    elif num_participants == "single": 
        file_name = json_files[0]
        with open(file_name, "r") as f:
            final_df = json.load(f)
            final_df = pd.json_normalize(final_df)   
    else: 
        print(f"You must specify 'merged' or 'single' to specify if you are interested in the behavior of multiple participants (merged) or a single participant (single). Your input {num_participants} is not a valid input to the function.")            
    
    return final_df
    
    
    
def processing_data(df): 
    """
    Processes and cleans the raw participant dataframe.
    - Renames columns
    - Filters out instructions and invalid responses
    - Converts response column to numeric
    - Removes duplicated reward columns if any

    Args:
        df (pd.DataFrame): Raw data to process
    Returns:
        pd.DataFrame: Cleaned dataframe containing only relevant data.
    """
    # Rename columns for consistency
    df.rename(columns=lambda x: x.replace("bean_", ""), inplace=True)

    #filter out instructions, by removing any rows from df where the column 'bandits' has NaN
    df = df.dropna(subset=['bandits'])
    # drop rows containing NaN and first row containing "q" to start experiment
    df = df[~df['response'].astype(str).str.contains(r'\bq\b', na=False)]
    df = df[df['response'].astype(str).str.strip() != '']
    df = df.dropna(subset=['response'])
    df['response'] = pd.to_numeric(df['response'])
    
    
    # Select relevant columns (choice and reward)
    if 'reward' in df.columns and df.columns.duplicated().sum() > 0:
        df = df.loc[:, ~df.columns.duplicated()]
    return df