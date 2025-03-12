import pandas as pd
import json
from google.colab import files


def load_data(): 
    # Upload the file manually
    uploaded = files.upload()

    # Get the filename from the uploaded dictionary
    file_name = list(uploaded.keys())[0]
    df_list = []

    for file_name in uploaded.keys():
        with open(file_name, "r") as f:
            data = json.load(f)


    # # Open and load the JSON file
    # with open(file_name, "r") as f:
    #     data = json.load(f)

    # Normalize JSON structure into a Pandas DataFrame
        df = pd.json_normalize(data)

        # Append processed DataFrame to the list
        df_list.append(df)

    # Concatenate all DataFrames into a single DataFrame
    final_df = pd.concat(df_list, ignore_index=True)

    return final_df
    
    
    
def processing_data(df): 
    
    # Rename columns for consistency
    df.rename(columns=lambda x: x.replace("bean_", ""), inplace=True)

    #filter out instructions, by removing any rows from df where the column 'bandits' has NaN
    df = df.dropna(subset=['bandits'])
    # drop rows containing NaN and first row containing "q" to start experiment
    df = df[~df['response'].astype(str).str.contains(r'\bq\b', na=False)]
    df = df[df['response'].astype(str).str.strip() != '']
    df = df.dropna(subset=['response'])
    # print(df["response"])
    df['response'] = pd.to_numeric(df['response'])
        
    print("dataframe", df)
    return df