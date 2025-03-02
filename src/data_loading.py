import pandas as pd
import json
from google.colab import files


def load_data(): 
    # Upload the file manually
    uploaded = files.upload()

    # Get the filename from the uploaded dictionary
    file_name = list(uploaded.keys())[0]

    # Open and load the JSON file
    with open(file_name, "r") as f:
        data = json.load(f)

    df = pd.json_normalize(data)
    
    print("Response pattern: ", df["response"])
    return df
    
    
    
def processing_data(df): 
    print(df.columns)
    
    # Rename columns for consistency
    df.rename(columns=lambda x: x.replace("bean_", ""), inplace=True)

    #filter out instructions, by removing any rows from df where the column 'bandits' has NaN
    df = df.dropna(subset=['bandits'])
    
    print(df)