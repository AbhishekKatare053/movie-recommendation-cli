import pandas as pd
import ast

# Load raw datasets
metadata_df = pd.read_csv('data/metadata.csv', low_memory=False)
credits_df = pd.read_csv('data/credits.csv')

# 🔁 Convert both 'id' columns to string
metadata_df['id'] = metadata_df['id'].astype(str)
credits_df['id'] = credits_df['id'].astype(str)

# Extract cast names
def extract_cast_names(cast_str):
    try:
        cast_list = ast.literal_eval(cast_str)
        return ", ".join([cast['name'] for cast in cast_list[:5]])
    except:
        return ""

credits_df['cast_names'] = credits_df['cast'].apply(extract_cast_names)

# Merge datasets
def merge_data(metadata_df, credits_df):
    df = pd.merge(metadata_df, credits_df[['id', 'cast_names']], on='id', how='left')
    print(f"[INFO] Merged dataset: {df.shape}")
    df.to_csv("data/final_merged.csv", index=False)
    print("[INFO] Merged CSV saved at 'data/final_merged.csv'")
    return df

merge_data(metadata_df, credits_df)
