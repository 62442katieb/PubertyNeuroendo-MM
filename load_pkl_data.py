import pickle
import pandas as pd
import numpy as np

def load_pkl_data(file_path):
    """
    Load pickle file and return as pandas DataFrame
    
    Args:
        file_path (str): Path to the .pkl file
        
    Returns:
        pd.DataFrame: Loaded data as DataFrame
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, pd.DataFrame):
            print(f"Successfully loaded DataFrame with shape: {data.shape}")
            return data
        else:
            print(f"Warning: Loaded data is not a DataFrame, type: {type(data)}")
            return data
            
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def explore_data(df):
    """
    Display basic information about the DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame to explore
    """
    if df is None:
        return
    
    print("\n" + "="*50)
    print("DATA EXPLORATION")
    print("="*50)
    
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\nColumn names:")
    print(df.columns.tolist())
    
    print("\nData types:")
    print(df.dtypes.value_counts())
    
    print("\nFirst 5 rows, first 5 columns:")
    print(df.iloc[:5, :5])
    
    print("\nDescriptive statistics (first 5 columns):")
    print(df.iloc[:, :5].describe())
    
    # Check for hormone columns
    hormone_cols = ['filtered_dhea', 'filtered_ert', 'filtered_hse']
    available_hormones = [col for col in hormone_cols if col in df.columns]
    
    if available_hormones:
        print(f"\nHormone target columns found: {available_hormones}")
        for col in available_hormones:
            print(f"\n{col} statistics:")
            print(df[col].describe())
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    if missing_counts.any():
        print("\nMissing values:")
        print(missing_counts[missing_counts > 0])
    else:
        print("\nNo missing values found")

if __name__ == "__main__":
    # Load male dataset
    print("Loading male dataset...")
    df_male = load_pkl_data('synthetic_dset-M.pkl')
    if df_male is not None:
        explore_data(df_male)
    
    # # Load female dataset (if exists)
    # print("\n" + "="*80)
    # print("Loading female dataset...")
    # df_female = load_pkl_data('synthetic_dset_F.pkl')
    # if df_female is not None:
    #     explore_data(df_female)