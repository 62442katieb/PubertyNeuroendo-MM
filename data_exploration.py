'''
Run separate models for male and female participants, so I’ve attached two pickle files with the synthetic data, 
along with lists of the input features and the hormones we’re trying to predict from them. 
Note that the output hormone lists are different for male and female models.

Purpose of this file: Preprocess and explore the synthetic dataset.

Author: Jinyao Tian
Date: 2025-04-22
'''

import os
import pickle
import numpy as np
import pandas as pd

# Load the male pickle file
with open('synthetic_dset-M.pkl', 'rb') as f:
    m_data = pickle.load(f)

# Now `m_data` contains whatever Python object was saved
print(m_data.shape) # (5137, 158)
# Assuming m_data is a DataFrame
na_columns = m_data.isnull().any()
# Print columns with NA values
print(na_columns[na_columns])  

with open('synthetic_dset_F.pkl', 'rb') as f:
    f_data = pickle.load(f)

print(f_data.shape) # (3963, 159)
na_columns = f_data.isnull().any()
# Print columns with NA values
print(na_columns[na_columns]) 

# No dirty data in the dataset

'''
Focus on the male model first.
The model will be trained to predict the following hormones:
    1. "filtered_dhea",
    2. "filtered_ert",
'''

# Define the hormones-M list
hormones_M = [
    "filtered_dhea",
    "filtered_ert",
]

# Check if the specified hormones are in the m_data columns
missing_hormones = [h for h in hormones_M if h not in m_data.columns]
if missing_hormones:
    print("Missing hormones in m_data:", missing_hormones)
else:
    print("All hormones are present in m_data columns.")
    
    
'''
Check the results, which is in .npy file
'''

# Simple load
arr = np.load("train_output/logs/dnn_metric.npy")
print(arr.shape)
print(arr.dtype)