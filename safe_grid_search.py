import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import subprocess
import sys
        
# =============================================================================
# Modified In-Process Grid Search (Safer)
# =============================================================================

def grid_search():
    """
    Safer in-process grid search with proper cleanup and error handling.
    """
    import torch
    import gc
    
    # Load data once
    with open('synthetic_dset-M.pkl', 'rb') as f:
        m_data = pickle.load(f)
    
    # Load variable names
    with open('variable_names.txt', 'r') as file:
        content = file.read()
    
    # Parse variable names (your existing code)
    list_definitions = content.split('\n\n')
    input_features = []
    hormones_M = []
    
    for definition in list_definitions:
        if 'input_features = [' in definition:
            items = definition.split('[')[1].split(']')[0].strip()
            input_features = [item.strip(' ",') for item in items.split('\n') if item.strip(' ",')]
        elif 'hormones-M = [' in definition:
            items = definition.split('[')[1].split(']')[0].strip()
            hormones_M = [item.strip(' ",') for item in items.split('\n') if item.strip(' ",')]
    
    # Parameter combinations
    param_combinations = [
        # n_nodes=[64,32,16] combinations
        {"n_nodes": [64,32,16], "batch_size": 64, "lr": 1e-3, "dropout": 0.3},
        {"n_nodes": [64,32,16], "batch_size": 64, "lr": 1e-3, "dropout": 0.5},
        {"n_nodes": [64,32,16], "batch_size": 64, "lr": 5e-4, "dropout": 0.3},
        {"n_nodes": [64,32,16], "batch_size": 64, "lr": 5e-4, "dropout": 0.5},
        {"n_nodes": [64,32,16], "batch_size": 32, "lr": 1e-3, "dropout": 0.3},
        {"n_nodes": [64,32,16], "batch_size": 32, "lr": 1e-3, "dropout": 0.5},
        {"n_nodes": [64,32,16], "batch_size": 32, "lr": 5e-4, "dropout": 0.3},
        {"n_nodes": [64,32,16], "batch_size": 32, "lr": 5e-4, "dropout": 0.5},
        
        # n_nodes=[128,64,32] combinations
        {"n_nodes": [128,64,32], "batch_size": 64, "lr": 1e-3, "dropout": 0.3},
        {"n_nodes": [128,64,32], "batch_size": 64, "lr": 1e-3, "dropout": 0.5},
        {"n_nodes": [128,64,32], "batch_size": 64, "lr": 5e-4, "dropout": 0.3},
        {"n_nodes": [128,64,32], "batch_size": 64, "lr": 5e-4, "dropout": 0.5},
        {"n_nodes": [128,64,32], "batch_size": 32, "lr": 1e-3, "dropout": 0.3},
        {"n_nodes": [128,64,32], "batch_size": 32, "lr": 1e-3, "dropout": 0.5},
        {"n_nodes": [128,64,32], "batch_size": 32, "lr": 5e-4, "dropout": 0.3},
        {"n_nodes": [128,64,32], "batch_size": 32, "lr": 5e-4, "dropout": 0.5},
    ]
    
    print(f"=> Testing {len(param_combinations)} hyper-parameter combinations\n")
    results = []
    
    for i, params in enumerate(param_combinations):
        print(f"\n=== Combination {i+1}/{len(param_combinations)} ===")
        print(f"Parameters: {params}")
        
        try:
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Import fresh copy of model
            import importlib
            import model
            importlib.reload(model)
            
            # Set parameters
            model.ARGS_OVERRIDE = {
                "batch_size": params["batch_size"],
                "lr": params["lr"],
                "dropout": params["dropout"],
                "num_workers": 0,  # Always 0 for safety
            }
            
            # Run training
            val_loss = model.train(
                dset=m_data,
                features=input_features,
                phenotypes=hormones_M,
                runs=1,
                epochs=2,  # Reduced for faster testing
                seed=42,
                n_nodes=params["n_nodes"],
                out_dir=f"./gs_output/combo_{i}",
                across_dataset=False,
                exp_dataset=False
            )
            
            results.append({"params": params, "val_loss": val_loss})
            print(f"✅ Success! Val loss: {val_loss:.5f}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save and display results
    if results:
        results_sorted = sorted(results, key=lambda x: x["val_loss"])
        with open("grid_search_results.json", "w") as f:
            json.dump(results_sorted, f, indent=2)
        
        print(f"\n🏆 Best parameters: {results_sorted[0]['params']}")
        print(f"🏆 Best val loss: {results_sorted[0]['val_loss']:.5f}")

        
if __name__ == "__main__":
    grid_search()