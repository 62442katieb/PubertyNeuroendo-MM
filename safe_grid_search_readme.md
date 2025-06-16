# Safe Grid Search for Neural Network Hyperparameters

## Overview

This documentation explains the **safe_grid_search.py** script for hyperparameter optimization of deep neural networks.

## File Structure

```
your_project/
├── model.py                    # Your training module
├── synthetic_dset-M.pkl        # Training data
├── variable_names.txt          # Feature and target variable names
├── safe_grid_search.py        # Safe version (use this)
└── gs_output/                 # Output directory for results
    ├── combo_1/
    ├── combo_2/
    └── ...
```

## Usage

### Prerequisites

Ensure you have the following files in your working directory:
- `model.py` - Contains the `train()` function
- `synthetic_dset-M.pkl` - Your training dataset
- `variable_names.txt` - Variable definitions

### Running the Script

```bash
python safe_grid_search.py
```

The script will:
1. Load your data and parse variable names
2. Test all 16 hyperparameter combinations
3. Save results to `grid_search_results.json`
4. Display the best configuration

## Parameter Grid

The script tests all combinations of:

```python
param_grid = {
    "n_nodes":    [[64,32,16], [128,64,32]],    # Network architectures
    "batch_size": [64, 32],                      # Batch sizes
    "lr":         [1e-3, 5e-4],                 # Learning rates
    "dropout":    [0.3, 0.5]                    # Dropout rates
}
```

**Total combinations: 2 × 2 × 2 × 2 = 16**

## Customization Guide

### 1. Modify Hyperparameters

To change the parameter grid, edit the `param_combinations` list in `safe_grid_search.py`:

```python
param_combinations = [
    # Add your custom combinations here
    {"n_nodes": [128,64,32], "batch_size": 128, "lr": 1e-4, "dropout": 0.2},
    {"n_nodes": [256,128,64], "batch_size": 64, "lr": 5e-4, "dropout": 0.4},
    # ... more combinations
]
```

### 2. Adjust Training Parameters

Modify training settings in the `model.train()` call:

```python
val_loss = model.train(
    dset=m_data,
    features=input_features,
    phenotypes=hormones_M,
    runs=1,
    epochs=100,  # ← Change number of epochs
    seed=42,     # ← Change random seed
    out_dir=str(combo_dir),
    across_dataset=False,
    exp_dataset=False
)
```

### 3. Change Target Variables

The script currently targets male hormones (`hormones_M`). To switch to female hormones:

```python
# In the variable parsing section, use:
phenotypes = hormones_F  # Instead of hormones_M

# And load the corresponding dataset:
with open('synthetic_dset-F.pkl', 'rb') as f:  # Instead of synthetic_dset-M.pkl
    data = pickle.load(f)
```

### 4. Add New Hyperparameters

To test additional parameters, add them to your combinations:

```python
param_combinations = [
    {
        "n_nodes": [64,32,16], 
        "batch_size": 64, 
        "lr": 1e-3, 
        "dropout": 0.3,
        "weight_decay": 1e-4,     # ← New parameter
        "momentum": 0.9           # ← New parameter
    },
    # ... more combinations
]
```

Then add them to `ARGS_OVERRIDE`:

```python
model.ARGS_OVERRIDE = {
    "batch_size": params["batch_size"],
    "lr": params["lr"],
    "dropout": params["dropout"],
    "weight_decay": params["weight_decay"],    # ← Add new parameters
    "momentum": params["momentum"],
    "n_nodes": params["n_nodes"],
    "num_workers": 0,  # Keep this at 0!
}
```

## Output

### Console Output
```
=== Combination 1/16 ===
Parameters: {'n_nodes': [64, 32, 16], 'batch_size': 64, 'lr': 0.001, 'dropout': 0.3}
✅ Success! Val loss: 0.45231

...

============================================================
🏆 GRID SEARCH COMPLETE!
🏆 Best hyper-parameters:
{
  "n_nodes": [128, 64, 32],
  "batch_size": 32,
  "lr": 0.0005,
  "dropout": 0.3
}
🏆 Best validation loss: 0.38745
============================================================
```

### JSON Results File
Results are saved to `grid_search_results.json`:

```json
[
  {
    "config": {
      "n_nodes": [128, 64, 32],
      "batch_size": 32,
      "lr": 0.0005,
      "dropout": 0.3
    },
    "best_val_loss": 0.38745
  },
  ...
]
```

## Important Notes

### ⚠️ Critical Requirements

1. **Always keep `num_workers=0`** - This is essential to prevent deadlocks
2. **Don't use `ParameterGrid`** - Manual parameter lists are more stable
3. **Ensure sufficient disk space** - Each combination creates output files

### 🔧 Troubleshooting

**Memory Issues:**
- Reduce batch sizes in your parameter combinations
- Decrease the number of epochs for initial testing
- Monitor GPU memory usage

**Import Errors:**
- Ensure `model.py` is in the same directory
- Check that all required packages are installed
- Verify your `model.py` has a `train()` function

**File Not Found:**
- Confirm `synthetic_dset-M.pkl` and `variable_names.txt` exist
- Check file paths and permissions

