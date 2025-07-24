# Fully-Connected 4-Layer DNN – Architecture & Training Report

**Date:** May 26, 2025  
**Author:** James

---

## 1. Hyper-parameters  

| Symbol / Arg               | Value                                                        | Rationale                                                                    |
|----------------------------|--------------------------------------------------------------|------------------------------------------------------------------------------|
| **`input_size`**           | 158                                                          | Number of engineered features per sample                                     |
| **Hidden sizes**           | `n_l1 = 64`, `n_l2 = 32`, `n_l3 = 16`                        | Pyramid shape balances expressiveness and overfitting risk                   |
| **`output_size`**          | *n*<sub>phe</sub> (number of hormones)                       | One output per hormone for multi-output regression                           |
| **Dropout `p`**            | 0.5                                                          | Strong regularization for small dataset                                       |
| **Batch size**             | 64                                                           | Good throughput on CPU/GPU                                                   |
| **Optimizer**              | SGD (lr = 1 × 10<sup>−3</sup>, momentum = 0.9, weight-decay = 1 × 10<sup>−4</sup>) | Classic momentum + L2                                                        |
| **LR scheduler**           | Multi-step: ×0.1 at epochs 40 & 80                           | Step-drop after stagnation                                                    |
| **Epochs**                 | 128                                                          | Enough to converge after two LR drops                                         |
| **Early stopping patience**| 10 epochs                                                    | Stop if validation loss does not improve for 10 consecutive epochs            |
| **Data split**             | 64 % / 16 % / 20 % (train/val/test)                          | Fixed random seed ensures reproducibility                                     |
| **Loader workers**         | 4                                                            | Parallel data loading on macOS/Windows                                        |

---

## 2. Model Architecture  

<div class="mermaid">
graph LR
    A["Input<br/>(158 features)"]
    B["FC 158→64<br/>ReLU → BN → Dropout 0.5"]
    C["FC 64→32<br/>ReLU → BN → Dropout 0.5"]
    D["FC 32→16<br/>ReLU → BN → Dropout 0.5"]
    E["FC 16→n_phe<br/>(Linear)"]

    A --> B
    B --> C
    C --> D
    D --> E

    style A fill:#dfe6fd,stroke:#4361ee
    style B fill:#b5e48c
    style C fill:#fee440
    style D fill:#ff9f1c
    style E fill:#f15bb5
</div>

**Figure 1** – 4-layer fully-connected DNN: each block = Linear → ReLU → BatchNorm1d → Dropout.

---

## 3. Data I/O  

| Stage                  | Shape                           | Notes                                            |
|------------------------|---------------------------------|--------------------------------------------------|
| **Raw X**              | `(N, 158)`                      | Float features (optionally standardized)         |
| **Raw y**              | `(N, n_phe)`                    | Each column = a hormone outcome                  |
| **Train/Val/Test**     | 64 % / 16 % / 20 %              | `train_test_split` twice                         |
| **Batch**              | `(64, 158)` → … → `(64, n_phe)`  | DataLoader w/ `pin_memory` for CUDA transfers    |

Normalization:  
- y targets z-normalized per hormone (µ, σ from train).  
- MAE later rescaled by σ for original units.

---

## 4. Validation, Logging & Early Stopping  

Each epoch on validation produces:

| Metric        | Function         | Meaning                                         |
|---------------|------------------|-------------------------------------------------|
| **Val Loss**  | `msenanloss`     | Masked MSE over validation set                  |
| **Pearson r** | `nanpearsonr`    | Linear correlation, NaN-safe                    |
| **R² (COD)**  | `nancod_znormed` | Variance explained after z-normalization        |
| **MAE**       | `nanmean(|ŷ−y|)×σ` | Rescaled mean absolute error in original units  |

**Early stopping:** stops training if **validation loss** fails to improve for **10** consecutive epochs; best weights saved for each run.

**Checkpoints & Logs:**  
- **Model snapshots:** `./train_output/trained_model_abcd/run{run}_epoch{epoch}.pt`  
- **Best model:** `.../trained_model_abcd/best_run{run}.pt`  
- **Logs:** `./train_output/logs/{model_str}_{record}.npy`

---

## 5. Explainability via Integrated Gradients  

We use Captum’s **IntegratedGradients** to attribute input features:

- **Baseline:** zero vector, broadcasted to batch  
- **Steps:** 50 Riemann steps for path integral  
- **Target:** output neuron index per hormone  

```python
from captum.attr import IntegratedGradients

ig = IntegratedGradients(net)
attr = ig.attribute(
    inputs=x,
    baselines=torch.zeros((1, x.size(1))).to(device),
    target=target_index,
    n_steps=50
)
```

**Output:** `./train_output/feature_importance_ig/run{run}_target{target}_relevance.npy`

---

## 6. Participant Mapping & Index Tracking

**Problem:** The training logs save predictions (`tes_res_record`, `val_res_record`) but without participant IDs, making it impossible to match predictions back to individual participants.

**Solution:** Added participant index tracking to maintain the mapping between prediction arrays and original dataset rows.

### What's Added to model.py:

1. **Index tracking variables** (lines 438-443):
   ```python
   # Track participant indices for each split
   original_train_idx = idx_train  # Original dataset indices for training split
   original_test_idx = idx_test    # Original dataset indices for test split
   original_val_idx = idx_train[idx_val]  # Original dataset indices for validation split
   original_train_final_idx = idx_train[idx_tr]  # Original dataset indices for final training split
   ```

2. **Participant mapping dictionary** (lines 588-600):
   ```python
   participant_mapping = {
       'test_indices': original_test_idx,          # Maps tes_res_record[run, epoch, i] -> original_test_idx[i]
       'val_indices': original_val_idx,            # Maps val_res_record[run, epoch, i] -> original_val_idx[i]
       'train_final_indices': original_train_final_idx,  # Final training set indices
       'split_info': {
           'total_participants': N,
           'train_size': len(original_train_final_idx),
           'val_size': len(original_val_idx),
           'test_size': len(original_test_idx),
           'random_seed': seed
       }
   }
   ```

3. **Saved to logs**: All index mappings are saved as `.npy` files in `./train_output/logs/`:
   - `original_test_idx.npy`: Maps test predictions to original dataset rows
   - `original_val_idx.npy`: Maps validation predictions to original dataset rows
   - `original_train_final_idx.npy`: Maps training predictions to original dataset rows
   - `participant_mapping.npy`: Complete mapping information

4. **Training DataLoader**: Changed `shuffle=False` to maintain index order for participant tracking.

### How to Use Participant Mapping:

#### Step 1: Load your original dataset
```python
import pandas as pd
import numpy as np

# Load original dataset
df = pd.read_pickle('synthetic_dset-M.pkl')  # or synthetic_dset_F.pkl
```

#### Step 2: Read the mapping files
```python
# Read participant mapping
test_indices = np.load('./train_output/logs/original_test_idx.npy')
val_indices = np.load('./train_output/logs/original_val_idx.npy')
participant_mapping = np.load('./train_output/logs/participant_mapping.npy', allow_pickle=True).item()

# Read prediction results
test_predictions = np.load('./train_output/logs/dnn_tes_res_record.npy')  # Shape: (runs, epochs, n_test, n_hormones)
val_predictions = np.load('./train_output/logs/dnn_val_res_record.npy')   # Shape: (runs, epochs, n_val, n_hormones)
```

#### Step 3: Match predictions to participants
```python
# Example: Get test predictions for run 0, epoch 50
run_id = 0
epoch_id = 50
test_pred = test_predictions[run_id, epoch_id]  # Shape: (n_test, n_hormones)

# Match to original participants
test_participant_data = df.iloc[test_indices]  # Original participant data
test_participant_data['predicted_dhea'] = test_pred[:, 0]  # Add predictions
test_participant_data['predicted_ert'] = test_pred[:, 1]

# Now you can analyze by participant characteristics
print("Test predictions with participant info:")
print(test_participant_data[['site_id_l', 'filtered_dhea', 'predicted_dhea', 'filtered_ert', 'predicted_ert']].head())
```

### Using read_npy.py for Participant Mapping:

```bash
# List all available log files
python read_npy.py list ./train_output/logs

# Check participant mapping structure
python read_npy.py ./train_output/logs/participant_mapping.npy

# Check test indices
python read_npy.py ./train_output/logs/original_test_idx.npy

# Check prediction results
python read_npy.py ./train_output/logs/dnn_tes_res_record.npy
```

The `read_npy.py` script will show you:
- Array shapes and dimensions
- Sample data for understanding structure
- Interactive options to explore specific slices
- Ability to save as text files for external analysis

---
