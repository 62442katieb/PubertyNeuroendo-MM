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