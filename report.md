# Fully-Connected 4-Layer DNN – Architecture & Training Report

**Date:** May 6, 2025
**Author:** James

---

## 1. Hyper-parameters

| Symbol / Arg         | Value                                                        | Rationale                                                                                 |
| -------------------- | ------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| \`\`                 | 158                                                          | Number of engineered features in each sample                                              |
| **Hidden sizes**     | `n_l1 = 64`, `n_l2 = 32`, `n_l3 = 16`                        | Pyramid shape gives good expressiveness without over-parameterising a 5 k-sample data set |
| \`\`                 | *n*phe (= number of hormones)                                | Multi-output regression – one output per hormone                                          |
| \*\*Dropout \*\*\`\` | 0.5                                                          | Aggressive regularisation for small dataset                                               |
| **Batch size**       | 64                                                           | Keeps CPU busy while fitting in cache/RAM                                                 |
| **Optimizer**        | SGD (lr = 1 × 10−3, momentum = 0.9, weight-decay = 1 × 10−4) | Classic recipe; weight-decay ≈ L2                                                         |
| **LR scheduler**     | Multi-step ↓×0.1 at epochs 40 & 80                           | Step-decay after early “plateau”                                                          |
| **Epochs**           | 128                                                          | Enough to converge after two LR drops                                                     |
| **Data split**       | 64 % train / 16 % val / 20 % test                            | Standard practice with fixed random seed                                                  |
| **Loader workers**   | 4                                                            | Parallel I/O on macOS/CPU                                                                 |

---

## 2. Model Architecture

<div class="mermaid">
graph LR
    %% Define nodes with quoted labels so HTML tags work
    A["Input<br/>(158 features)"]
    B["FC 158 → 64<br/>ReLU / BN / Dropout 0.5"]
    C["FC 64 → 32<br/>ReLU / BN / Dropout 0.5"]
    D["FC 32 → 16<br/>ReLU / BN / Dropout 0.5"]
    E["FC 16 → n_phe<br/>(Linear)"]

    %% Now connect them
    A --> B
    B --> C
    C --> D
    D --> E

    %% Styling
    style A fill:#dfe6fd,stroke:#4361ee
    style B fill:#b5e48c
    style C fill:#fee440
    style D fill:#ff9f1c
    style E fill:#f15bb5
</div>

**Figure 1** – Forward pass through 4 fully-connected layers.

* **BatchNorm1d** stabilises activations.
* **Dropout** applied after activation to regularise.

### Loss & Optimiser

* **Loss:** `msenanloss` → mean-squared-error that ignores NaNs in the target (`torch.nanmean`).
* **Optimizer:** SGD + momentum + weight-decay.
* **Scheduler:** `MultiStepLR` stepping down LR by 0.1 at epochs 40 & 80.

---

## 3. Data I/O

| Stage                  | Shape                           | Notes                                            |
| ---------------------- | ------------------------------- | ------------------------------------------------ |
| **Raw X**              | `(N, 158)`                      | Float features (optionally standardised)         |
| **Raw y**              | `(N, n_phe)`                    | Each column = one hormone                        |
| **Train / Val / Test** | 64 % / 16 % / 20 %              | `train_test_split` twice with fixed seed         |
| **Batch (train)**      | `(64, 158)` → … → `(64, n_phe)` | DataLoader with `pin_memory` when CUDA available |

**Normalization:**

* Targets **y** are z-normalised per hormone (μ, σ from training set).
* MAE is later rescaled by σ to report in original units.

---

## 4. Validation & Logging

**Metrics computed on validation set each epoch:**

| Metric        | Function         | Meaning                                  |         |                                              |
| ------------- | ---------------- | ---------------------------------------- | ------- | -------------------------------------------- |
| **Val Loss**  | `msenanloss`     | MSE over validation set (masking NaNs)   |         |                                              |
| **Pearson r** | `nanpearsonr`    | Linear correlation, NaN-safe             |         |                                              |
| **R² (COD)**  | `nancod_znormed` | Variance explained after z-normalisation |         |                                              |
| **MAE**       | \`nanmean(       | ŷ – y                                    | ) × σ\` | Mean absolute error in original target units |

**Checkpoints & Logs:**

* **Model snapshots:**
  `./train_output/trained_model_abcd/run{run}_epoch{epoch}.pt`
* **Logs:** NumPy arrays (`.npy`) under
  `./train_output/logs/{model_str}_{record_name}.npy`

**Early-stop / Selection:**
Choose the epoch with highest **mean R²** on validation.

---
