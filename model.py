#!/usr/bin/env python3
# coding: utf-8

"""
Written by Katherine L. Bottenhorn, 
Based on code from Tong He & CBIG: 
https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/predict_phenotypes/He2022_MM/
"""
import lrp_captum
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.init as init
from sklearn.model_selection import train_test_split
import numpy as np
import os 
import random
import time
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from pathlib import Path
from typing import Optional
from ig_captum import explain_all_targets

class abcd_dataset(torch.utils.data.Dataset):
    """PyTorch dataset class

    Attributes:
        x (torch.Tensor): input features
        y (torch.Tensor): target values
    """

    def __init__(self, x, y, for_finetune=False):
        x = np.asarray(x)
        y = np.asarray(y)
        self.x = torch.from_numpy(x).float()
        if for_finetune:
            # ensure y has shape (N, 1)
            self.y = torch.from_numpy(y).float().view(-1, 1)
        else:
            self.y = torch.from_numpy(y).float()

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.x.size(0)


def torch_nanmean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean of x excluding positions where mask is True.
    """
    # count non-masked entries
    valid = ~mask
    count = valid.sum()
    if count == 0:
        return torch.tensor(float('nan'), device=x.device)
    total = x.masked_fill(mask, 0.0).sum()
    return total / count


def msenanloss(input: torch.Tensor, target: torch.Tensor,  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Mean Squared Error loss ignoring NaN targets.
    """
    # squared error
    se = (input - target) ** 2
    # default mask: where target is NaN
    if mask is None:
        mask = torch.isnan(target)
    return torch_nanmean(se, mask)


def mics_z_norm(train_y: np.ndarray,
                valid_y: np.ndarray,
                test_y: np.ndarray=None):
    """
    Z-normalize Y arrays based on training set stats.

    Returns:
        train_y_z, valid_y_z, test_y_z, t_sigma
    """
    # compute mean and std on training set, ignoring NaNs
    t_mu = np.nanmean(train_y, axis=0, keepdims=True)
    y_train = train_y - t_mu
    y_valid = valid_y - t_mu
    y_test = None
    if test_y is not None:
        y_test = test_y - t_mu

    t_sigma = np.nanstd(y_train, axis=0)
    # avoid division by zero
    t_sigma_safe = np.where(t_sigma == 0, 1.0, t_sigma)

    # reshape for broadcasting
    if y_train.ndim == 2:
        sigma_b = t_sigma_safe[np.newaxis, :]
    else:
        sigma_b = t_sigma_safe

    y_train_z = y_train / sigma_b
    y_valid_z = y_valid / sigma_b
    y_test_z = None
    if y_test is not None:
        y_test_z = y_test / sigma_b

    return y_train_z, y_valid_z, y_test_z, t_sigma_safe


def nanpearsonr(x, y):
    """
    Compute Pearson’s r between x and y, ignoring any pair where x or y is NaN.
    Returns a float (or an array of floats if x,y are 2D with shape [N, D]).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # handle 1-D case
    if x.ndim == 1:
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan
        return pearsonr(x[mask], y[mask])[0]

    # handle multi-output: compute per-column
    corrs = []
    for i in range(x.shape[1]):
        xi = x[:, i]
        yi = y[:, i]
        mask = ~np.isnan(xi) & ~np.isnan(yi)
        if mask.sum() < 2:
            corrs.append(np.nan)
        else:
            corrs.append(pearsonr(xi[mask], yi[mask])[0])
    return np.array(corrs)


def nancod_znormed(x, y):
    """
    Compute Coefficient of Determination (R²) between x and y, ignoring NaNs,
    *after* z-normalizing x (so you get variance-explained on a standardized scale).
    Returns a float or array of floats.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # 1-D case
    if x.ndim == 1:
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan
        x_clean = x[mask]
        y_clean = y[mask]
        # z-normalize x
        xz = (x_clean - x_clean.mean()) / x_clean.std()
        yz = (y_clean - y_clean.mean()) / y_clean.std()
        ss_res = np.sum((xz - yz) ** 2)
        ss_tot = np.sum((xz - xz.mean()) ** 2)
        return 1.0 - ss_res / ss_tot

    # multi-output: per column
    cods = []
    for i in range(x.shape[1]):
        xi = x[:, i]
        yi = y[:, i]
        mask = ~np.isnan(xi) & ~np.isnan(yi)
        if mask.sum() < 2:
            cods.append(np.nan)
        else:
            xi_c = xi[mask]; yi_c = yi[mask]
            xz = (xi_c - xi_c.mean()) / xi_c.std()
            yz = (yi_c - yi_c.mean()) / yi_c.std()
            ss_res = ((xz - yz) ** 2).sum()
            ss_tot = ((xz - xz.mean()) ** 2).sum()
            cods.append(1.0 - ss_res / ss_tot)
    return np.array(cods)

def print_result(corr, cod, mae, loss, epoch):
    msg = (
        f"Epoch {epoch:3d} — "
        f"Corr: {corr.mean():.3f}, "
        f"R²: {cod.mean():.3f}, "
        f"MAE: {mae.mean():.3f}, "
        f"Val Loss: {loss:.4f}"
    )
    print(msg)


def mics_log(model_str, out_dir, **records):
    """
    Save training records to disk for later analysis.
    `records` might include arrays like tra_los_record, val_cor_record, etc.
    """
    import os
    import numpy as np

    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    for name, arr in records.items():
        path = os.path.join(log_dir, f"{model_str}_{name}.npy")
        np.save(path, arr)
    print(f"Saved logs to {log_dir}")

def mics_infer_metric(dataloader, net, criterion, device,
                      t_sigma=None, need_value=False, output_size=1):
    '''performance inference with net on data from dataloader and calculate
        metric

    Args:
        dataloader: dataloader to load data for PyTorch framework
        net: PyTorch deep learning network
        criterion: criterion for loss calculation
        device: torch device indicate which GPU is running
        t_sigma (float, optional): std of training y data, only use if sex is
            not the behavioral measuers
        need_value (bool, optional): whether return record of real and
            predicted value
        output_size (int, optional): size of network output

    Returns:
        Tuple: if t_sigma is not None, correlation, R^2, MAE and loss are returned.
            If t_sigma is None, auccuracy and loss are returned. If need_value
            set to True, tuple returned also returns record of real and
            predicted y value alongside the metrics. If need_value is false,
            only metrics are returned.
    '''
    # initialize variable for record
    record_loss = 0.0
    
    # prepare placeholder shapes
    sample0, target0 = next(iter(dataloader))
    out_dim = net(sample0.to(device)).shape[-1]
    
    if t_sigma is None:
        record_correct = 0.0  # count of correct prediction
        record_total = 0.0  # count of total prediction
        record_real = np.zeros((0,))
        record_pred = np.zeros((0, out_dim))
    else:
        record_real = np.zeros((0, output_size))  # real value
        record_pred = np.zeros((0, output_size))  # prediction value

    # disable grads
    net.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = net(x)

            # apply mask if needed
            if t_sigma is None:
                loss = criterion(outputs, y)
            else:
                mask = torch.isnan(y)
                loss = criterion(outputs, y, mask=mask)

            record_loss += loss.item()
            record_real = np.concatenate((record_real, y.cpu().numpy()), axis=0)
            record_pred = np.concatenate((record_pred, outputs.cpu().numpy()), axis=0)

            if t_sigma is None:
                _, predicted = torch.max(outputs, dim=1)
                record_total += y.size(0)
                record_correct += (predicted == y).sum().item()

    avg_loss = record_loss / len(dataloader)

    if t_sigma is None:
        accuracy = record_correct / record_total
        if need_value:
            return accuracy, avg_loss, record_real, record_pred
        return accuracy, avg_loss

    # regression metrics
    corr = nanpearsonr(record_real, record_pred)
    cod  = nancod_znormed(record_real, record_pred)
    mae  = np.nanmean(np.abs(record_real - record_pred), axis=0) * t_sigma

    if need_value:
        return corr, cod, mae, avg_loss, record_real, record_pred
    return corr, cod, mae, avg_loss


class dnn_4l(nn.Module):
    '''4 layer DNN model

    Attributes:
        fc1 (torch.nn.Sequential): First layer of DNN
        fc2 (torch.nn.Sequential): Second layer of DNN
        fc3 (torch.nn.Sequential): Third layer of DNN
        fc4 (torch.nn.Sequential): Fourth layer of DNN
    '''

    def __init__(self, input_size, n_l1, n_l2, n_l3, dropout, output_size=1):
        """initialization of 4 layer DNN

        Args:
            input_size (int): dimension of input data
            n_l1 (int): number of node in first layer
            n_l2 (int): number of node in second layer
            n_l3 (int): number of node in third layer
            dropout (float): rate of dropout
            output_size (int, optional): dimension of output data
        """
        super(dnn_4l, self).__init__()
        # Input validation
        assert isinstance(input_size, int) and input_size > 0, "input_size must be a positive integer"
        assert isinstance(n_l1, int) and n_l1 > 0, "n_l1 must be a positive integer"
        assert isinstance(n_l2, int) and n_l2 > 0, "n_l2 must be a positive integer"
        assert isinstance(n_l3, int) and n_l3 > 0, "n_l3 must be a positive integer"
        assert isinstance(output_size, int) and output_size > 0, "output_size must be a positive integer"
        assert 0 <= dropout < 1, "dropout must be in [0, 1)"
        
        '''
        Discussion: Placing dropout before the linear layer is unusual. Typically, dropout is applied after the activation (e.g., ReLU) or between layers to prevent overfitting on the activations rather than the raw input. Applying dropout to the input might randomly zero out entire input features, which could harm performance, especially if the input is normalized or sparse.
        '''
        # self.fc1 = nn.Sequential(
        #     nn.Dropout(p=dropout),
        #     nn.Linear(input_size, n_l1),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(n_l1),
        # )
        # self.fc2 = nn.Sequential(
        #     nn.Dropout(p=dropout),
        #     nn.Linear(n_l1, n_l2),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(n_l2),
        # )
        # self.fc3 = nn.Sequential(
        #     nn.Dropout(p=dropout),
        #     nn.Linear(n_l2, n_l3),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(n_l3),
        # )
        # self.fc4 = nn.Sequential(
        #     nn.Dropout(p=dropout),
        #     nn.Linear(n_l3, output_size),
        # )
        
        # Modified Version
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, n_l1),
            nn.ReLU(),
            nn.BatchNorm1d(n_l1),
            nn.Dropout(p=dropout),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(n_l1, n_l2),
            nn.ReLU(),
            nn.BatchNorm1d(n_l2),
            nn.Dropout(p=dropout),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(n_l2, n_l3),
            nn.ReLU(),
            nn.BatchNorm1d(n_l3),
            nn.Dropout(p=dropout),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(n_l3, output_size),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x



def train(
    dset,
    features,
    phenotypes,
    runs: int = 1,
    epochs: int = 128,
    seed: int = 4,
    n_nodes: list = [64, 32, 16],
    out_dir: str = './train_output',
    across_dataset: bool = False,
    exp_dataset: bool = False,
) -> float:
    """
    Main training loop with early stopping. Returns best validation loss.
    """
    args = {
    "batch_size":      64,       # ~80 batches/epoch → good CPU/GIL utilization 
    "num_workers":     4,        # parallel data‐loads (macOS uses 'spawn', so keep modest)
    "lr":              1e-3,     # start with 0.001
    "momentum":        0.9,      # classic momentum for SGD
    "weight_decay":    1e-4,     # L2 regularization to combat overfitting
    "dropout":         0.5,      # fairly strong dropout since small data
    "scheduler_decrease": 40,    # drop LR at epoch 40 and 80 (with epochs=128)
    "gamma":           0.1       # LR → 1e-4 and then 1e-5
    }
    # apply any overrides from grid_search.py
    override = globals().get("ARGS_OVERRIDE", {})
    args.update(override)
    
    # reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = (device.type == "cuda")

    # === data split & z-norm y ===
    N = len(dset)
    idx_train, idx_test = train_test_split(
        np.arange(N), test_size=0.2, random_state=seed
    )
    X = dset[features].values
    Y = dset[phenotypes].values

    x_tr_raw, y_tr_raw = X[idx_train], Y[idx_train]
    x_te_raw, y_te_raw = X[idx_test],  Y[idx_test]
    idx_tr, idx_val = train_test_split(
        np.arange(len(x_tr_raw)), test_size=0.2, random_state=seed
    )
    x_tr, y_tr = x_tr_raw[idx_tr], y_tr_raw[idx_tr]
    x_val, y_val = x_tr_raw[idx_val], y_tr_raw[idx_val]

    # Track participant indices for each split
    original_train_idx = idx_train  # Original dataset indices for training split
    original_test_idx = idx_test    # Original dataset indices for test split
    # For validation: need to map back to original dataset indices
    original_val_idx = idx_train[idx_val]  # Original dataset indices for validation split
    original_train_final_idx = idx_train[idx_tr]  # Original dataset indices for final training split

    y_tr, y_val, y_te, t_sigma = mics_z_norm(y_tr, y_val, y_te_raw)

    # === DataLoaders ===
    print("→ about to create DataLoaders")
    trainloader = DataLoader(
        abcd_dataset(x_tr, y_tr),
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
        pin_memory=pin_memory
    )
    print("→ DataLoaders created successfully!")
    validLoader = DataLoader(
        abcd_dataset(x_val, y_val),
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
        pin_memory=pin_memory
    )
    testLoader = DataLoader(
        abcd_dataset(x_te_raw, y_te),
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
        pin_memory=pin_memory
    )

    # prepare logging arrays…
    n_phe = y_tr.shape[1]
    tra_los_record = np.zeros((runs, epochs))
    val_los_record = np.zeros((runs, epochs))
    tra_cor_record = np.zeros((runs, epochs, n_phe))
    val_cor_record = np.zeros((runs, epochs, n_phe))
    tra_cod_record = np.zeros((runs, epochs, n_phe))
    val_cod_record = np.zeros((runs, epochs, n_phe))
    tra_mae_record = np.zeros((runs, epochs, n_phe))
    val_mae_record = np.zeros((runs, epochs, n_phe))
    tes_res_record = np.zeros((runs, epochs, len(x_te_raw), n_phe))
    val_res_record = np.zeros((runs, epochs, len(x_val),    n_phe))

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model_dir = Path(out_dir) / 'trained_model_abcd'
    model_dir.mkdir(exist_ok=True)

    # early stopping
    patience = 10
    best_run_loss = float('inf')

    for run in range(runs):
        best_val_loss = float('inf')
        epochs_no_improve = 0

        net = dnn_4l(
            input_size=x_tr.shape[1],
            n_l1=n_nodes[0],
            n_l2=n_nodes[1],
            n_l3=n_nodes[2],
            dropout=args["dropout"],
            output_size=n_phe
        ).to(device)

        criterion = msenanloss
        optimizer = optim.SGD(
            net.parameters(),
            lr=args["lr"],
            momentum=args["momentum"],
            weight_decay=args["weight_decay"]
        )
        scheduler = MultiStepLR(
            optimizer,
            milestones=[args["scheduler_decrease"], args["scheduler_decrease"]*2],
            gamma=args["gamma"]
        )

        for epoch in range(epochs):
            net.train()
            train_loss = 0.0
            for xb, yb in trainloader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = net(xb)
                loss = criterion(out, yb, mask=torch.isnan(yb))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            tra_los_record[run, epoch] = train_loss / len(trainloader)

            # training metrics
            corr, cod, mae, _ = mics_infer_metric(
                trainloader, net, criterion, device, t_sigma, output_size=n_phe
            )
            tra_cor_record[run, epoch] = corr
            tra_cod_record[run, epoch] = cod
            tra_mae_record[run, epoch] = mae

            # validation
            corr, cod, mae, val_loss, _, val_pred = mics_infer_metric(
                validLoader, net, criterion, device, t_sigma,
                output_size=n_phe, need_value=True
            )
            val_cor_record[run, epoch] = corr
            val_cod_record[run, epoch] = cod
            val_mae_record[run, epoch] = mae
            val_los_record[run, epoch] = val_loss
            val_res_record[run, epoch] = val_pred.squeeze()
            print_result(corr, cod, mae, val_loss, epoch)

            # early stopping
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(net.state_dict(), model_dir/f'best_run{run}.pt')
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Run {run}: early stopping at epoch {epoch}")
                    break

            # test predictions
            _, _, _, _, _, tes_pred = mics_infer_metric(
                testLoader, net, criterion, device, t_sigma,
                output_size=n_phe, need_value=True
            )
            tes_res_record[run, epoch] = tes_pred.squeeze()

            scheduler.step()
            torch.save(net.state_dict(), model_dir/f'run{run}_epoch{epoch}.pt')

        # load best and explain
        net.load_state_dict(torch.load(model_dir/f'best_run{run}.pt'))
        explain_all_targets(
            net, validLoader, device,
            feature_names=features,
            out_dir=out_dir,
            run=run, steps=50, max_batches=50
        )
        best_run_loss = min(best_run_loss, best_val_loss)

    # final logging
    model_str = 'dnn'
    if across_dataset: model_str += '_across_dataset'
    if exp_dataset:    model_str += '_exp_dataset'

    # Create participant mapping info for easy reconstruction
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

    mics_log(
        model_str, out_dir,
        tra_los_record=tra_los_record,
        val_los_record=val_los_record,
        tra_cor_record=tra_cor_record,
        val_cor_record=val_cor_record,
        tra_cod_record=tra_cod_record,
        val_cod_record=val_cod_record,
        tra_mae_record=tra_mae_record,
        val_mae_record=val_mae_record,
        tes_res_record=tes_res_record,
        val_res_record=val_res_record,
        t_sigma=t_sigma,
        # Add participant index mappings
        original_test_idx=original_test_idx,
        original_val_idx=original_val_idx,
        original_train_final_idx=original_train_final_idx,
        participant_mapping=participant_mapping
    )

    return best_run_loss