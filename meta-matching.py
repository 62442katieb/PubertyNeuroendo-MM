#!/usr/bin/env python3
# coding: utf-8

"""
Written by Katherine L. Bottenhorn, 
Based on code from Tong He & CBIG: 
https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/predict_phenotypes/He2022_MM/
"""

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.init as init
from sklearn.model_selection import train_test_split

class abcd_dataset(torch.utils.data.Dataset):
    """PyTorch dataset class

    Attributes:
        x (torch.tensor): tensor for x data
        y (torch.tensor): tensor for y data
    """

    def __init__(self, x, y, for_finetune=False):
        """initialization of PyTorch dataset class

        Args:
            x (ndarray): x data
            y (ndarray): y data
            for_finetune (bool, optional): whether the network is used for
                finetune
        """
        self.x = torch.from_numpy(x).float()
        if for_finetune:
            self.y = torch.from_numpy(y).float().view(-1, 1)
        else:
            self.y = torch.from_numpy(y).float()

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x, y

    def __len__(self):
        return int(self.x.shape[0])

def torch_nanmean(x, mask):
    '''Calculate mean and omit NAN

    Args:
        x (torch.tensor): input data
        mask (torch.tensor, optional): mask indicated NAN

    Returns:
        torch.Tensor: mean value (omit NAN)
    '''
    num = torch.where(mask, torch.full_like(x, 0), torch.full_like(x, 1)).sum()
    value = torch.where(mask, torch.full_like(x, 0), x).sum()
    return value / num


def msenanloss(input, target, mask=None):
    '''Calculate MSE (mean absolute error) and omit NAN

    Args:
        input (torch.tensor): predicted value
        target (torch.tensor): original value
        mask (torch.tensor, optional): mask indicated NAN

    Returns:
        torch.Tensor: MSE loss (omit NAN)
    '''
    ret = (input - target)**2
    if mask is None:
        mask = torch.isnan(ret)
    return torch_nanmean(ret, mask)

def mics_z_norm(train_y, valid_y, test_y=None):
    '''z normalize y of training, validation and test set based on training set

    Args:
        train_y (ndarray): training y data
        valid_y (ndarray): validation y data
        test_y (ndarray, optional): testing y data

    Returns:
        Tuple: contains z-normed y data and std of training y data
    '''
    # subtract mean of y of training set
    t_mu = np.nanmean(train_y, axis=0, keepdims=True)
    train_y = train_y - t_mu
    valid_y = valid_y - t_mu
    if test_y:
        test_y = test_y - t_mu

    # divide std of y of training set
    t_sigma = np.nanstd(train_y, axis=0)
    if train_y.ndim == 2:
        t_sigma_d = t_sigma[np.newaxis, :]
    else:
        t_sigma_d = t_sigma
        if t_sigma == 0:
            print('t_sigma is 0, pass divide std')
            return [train_y, valid_y, test_y, t_sigma]
    train_y = train_y / t_sigma_d
    valid_y = valid_y / t_sigma_d
    if test_y:
        test_y = test_y / t_sigma_d

    # return processed y and std for future MAE calculation
    return [train_y, valid_y, test_y, t_sigma]

def mics_infer_metric(dataloader,
                      net,
                      criterion,
                      device,
                      t_sigma=None,
                      need_value=False,
                      output_size=1):
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
        Tuple: if t_sigma is not None, correlation, MAE and loss are returned.
            If t_sigma is None, auccuracy and loss are returned. If need_value
            set to True, tuple returned also returns record of real and
            predicted y value alongside the metrics. If need_value is false,
            only metrics are returned.
    '''
    # initialize variable for record
    record_loss = 0.0
    if t_sigma is None:
        record_correct = 0.0  # count of correct prediction
        record_total = 0.0  # count of total prediction
        record_real = np.zeros((0))
        record_pred = np.zeros((0, 2))
    else:
        record_real = np.zeros((0, output_size))  # real value
        record_pred = np.zeros((0, output_size))  # prediction value

    # perform inference
    for (x, y) in dataloader:
        x, y = x.to(device), y.to(device)
        outputs = net(x)
        loss = criterion(outputs, y)
        record_loss += loss.item()
        record_real = np.concatenate((record_real, y.data.cpu().numpy()),
                                     axis=0)
        record_pred = np.concatenate((record_pred, outputs.data.cpu().numpy()),
                                     axis=0)
        if t_sigma is None:
            _, predicted = torch.max(outputs.data, 1)
            record_total += y.size(0)
            record_correct += (predicted == y.data).sum()

    # metric calculation
    loss = record_loss / len(dataloader)
    if t_sigma is None:
        aucc = record_correct.to(torch.float) / record_total
        if need_value:
            return aucc, loss, record_real, record_pred
        else:
            return aucc, loss
    else:
        corr = nanpearsonr(record_real, record_pred)
        cod = nancod_znormed(record_real, record_pred)
        mae = np.nanmean(np.abs(record_real - record_pred), 0) * t_sigma
        if need_value:
            return corr, cod, mae, loss, record_real, record_pred
        else:
            return corr, cod, mae, loss


class dnn_4l(nn.Module):
    '''4 layer DNN model

    Attributes:
        fc1 (torch.nn.Sequential): First layer of DNN
        fc2 (torch.nn.Sequential): Second layer of DNN
        fc3 (torch.nn.Sequential): Third layer of DNN
        fc4 (torch.nn.Sequential): Fourth layer of DNN
    '''

    def __init__(self, input_size, n_l1, n_l2, n_l3, dropout, output_size=1):
        """initialization of 3 layer DNN

        Args:
            input_size (int): dimension of input data
            n_l1 (int): number of node in first layer
            n_l2 (int): number of node in second layer
            n_l3 (int): number of node in third layer
            dropout (float): rate of dropout
            output_size (int, optional): dimension of output data
        """
        super(dnn_4l, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(input_size, n_l1),
            nn.ReLU(),
            nn.BatchNorm1d(n_l1),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_l1, n_l2),
            nn.ReLU(),
            nn.BatchNorm1d(n_l2),
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_l2, n_l3),
            nn.ReLU(),
            nn.BatchNorm1d(n_l3),
        )
        self.fc4 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(n_l3, output_size),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Conv1d):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x



def train(
    dset, input, phenotypes, runs=10, epochs=128, seed=4,
    n_nodes=[64, 32, 32, 32], lr=0.01, momentum=0.9, weight_decay=0.02,
    dropout=0.5, scheduler_decrease=75,
    out_dir='./train_output'
):
    '''main function for DNN network

    Args:
        dset : pandas DataFrame with `input` and `phenotype` as columns, participants as rows
        input : list or regex, features used as input for DNN, present as columns in `dset`
        phenotypes : list, outcomes/phenotypes predicted by DNN, present as columns in `dset` 
        runs : int, 
        epochs : int,
        seed : int, sets random state
        n_nodes : list of ints, len(layers) number of nodes per layer

    Returns:
        None
    '''
    args = {

    }
    t_overall = time.time()
    print('\nMM DNN for ABCD Study data with arguments: ' + str(args))

    # set all the seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # set gpu number
    os.environ["CUDA_VISIBLE_DEVICES"] = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_phenotype = phenotypes[:-1]
    test_phenotype = phenotypes[-1]
    # ABCD Study using stratified kfold?
    split_train, split_test = train_test_split(
        np.arange(dset[features].values.shape[0]), test_size=0.2, random_state=seed)
    x_train_raw = dset[features].values[split_train,:]
    y_train_raw = dset[phenotype].values[split_train,:]

    x_test_raw = dset[features].values[split_test,:]
    y_test_raw = dset[phenotype].values[split_test,:]

    split_tra, split_val = train_test_split(
        np.arange(x_train_raw.shape[0]), test_size=0.2, random_state=seed)

    x_train = x_train_raw[split_tra, :]
    x_valid = x_train_raw[split_val, :]
    y_train = y_train_raw[split_tra, :]
    y_valid = y_train_raw[split_val, :]

    # z norm based on y_train
    y_train, y_valid, y_test, t_sigma = mics_z_norm(y_train, y_valid, y_test_raw)

    # since we perform z normalization before, and we not need MAE now
    # so we use ones for t_sigma that we might need it in the future
    n_phe = y_train.shape[1]

    # load dataset for PyTorch
    dset_train = abcd_dataset(x_train, y_train)
    trainloader = DataLoader(
        dset_train, batch_size=args.batch_size, shuffle=True, num_workers=8)
    dset_valid = abcd_dataset(x_valid, y_valid)
    validLoader = DataLoader(
        dset_valid, batch_size=args.batch_size, shuffle=True, num_workers=8)
    dset_test = abcd_dataset(x_test, y_test)
    testLoader = DataLoader(
        dset_test, batch_size=args.batch_size, shuffle=False, num_workers=8)

    #runs = args.runs  # numbers of ensemble runs
    #epochs = args.epochs  # numbers of epochs per run

    # initialization of result record
    tra_los_record = np.zeros((runs, epochs))
    val_los_record = np.zeros((runs, epochs))
    tra_cor_record = np.zeros((runs, epochs, n_phe))
    val_cor_record = np.zeros((runs, epochs, n_phe))
    tra_cod_record = np.zeros((runs, epochs, n_phe))
    val_cod_record = np.zeros((runs, epochs, n_phe))
    tra_mae_record = np.zeros((runs, epochs, n_phe))
    val_mae_record = np.zeros((runs, epochs, n_phe))
    tes_res_record = np.zeros((runs, epochs, x_test.shape[0], n_phe))
    val_res_record = np.zeros((runs, epochs, x_valid.shape[0], n_phe))
    final_original = None

    os.makedirs(out_dir, exist_ok=True)
    dir_model = os.path.join(out_dir, 'trained_model_abcd')
    os.makedirs(dir_model, exist_ok=True)
    dir_model_temp = os.path.join(dir_model, 'dnn_model_save_base_exp_dataset')
    os.makedirs(dir_model_temp, exist_ok=True)

    # Code running - with multiple ensemble runs
    for run in range(runs):

        net = dnn_4l(
            x_train.shape[1],
            n_nodes[0],
            n_nodes[1],
            n_nodes[2],
            dropout,
            output_size=n_phe)
       
        net.to(device)

        # other components of network
        criterion = msenanloss
        optimizer = optim.SGD(
            net.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay)
        scheduler = MultiStepLR(
            optimizer,
            milestones=[scheduler_decrease, scheduler_decrease * 2],
            gamma=0.1)

        # start epoch training
        for epoch in range(epochs):

            # training
            train_loss = 0.0
            net.train(True)
            for (x, y) in trainloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = net(x)
                mask = torch.isnan(y)
                y.masked_fill_(mask, 0)
                loss = criterion(outputs, y, mask=mask)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            tra_los_record[run, epoch] = train_loss / len(trainloader)

            net.train(False)
            # Training
            corr, cod, mae, _ = mics_infer_metric(
                trainloader,
                net,
                criterion,
                device,
                t_sigma,
                output_size=n_phe)
            tra_cor_record[run, epoch, :] = corr
            tra_cod_record[run, epoch, :] = cod
            tra_mae_record[run, epoch, :] = mae

            # validation
            corr, cod, mae, loss, _, pred = mics_infer_metric(
                validLoader,
                net,
                criterion,
                device,
                t_sigma,
                output_size=n_phe,
                need_value=True)
            val_cor_record[run, epoch, :] = corr
            val_cod_record[run, epoch, :] = cod
            val_mae_record[run, epoch, :] = mae
            val_los_record[run, epoch] = loss
            val_res_record[run, epoch, :, :] = np.squeeze(pred)
            print_result(
                tra_cor_record[run, epoch, :], tra_cod_record[run, epoch, :],
                tra_mae_record[run, epoch, :], loss, corr, cod, mae, epoch)

            record_pred = np.zeros((0, n_phe))  # prediction value
            for (x, y) in testLoader:
                x, y = x.to(device), y.to(device)
                outputs = net(x)
                record_pred = np.concatenate(
                    (record_pred, outputs.data.cpu().numpy()), axis=0)
            tes_res_record[run, epoch, :, :] = np.squeeze(record_pred)

            file_trained = os.path.join(
                dir_model_temp, 'CBIG_ukbb_dnn_run_' + str(run) + '_epoch_' +
                str(epoch) + '.pkl_torch')
            torch.save(net, file_trained)

            scheduler.step()

    log_args = {
        'tra_los_record': tra_los_record,
        'val_los_record': val_los_record,
        'tra_cor_record': tra_cor_record,
        'val_cor_record': val_cor_record,
        'tra_cod_record': tra_cod_record,
        'val_cod_record': val_cod_record,
        'tra_mae_record': tra_mae_record,
        'val_mae_record': val_mae_record,
        'tes_res_record': tes_res_record,
        'final_original': final_original,
        'val_res_record': val_res_record,
        't_sigma': t_sigma
    }

    model_str = 'dnn'
    if args.across_dataset:
        model_str += '_across_dataset'
    if args.exp_dataset:
        model_str += '_exp_dataset'
    mics_log(
        model_str,
        out_dir,
        metric=args.metric,
        index=args.index,
        **log_args)
    print("time spent: {:.4f}".format(time.time() - t_overall))

    return

