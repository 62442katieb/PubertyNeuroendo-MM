#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from os.path import join

# ugh why d I always make things harder than they need t be
import torch
from torch import nn
from torch.autograd import Variable


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#### based on example from https://chrisorm.github.io/VAE-pyt.html

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.en1 = nn.Linear(input_size, 200)
        self.en_mu = nn.Linear(200, representation_size)
        self.en_std = nn.Linear(200, representation_size)
        self.de1 = nn.Linear(representation_size, 200)
        self.de2 = nn.Linear(200, input_size)
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        self.identity = nn.Identity()
        
    def encode(self, x):
        """Encode a batch of samples, and return posterior parameters for each point."""
        h1 = self.relu(self.en1(x))
        return self.en_mu(h1), self.en_std(h1)
    
    def decode(self, z):
        """Decode a batch of latent variables"""
        
        h2 = self.relu(self.de1(z))
        #return self.sigmoid(self.de2(h2))
        return self.identity(self.de2(h2))
    
    def reparam(self, mu, logvar):
        """Reparameterisation trick to sample z values. 
        This is stochastic during training,  and returns the mode during evaluation."""
        
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
            
    
    def forward(self, x):
        """Takes a batch of samples, encodes them, and then decodes them again to compare."""
        mu, logvar = self.encode(x.view(-1, input_size))
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar
    
    def loss(self, reconstruction, x, mu, logvar):
        """MSELoss bc entries of x are not binary variables, so prev ELBO was inappropriate."""
        msel = nn.MSELoss()
        output = msel(reconstruction, x)
        #bce = torch.nn.functional.binary_cross_entropy(reconstruction, x.view(-1, input_size))
        #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        #KLD /= x.view(-1, input_size).data.shape[0] * input_size
        return output
        #return bce + KLD
    
    def get_z(self, x):
        """Encode a batch of data points, x, into their z representations."""
        
        mu, logvar = self.encode(x.view(-1, input_size))
        return self.reparam(mu, logvar)

PROJ_DIR = "/Volumes/projects_herting/LABDOCS/Personnel/Katie/kangaroo_aim1"
DATA_DIR = "data/"
FIGS_DIR = "figures/"
OUTP_DIR = "output/"

# don't hardcode any characteristics of ABCD Study data
# calculate everything on the fly

df = pd.read_pickle(
    join(PROJ_DIR, DATA_DIR, 'dset_clean.pkl')
)
smri_cols = list(df.filter(like='smri', axis=1).columns)
hormones_m = [
    'filtered_dhea',
    'filtered_ert'
]
hormones_f = [
    'filtered_dhea',
    'filtered_ert',
    'filtered_hse'
]
#batch_size = 128
seed = 4

f_df = df[df['demo_sex_v2_bl'] == 'Female'][smri_cols + hormones_f].dropna()
m_df = df[df['demo_sex_v2_bl'] == 'Male'][smri_cols + hormones_m].dropna()

## add in train/test split (jfc why am I making this so hard)
# ladies first

input_size = len(f_df.columns)
n_samples = len(f_df.index)
batch_size = 512
representation_size = 50

n_samples_per_batch = n_samples//input_size

#y = np.array([i for i in range(input_size)  for _ in range(n_samples_per_batch)])

d = np.identity(input_size)
x = StandardScaler().fit_transform(f_df.values)

model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train(epoch, batches_per_epoch = 501, log_interval=500):
    model.train()
    
    ind = np.arange(x.shape[0])
    for i in range(batches_per_epoch):
        data = torch.from_numpy(x[np.random.choice(ind, size=batch_size)]).float()
        data = Variable(data, requires_grad=False)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = model.loss(recon_batch, data, mu, logvar)
        loss.backward()
        #train_loss += loss.data[0]
        optimizer.step()
        if (i % log_interval == 0) and (epoch % 5 ==0):
            #Print progress
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, i * batch_size, batch_size*batches_per_epoch,
                loss.data / len(data)))

    print('====> Epoch: {} done!'.format(
          epoch))


def test(epoch, batches_per_epoch=1):
    model.eval()

    ind = np.arange(x.shape[0])
    for i in range(batches_per_epoch):
        
        data = torch.from_numpy(x[np.random.choice(ind, size=batch_size)]).float()
        data = Variable(data, requires_grad=False)
        recon_batch, mu, logvar = model(data)
        if i == 0:
            n = min(data.size(0), 2)
            print(data.view(batch_size, 2,2)[:n])
            print(recon_batch.view(batch_size, 2,2)[:n])

data = Variable(torch.from_numpy(x).float(), requires_grad=False)

model.train()
zs = model.get_z(data).data.numpy()

print_examples = False #Change to true if you want to see some examples at each step!
for epoch in range(1, 50):
    train(epoch)
    if print_examples:
        test(epoch)


recon, _, _ = model(data)

fake = pd.DataFrame(
    dtype=float,
    data=recon.detach().numpy(),
    index=range(len(f_df.index)),
    columns=f_df.columns
)

##### now for the boys


input_size = len(m_df.columns)
n_samples = len(m_df.index)
batch_size = 512
representation_size = 50

n_samples_per_batch = n_samples//input_size

#y = np.array([i for i in range(input_size)  for _ in range(n_samples_per_batch)])

d = np.identity(input_size)
x = StandardScaler().fit_transform(m_df.values)

model = VAE()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train(epoch, batches_per_epoch = 501, log_interval=500):
    model.train()
    
    ind = np.arange(x.shape[0])
    for i in range(batches_per_epoch):
        data = torch.from_numpy(x[np.random.choice(ind, size=batch_size)]).float()
        data = Variable(data, requires_grad=False)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = model.loss(recon_batch, data, mu, logvar)
        loss.backward()
        #train_loss += loss.data[0]
        optimizer.step()
        if (i % log_interval == 0) and (epoch % 5 ==0):
            #Print progress
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, i * batch_size, batch_size*batches_per_epoch,
                loss.data / len(data)))

    print('====> Epoch: {} done!'.format(
          epoch))


def test(epoch, batches_per_epoch=1):
    model.eval()

    ind = np.arange(x.shape[0])
    for i in range(batches_per_epoch):
        
        data = torch.from_numpy(x[np.random.choice(ind, size=batch_size)]).float()
        data = Variable(data, requires_grad=False)
        recon_batch, mu, logvar = model(data)
        if i == 0:
            n = min(data.size(0), 2)
            print(data.view(batch_size, 2,2)[:n])
            print(recon_batch.view(batch_size, 2,2)[:n])

data = Variable(torch.from_numpy(x).float(), requires_grad=False)

model.train()
zs = model.get_z(data).data.numpy()

print_examples = False #Change to true if you want to see some examples at each step!
for epoch in range(1, 50):
    train(epoch)
    if print_examples:
        test(epoch)


recon, _, _ = model(data)

fake2 = pd.DataFrame(
    dtype=float,
    data=recon.detach().numpy(),
    index=range(len(m_df.index)),
    columns=m_df.columns
)

fake = pd.concat(
    [
        fake,
        fake2
    ],
    axis=0
)
fake['site_id_l'] = np.random.choice(
    df['site_id_l'].unique(), 
    len(fake.index), 
    replace=True
)

fake.to_csv(
    join(PROJ_DIR, DATA_DIR, 'synthetic_dset.csv')
)
fake.to_pickle(
    join(PROJ_DIR, DATA_DIR, 'synthetic_dset.pkl')
)
