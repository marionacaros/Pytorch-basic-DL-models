# coding=utf-8

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F  # layers, activations and more
from torch.autograd import Variable  # variable node in computation graph
import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import datetime

# Define parameters
num_epochs = 100
batch_size = 256
learning_rate = 1e-6
weight_decay = 1e-5
z_dims = 100
h1_dims = 1000
kld_coef = 1

SEED = 1

# CUDA
CUDA = torch.cuda.is_available()
print("Cuda available", torch.cuda.is_available())

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

# Define Tensorboard location and plot names
now = datetime.datetime.now()
location = 'runs/' + now.strftime("%m-%d-%H:%M") + 'z' + str(num_epochs) + 'b' + str(batch_size) + 'lr' + \
           str(learning_rate) + 'w' + str(weight_decay)
writer = SummaryWriter(location)


# Create dummy df for debugging

# dataset = pd.read_csv("one_hot_df.csv", sep=',', index_col=0)
# dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
# dummy_df = dataset[:batch_size]
# dummy_df.to_csv('dummy_df.csv', index=False)


# --------------------------------------------- FUNCTIONS -------------------------------------------------------------


def split_data(dataframe):
    split_1 = int(0.8 * len(dataframe))
    split_2 = int(0.9 * len(dataframe))
    train_d = dataframe[:split_1]
    dev = dataframe[split_1:split_2]
    test = dataframe[split_2:]

    return train_d, dev, test


class Autoencoder(nn.Module):
    def __init__(self, input_size=None):
        super(Autoencoder, self).__init__()

        # ENCODER
        self.fc1 = nn.Linear(input_size, h1_dims)
        self.fc21 = nn.Linear(h1_dims, z_dims)
        self.fc22 = nn.Linear(h1_dims, z_dims)
        self.relu = nn.ReLU()

        # DECODER
        self.fc3 = nn.Linear(z_dims, h1_dims)
        self.fc4 = nn.Linear(h1_dims, input_size)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        # h1 is [batch_size, 10000]
        h1 = F.relu(self.fc1(x))  # type: Variable
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

    def reparametrize(self, mu, logvar):
        """
        Parameters
        ----------
        mu : [batch, z_dims] mean matrix
        logvar : [batch, z_dims] variance matrix

        """
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def get_z(self, x):
        """Encode a batch of data points, x, into their z representations."""
        mu, logvar = self.encode(x)
        return self.reparametrize(mu, logvar)


def loss_function(recon_x, x, mu, logvar, kld_coef=1):
    """
    KLD is Kullback–Leibler divergence -- how much does one learned
    distribution deviate from another, in this specific case the
    learned distribution from the unit Gaussian

    Parameters
    ----------
    recon_x: reconstructed data
    x: original data
    mu: latent mean
    logvar: latent log variance
    kld_coef
    """
    BCE = F.binary_cross_entropy(recon_x, x)  # reduction='sum'

    #   KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    # Normalise by same number of elements as in reconstruction
    KLD /= batch_size * D

    return BCE, kld_coef * KLD


# --------------------------------------------------- TRAIN -----------------------------------------------------------

def train(dataloader, ae, epoch, i):

    # toggle model to train mode
    ae.train()

    for sample_batch in dataloader:

        # sample_batch size = torch.Size([512, 14278])
        input = Variable(sample_batch).cuda()

        # Forward pass
        recon_batch, mu, logvar = ae.forward(input)
        BCE, KLD_loss = loss_function(recon_batch, input, mu, logvar, kld_coef)
        loss = BCE + kld_coef * KLD_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Get latent space z
        z = ae.reparametrize(mu, logvar)

        # if (i + 1) % 100 == 0:
        #    print('Epoch [%d/%d], Iter [%d] Loss: %.4f ' % ( epoch + 1, num_epochs, i + 1, loss.item()))

        # Tensorboard Logging
        writer.add_scalar('loss_epoch', loss.item(), epoch)
        writer.add_scalar('BCE_i', BCE.item(), i)
        writer.add_scalar('KLD_i', KLD_loss.item(), i)
        writer.add_histogram('hist', z, epoch)

        i += 1
    return i


# --------------------------------------------- LOAD DATA -------------------------------------------------------------


dataset = pd.read_csv("one_hot_df.csv", sep=',', index_col=0)

# Split data
train_df, dev_df, test_df = split_data(dataset)
training_set = np.array(train_df, dtype='int')
dev_set = np.array(dev_df, dtype='int')
test_set = np.array(test_df, dtype='int')

print(dataset.head(5))

N = dataset.shape[0]  # num of logs
D = dataset.shape[1]  # sequence length

print('We have %s observations with %s dimensions' % (N, D))
# We have 40110 observations with 14804 dimensions

# Converting the data into Torch tensors
training_set = torch.tensor(training_set, dtype=torch.float)

# Make batches
num_batches=training_set.size(0)//batch_size         # Get number of batches
data_train=training_set[:num_batches*batch_size]     # Trim last elements

# Define data loader: iterator over the dataset
dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=False)

iter_per_epoch = len(dataloader)
data_iter = iter(dataloader)
print("iter_per_epoch: ", iter_per_epoch)

# --------------------------------------------------- INIT -----------------------------------------------------------

# Instantiate and init the model, and move it to the GPU
ae = Autoencoder(input_size=D).cuda()
print(ae)

# Define optimizer
optimizer = torch.optim.Adam(ae.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Iterations
i = 0

for epoch in tqdm(range(num_epochs)):

    i = train(dataloader, ae, epoch, i)

writer.close()

# torch.save(ae.state_dict(), './sim_autoencoder.pth')


