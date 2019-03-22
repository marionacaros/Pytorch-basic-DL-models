# coding=utf-8
import torch
from torch import nn
import torch.nn.functional as F  # layers, activations and more
import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
import datetime

data_directory = '/home/b.mcr/M2M/code/IPXanalyzer/data/'
directory = '/home/b.mcr/M2M/code/IPXanalyzer/'

# Define parameters
num_epochs = 100
batch_size = 128
learning_rate = 1e-5
weight_decay = 1e-5
num_layers = 1
dropout_prob = 0
BPTT = 50
hidden_size = 1000
z_dims = 100  # connections through the autoencoder bottleneck
kld_coef = 1

SEED = 1

CUDA = torch.cuda.is_available()
print("Cuda available", torch.cuda.is_available())

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

# Define Tensorboard location and plot names
now = datetime.datetime.now()
location = directory + 'runs/' + now.strftime("%m-%d-%H:%M") + 'z' + str(num_epochs) + 'b' + str(batch_size) + 'lr' + \
           str(learning_rate) + 'w' + str(weight_decay) + 'RNN'
writer = SummaryWriter(location)


# --------------------------------------------- FUNCTIONS -------------------------------------------------------------


def split_data(dataframe):
    split_1 = int(0.8 * len(dataframe))
    split_2 = int(0.9 * len(dataframe))
    train_d = dataframe[:split_1]
    dev = dataframe[split_1:split_2]
    test = dataframe[split_2:]

    return train_d, dev, test


class RNN_Autoencoder(nn.Module):

    def __init__(self, input_size=None, h_size=hidden_size, z_size=z_dims, n_layers=num_layers):
        super(RNN_Autoencoder, self).__init__()

        self.input_size = input_size
        self.h_size = h_size
        self.z_size = z_size
        self.n_layers = n_layers

        # Encoder
        self.lstm_enc = torch.nn.LSTM(self.input_size, self.h_size, num_layers, dropout=dropout_prob, batch_first=False)
        # self.drop = torch.nn.Dropout(dropout_prob)
        self.fc21 = torch.nn.Linear(self.h_size, z_dims)
        self.fc22 = torch.nn.Linear(self.h_size, z_dims)

        # Decoder
        self.fc3 = torch.nn.Linear(z_dims, self.h_size)
        self.lstm_dec = torch.nn.LSTM(self.h_size, self.h_size, num_layers, dropout=dropout_prob, batch_first=False)
        self.fc4 = torch.nn.Linear(self.h_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def reparametrize(self, mu, logvar):
        """
        Parameters
        ----------
        mu : [batch, z_dims] mean matrix
        logvar : [batch, z_dims] variance matrix
        """
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, h):

        # h = states

        # ----------------------------- ENCODER --------------------------------------

        # LSTM
        y, h = self.lstm_enc(x, h)

        # Reshape: LSTM -> BPTT x batch x D -> FC -> (BPTT x b) x D   (BPTT x b) x 1000
        y = y.view(-1, self.h_size)

        # Fully-connected to latent space z
        mean = self.fc21(y)
        logvar = self.fc22(y)
        z = self.reparametrize(mean, logvar)

        # ----------------------------- DECODER --------------------------------------
        # Fully-connected
        y = self.fc3(z)

        # Reshape
        y = y.view(-1, batch_size, self.h_size)

        # LSTM
        y, h = self.lstm_dec(y, h)
        # y = self.drop(y)

        # Reshape
        y = y.view(-1, self.h_size)

        # Fully-connected to output
        y = self.fc4(y)
        y = self.sigmoid(y)

        return y, mean, logvar, h

    def get_initial_states(self, batch_size):
        # Set initial hidden and memory states to 0
        return (torch.zeros(num_layers, batch_size, hidden_size)).cuda(), (torch.zeros(num_layers, batch_size, hidden_size)).cuda()

    def detach(self, h):
        # Detach returns a new variable, decoupled from the current computation graph
        return h[0].detach(), h[1].detach()


def loss_function(recon_x, x, mu, logvar, kld_coef):
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
    """

    BCE = F.binary_cross_entropy(recon_x, x)

    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    # Normalise by same number of elements as in reconstruction
    KLD /= batch_size * D

    return BCE, KLD


# --------------------------------------------------- TRAIN -----------------------------------------------------------

def train(data, model, epoch):
    # toggle model to train mode
    model.train()

    # Get initial hidden and memory states
    states = model.get_initial_states(data.size(1))

    # Loop sequence length (train)
    # range([start], stop, step)
    # start: Starting number of the sequence.
    # stop: Generate numbers up to, but not including this number.
    # step: Difference between each number in the sequence.

    for i in tqdm(range(0, data.size(0) - 1, BPTT), desc='> Train', ncols=100, ascii=True):

        # Get the chunk and wrap the variables into the gradient propagation chain + move them to the GPU
        seqlen = int(np.min([BPTT, data.size(0) - 1 - i]))
        # input of shape (seq_len, batch, input_size)

        # print('DATA SIZE: ', data.size())
        # DATA SIZE: torch.Size([1425, 64, 14278])

        x = data[i:i + seqlen, :, :].cuda()

        # Truncated backpropagation
        # Otherwise the model would try to backprop all the way to the start of the data set
        states = model.detach(states)

        # Forward pass
        recon_batch, mu, logvar, states = model.forward(x, states)
        BCE, KLD_loss = loss_function(recon_batch, x.view(-1, D), mu, logvar, kld_coef)
        loss = BCE + kld_coef * KLD_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Get latent space z
        z = model.reparametrize(mu, logvar)

        # Tensorboard logging - Loss per batch
        writer.add_scalar('loss_epoch', loss.item(), epoch)
        writer.add_scalar('BCE_epoch', BCE.item(), epoch)
        writer.add_scalar('KLD_epoch', KLD_loss.item(), epoch)
        writer.add_histogram('hist', z, epoch)


# --------------------------------------------- LOAD DATA -------------------------------------------------------------

print(' Reading data...')
# dataset = pd.read_csv(data_directory + "one_hot_df.csv", sep=',', index_col=0)
dataset = pd.read_hdf(data_directory + 'one_hot_df.h5')

# Split data
train_df, dev_df, test_df = split_data(dataset)
training_set = np.array(train_df, dtype='int')
dev_set = np.array(dev_df, dtype='int')
test_set = np.array(test_df, dtype='int')

print(dataset.head(5))

N = dataset.shape[0]  # num of logs
D = dataset.shape[1]  # sequence length (dims)

# Converting the data into Torch tensors
data_train = torch.tensor(training_set, dtype=torch.float)

# Make batches
num_batches = data_train.size(0) // batch_size  # Get number of batches
data_train = data_train[:num_batches * batch_size]  # Trim last elements
data_train = data_train.view(-1, batch_size, D)  # Reshape

print(" ----------------- DATA -----------------")
print('Observations: ', N)
print('Dimensions: ', D)
print('Number of batches: ', num_batches)

# --------------------------------------------------- INIT -----------------------------------------------------------

print(" -------------- PARAMETERS --------------")
print('num epochs: ', num_epochs)
print('batch size: ', batch_size)
print('learning_rate: ', learning_rate)
print('weight_decay: ', weight_decay)
print('h1_dims: ', hidden_size)
print('z_dims: ', z_dims)
print('kld_coef: ', kld_coef)
print(" -----------------------------------------")

# Instantiate and init the model, and move it to the GPU
model = RNN_Autoencoder(input_size=D).cuda()

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop epochs
for epoch in tqdm(range(num_epochs)):

    train(data_train, model, epoch)

# torch.save(ae.state_dict(), './sim_autoencoder.pth')
