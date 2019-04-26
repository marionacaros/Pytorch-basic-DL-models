# coding=utf-8

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F  # layers, activations and more
from tqdm import tqdm
from tensorboardX import SummaryWriter
import datetime
import pickle
import random
import json

data_directory = 'data/'
directory = ''

# Define parameters
num_epochs = 100
batch_size = 128
learning_rate = 1e-6
weight_decay = 1e-5
h1_dims = 240
z_dims = 100
kld_coef = 1

# Embeddings -  test different values
embedding_dim_imsi = 200
embedding_dim_day = 10
embedding_dim_hour = 10
embedding_dim_msgid = 10
embedding_dim_op = 10
# total embeddings = 200 + 10 + 10 + 10 + 10 = 240

day_size = 2
hour_size = 24
msgid_size = 2
op_size = 3

SEED = 1

# CUDA
CUDA = torch.cuda.is_available()
print("Cuda available", torch.cuda.is_available())

torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

# Define Tensorboard location and plot names
now = datetime.datetime.now()
name = now.strftime("%m-%d-%H:%M") + 'z' + str(num_epochs) + 'b' + str(batch_size) + 'lr' + \
           str(learning_rate) + 'w' + str(weight_decay)
location = directory + 'runs/' + name
writer = SummaryWriter(location)


def main():

    global imsis_vocab_size, N, D

    # --------------------------------------------- LOAD DATA ----------------------------------------------------------

    print(' Reading data...')
    dataset = pickle.load(open(data_directory + "encoded_data_list.pickle", "rb"))
 
    mapping_IMIS = json.load(open(data_directory + "mapping_imsi_to_ordinal_id.json", "rb"))

    # Number of different imsis
    imsis_vocab_size = len(mapping_IMIS)

    # Shuffle data
    random.seed(1)
    random.shuffle(dataset)
    # print('Shuffled data: ', dataset[0:5])

    # Split data
    training_set, dev_set, test_set = split_data(dataset)

    N = len(dataset)  # num of logs
    D = len(dataset[0])  # sequence length

    # Converting the data into Torch tensors
    data_train = torch.LongTensor(training_set)

    # Make batches
    num_batches = data_train.size(0) // batch_size  # Get number of batches
    data_train = data_train[:num_batches * batch_size]  # Trim last elements
    data_train = data_train.view(-1, batch_size, D)  # Reshape


    print(" ------------------ DATA -----------------")
    print('Observations: ', N)
    print('Dimensions/columns: ', D)
    print('Number of batches: ', num_batches)
    print('imsis_vocab_size: ', imsis_vocab_size)

    # print("iterations per epoch: ", iter_per_epoch)

    # ------------------------------------------------ INIT ------------------------------------------------------------

    print(" -------------- AUTOENCODER --------------")

    # Instantiate and init the model, and move it to the GPU
    ae = Autoencoder(input_size=D, imsis_size=imsis_vocab_size).cuda()
    print(ae)

    print(" --------------- PARAMETERS --------------")
    print('num epochs: ', num_epochs)
    print('batch size: ', batch_size)
    print('learning_rate: ', learning_rate)
    print('weight_decay: ', weight_decay)
    print('h1_dims: ', h1_dims)
    print('z_dims: ', z_dims)
    print('kld_coef: ', kld_coef)

    print(" ------------- TRAINING LOOP -------------")

    # Define optimizer
    optimizer = torch.optim.Adam(ae.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop epochs
    for epoch in tqdm(range(num_epochs)):
        loss, CE, KLD, losses = train(data_train, ae, optimizer, epoch)

        # Check if there was an improvement
        # is_best = loss > best_loss
        # best_loss = max(loss, best_loss)

        save_checkpoint(name, epoch, ae, optimizer, is_best=True)

        # Tensorboard Logging - loss per batch
        writer.add_scalar('Loss', loss.item(), epoch)
        writer.add_scalar('Cross Entropy Loss', CE.item(), epoch)
        writer.add_scalar('KLD', KLD.item(), epoch)
        # writer.add_histogram('hist', z, epoch)

        # Plot imsi, day, hour, msgid, opcode losses separately
        writer.add_scalar('IMSI loss', losses[0].item(), epoch)
        writer.add_scalar('Day loss', losses[1].item(), epoch)
        writer.add_scalar('Hour loss', losses[2].item(), epoch)
        writer.add_scalar('Msg Id loss', losses[3].item(), epoch)
        writer.add_scalar('Pp code loss', losses[4].item(), epoch)

    writer.close()


# --------------------------------------------- FUNCTIONS -------------------------------------------------------------


def split_data(data_list):
    split_1 = int(0.8 * len(data_list))
    split_2 = int(0.9 * len(data_list))
    train_d = data_list[0:split_1]
    dev = data_list[split_1:split_2]
    test = data_list[split_2:]

    return train_d, dev, test


class Autoencoder(nn.Module):
    def __init__(self, input_size=None, imsis_size=None):
        super(Autoencoder, self).__init__()

        # EMBEDDINGS this layer will map each imsi,day,etc to a feature space of size hidden_size
        self.embed_imsi = nn.Embedding(imsis_size, embedding_dim_imsi)  # vocabulary size, dimensionality
        self.embed_day = nn.Embedding(day_size, embedding_dim_day)
        self.embed_hour = nn.Embedding(hour_size, embedding_dim_hour)
        self.embed_msgid = nn.Embedding(msgid_size, embedding_dim_msgid)
        self.embed_op = nn.Embedding(op_size, embedding_dim_op)

        # ENCODER
        # self.fc1 = nn.Linear(input_size, h1_dims)
        self.fc21 = nn.Linear(h1_dims, z_dims)
        self.fc22 = nn.Linear(h1_dims, z_dims)
        self.relu = nn.ReLU()

        # DECODER
        self.fc3 = nn.Linear(z_dims, h1_dims)
        self.fc4 = nn.Linear(h1_dims, imsis_size + day_size + hour_size + op_size + msgid_size)
        self.sigmoid = nn.Sigmoid()

    def embedding(self, x):
        imsi_embed = self.embed_imsi(x[:, 0])
        day_embed = self.embed_day(x[:, 1])
        hour_embed = self.embed_hour(x[:, 2])
        msgid_embed = self.embed_msgid(x[:, 3])
        op_embed = self.embed_op(x[:, 4])

        # print('IMSI SIZE ', imsi_embed.size())
        # # print(' day ', x[:, 1])  # batch
        # print('SIZE day embed', day_embed.size())  # size [128, 2]
        # print('hour SIZE ', hour_embed.size())
        # print('msgid SIZE ', msgid_embed.size())
        # print('OP SIZE ', op_embed.size())

        embed_tensor = torch.cat([imsi_embed, day_embed, hour_embed, msgid_embed, op_embed], dim=1)
        # embed_tensor = embed_tensor.view(128, -1)
        # print('SIZE embed_tensor ', embed_tensor.size())

        return embed_tensor

    def forward(self, x):
        # ----------------------------- ENCODER --------------------------------------
        # h1 is [batch_size, 10000]
        # h1 = F.relu(self.fc1(x))
        embed = self.embedding(x)
        mean, logvar = self.fc21(embed), self.fc22(embed)

        # print('mean size: ', mean.size())
        # print('logvar size: ', logvar.size())
        z = self.reparametrize(mean, logvar)
        # print('z size: ', z.size())

        # ----------------------------- DECODER --------------------------------------
        h3 = self.relu(self.fc3(z))
        y = self.sigmoid(self.fc4(h3))
        # print('y size: ', y.size())

        return y, mean, logvar

    def reparametrize(self, mu, logvar):
        """
        mu : [batch, z_dims] mean matrix
        logvar : [batch, z_dims] variance matrix

        """
        std = logvar.mul(0.5).exp_()
        eps = std.new(std.size()).normal_()  # torch.FloatTensor()
        return eps.mul(std).add_(mu)


def loss_function_chunks(recon_x, x, mu, logvar, kld_coef=1):
    global imsis_vocab_size
    # print(" ------------------ LOSS -----------------")
    # recon_x = recon_x.view(5, -1)
    # imsi, d, h, msg_id, op = torch.chunk(recon_x, 5, dim=0)

    x = torch.transpose(x, 0, 1)

    # print('recon_x SIZE', recon_x.size())
    # # print('IMSI CHUNK SIZE: ', imsi[0].size())
    # print('X SIZE', x.size())
    # print('X[0] SIZE: ', x[0].size())

    imsi_in = recon_x[:, :imsis_vocab_size]
    day_in = recon_x[:, imsis_vocab_size : imsis_vocab_size + day_size]
    hour_in = recon_x[:, imsis_vocab_size + day_size: imsis_vocab_size + day_size + hour_size]
    msg_id_in = recon_x[:, imsis_vocab_size + day_size + hour_size: imsis_vocab_size + day_size + hour_size + msgid_size]
    op_in = recon_x[:, imsis_vocab_size + day_size + hour_size + msgid_size : ]

    # print('imsi_in SIZE', imsi_in.size())
    # print('day_in CHUNK SIZE: ', day_in.size())
    # print('hour_in SIZE', hour_in.size())
    # print('msg_id_in SIZE', msg_id_in.size())
    # print('op_in SIZE', op_in.size())

    # CrossEntropy Loss needs need input size (batch, C)
    loss = nn.CrossEntropyLoss()
    imsi_loss = loss(imsi_in, x[0])
    day_loss = loss(day_in, x[1])
    hour_loss = loss(hour_in, x[2])
    msg_loss = loss(msg_id_in, x[3])
    op_loss = loss(op_in, x[4])

    BCE = imsi_loss + day_loss + hour_loss + msg_loss + op_loss
    losses = [imsi_loss, day_loss, hour_loss, msg_loss, op_loss]

    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    # Normalise by same number of elements as in reconstruction
    KLD /= batch_size * D

    return BCE, KLD, losses


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

    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    # Normalise by same number of elements as in reconstruction
    KLD /= batch_size * D

    return BCE, kld_coef * KLD


def split_tensor(tensor, split_sizes, dim=0):
    """Splits the tensor according to chunks of split_sizes.

    Arguments:
        tensor (Tensor): tensor to split.
        split_sizes (list(int)): sizes of chunks
        dim (int): dimension along which to split the tensor.
    """
    if dim < 0:
        dim += tensor.dim()

    dim_size = tensor.size(dim)
    if dim_size != torch.sum(torch.Tensor(split_sizes)):
        raise KeyError("Sum of split sizes exceeds tensor dim")

    splits = torch.cumsum(torch.Tensor([0] + split_sizes), dim=0)[:-1]

    return tuple(tensor.narrow(int(dim), int(start), int(length))
                 for start, length in zip(splits, split_sizes))


def save_checkpoint(checkpoint_name, epoch, autoencoder, optimizer, is_best):

    state = {'epoch': epoch,
             'autoencoder': autoencoder,
             'optimizer': optimizer}
    filename = 'checkpoint_' + checkpoint_name + '.pth.tar'
    torch.save(state, filename)

    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    # if is_best:
    #     torch.save(state, 'BEST_' + filename)


# --------------------------------------------------- TRAIN -----------------------------------------------------------

def train(data, ae, optimizer, epoch):
    # toggle model to train mode
    ae.train()

    for i in tqdm(range(0, data.size(0) - 1, 1)):
        # print('DATA SIZE: ', data.size())  # DATA SIZE:  torch.Size([88197, 128, 5])
        # input of shape (seq_len, batch, dimensions)

        x = data[i, :, :].cuda()
        # print('X SIZE: ', x.size())  # X SIZE:  torch.Size([128, 5])

        # Forward pass
        recon_batch, mu, logvar = ae.forward(x)
        CE, KLD, losses = loss_function_chunks(recon_batch, x, mu, logvar, kld_coef)
        loss = CE + kld_coef * KLD

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Get latent space z
        z = ae.reparametrize(mu, logvar)

        # Print loss
        # if (i + 1) % 100 == 0:
        #    print('Epoch [%d/%d], Loss: %.4f ' % ( epoch + 1, num_epochs, loss.item()))

    return loss, CE, KLD, losses

if __name__ == "__main__":
    main()
