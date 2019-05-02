# coding=utf-8

import torch
from torch import nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
import datetime
import pickle
import random
import json
import numpy as np
from utils import split_data, split_tensor, save_checkpoint

data_directory = '/home/b.mcr/M2M/data/'
directory = '/home/b.mcr/M2M/'

# Embeddings -  test different values
embedding_dim_imsi = 200
embedding_dim_day = 5
embedding_dim_hour = 10
embedding_dim_msgid = 10
embedding_dim_op = 10
# total embeddings = 200 + 10 + 10 + 10 + 10 = 240

# Define parameters
num_epochs = 300
batch_size = 128
learning_rate = 1e-6
weight_decay = 1e-5
z_dims = 100
kld_coef = 1
h1_dims = embedding_dim_imsi + embedding_dim_day + embedding_dim_hour + embedding_dim_msgid + embedding_dim_op

day_size = 2
hour_size = 24
msgid_size = 2
op_size = 3

SEED = 1

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
    # All data: 14.111.573

    # dataset = dataset[:100000]
    print('Dataset size: ', len(dataset))
    # pickle.dump(dataset, open(data_directory + 'encoded_data_small.pickle', "wb"))
    # print('First data row: ', dataset[0])

    # Load mapping IMSIS to ordinals
    mapping_IMSI = json.load(open(data_directory + "mapping_imsi_to_ordinal_id.json", "rb"))

    # Number of different imsis
    imsis_vocab_size = len(mapping_IMSI)

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
    data_val = torch.LongTensor(dev_set)

    # Make batches
    num_batches = data_train.size(0) // batch_size  # Get number of batches
    data_train = data_train[:num_batches * batch_size]  # Trim last elements
    data_train = data_train.view(-1, batch_size, D)  # Reshape

    num_batches_val = data_val.size(0) // batch_size
    data_val = data_val[:num_batches_val * batch_size]
    data_val = data_val.view(-1, batch_size, D)

    # Define data loader: iterator over the dataset
    # dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=False)
    # iter_per_epoch = len(dataloader)
    # data_iter = iter(dataloader)

    print('')
    print(" ------------------ DATA -----------------")
    print('Observations: ', N)
    print('Dimensions/columns: ', D)
    print('Number of batches: ', num_batches)
    print('imsis_vocab_size: ', imsis_vocab_size)

    # --------------------------------------------- INIT MODEL -------------------------------------------------------

    # print(" -------------- AUTOENCODER --------------")

    checkpoint = 'BEST_checkpoint_04-26-22:50z100b128lr1e-06w1e-05.pth.tar'
    start_epoch = 0

    # Initialize / load checkpoint
    if checkpoint is None:

        # Instantiate and init the model, and move it to the GPU
        encoder = Encoder(imsis_size=imsis_vocab_size).cuda()
        decoder = Decoder(imsis_size=imsis_vocab_size).cuda()

        # Define optimizers
        encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=weight_decay)

    else:
        print('Loading checkpoint...')

        # Load the model, and move it to the GPU
        checkpoint = torch.load(checkpoint, map_location={'cuda:0': 'cpu'})
        start_epoch = checkpoint['epoch'] + 1
        decoder = checkpoint['decoder'].cuda()
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder'].cuda()
        encoder_optimizer = checkpoint['encoder_optimizer']
        print('Start epoch: ', start_epoch)

    print('')
    print(" --------------- PARAMETERS --------------")
    print('num epochs: ', num_epochs)
    print('batch size: ', batch_size)
    print('learning_rate: ', learning_rate)
    print('weight_decay: ', weight_decay)
    print('h1_dims: ', h1_dims)
    print('z_dims: ', z_dims)
    print('kld_coef: ', kld_coef)
    print('')
    print(" ------------- TRAINING LOOP -------------")

    epochs_since_improvement = 0
    best_loss = np.inf

    ############################################### TRAINING LOOP ##################################################

    for epoch in tqdm(range(start_epoch, num_epochs)):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        # if epochs_since_improvement == 40:
        #     break

        # One epoch's training
        avg_loss, loss, CE, KLD, losses, z = train(data_train, encoder, decoder, encoder_optimizer, decoder_optimizer, epoch)

        # One epoch's validation
        loss_val = validate(data_val, encoder, decoder)

        # Check if there was an improvement
        is_best = loss_val < best_loss
        best_loss = min(loss_val, best_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save model
        save_checkpoint(name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, is_best)

        # Tensorboard Logging - loss per batch
        writer.add_scalar('Training Loss', loss.item(), epoch)
        writer.add_scalar('Avg Training Loss', avg_loss.item(), epoch)
        writer.add_scalar('Cross Entropy Loss', CE.item(), epoch)
        writer.add_scalar('KLD', KLD.item(), epoch)
        writer.add_scalar('loss_val', loss_val, epoch)

        # writer.add_histogram('hist', z, epoch)

        # Plot imsi, day, hour, msgid, opcode losses separately
        writer.add_scalar('IMSI loss', losses[0].item(), epoch)
        writer.add_scalar('Day loss', losses[1].item(), epoch)
        writer.add_scalar('Hour loss', losses[2].item(), epoch)
        writer.add_scalar('Msg Id loss', losses[3].item(), epoch)
        writer.add_scalar('Op code loss', losses[4].item(), epoch)

    writer.close()


# ----------------------------------------------- MODEL -------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, imsis_size=None):
        super(Encoder, self).__init__()

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

    def embedding(self, x):

        if len(x.size()) > 1:
            imsi_embed = self.embed_imsi(x[:, 0])
            day_embed = self.embed_day(x[:, 1])
            hour_embed = self.embed_hour(x[:, 2])
            msgid_embed = self.embed_msgid(x[:, 3])
            op_embed = self.embed_op(x[:, 4])

            embed_tensor = torch.cat([imsi_embed, day_embed, hour_embed, msgid_embed, op_embed], dim=1)

        # print('IMSI SIZE ', imsi_embed.size())
        # # print(' day ', x[:, 1])  # batch
        # print('SIZE day embed', day_embed.size())  # size [128, 2]
        # print('hour SIZE ', hour_embed.size())
        # print('msgid SIZE ', msgid_embed.size())
        # print('OP SIZE ', op_embed.size())

        # embed_tensor = embed_tensor.view(128, -1)
        # print('SIZE embed_tensor ', embed_tensor.size())

        return embed_tensor

    def forward(self, x):
        # h1 is [batch_size, 10000]
        # h1 = F.relu(self.fc1(x))
        embed = self.embedding(x)
        mean, logvar = self.fc21(embed), self.fc22(embed)
        z = self.reparametrize(mean, logvar)
        # print('z size: ', z.size())

        return z, mean, logvar

    def reparametrize(self, mu, logvar):
        """
        Given mean and logvar returns z
        reparameterization trick: instead of sampling from Q(z|X), sample epsilon = N(0,I)

        nu, logvar: mean and log of variance of Q(z|X)
        """
        std = (logvar * 0.5).exp()
        eps = torch.randn_like(std)
        return mu + std * eps


class Decoder(nn.Module):
    def __init__(self, imsis_size=None):
        super(Decoder, self).__init__()

        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(z_dims, h1_dims)
        self.fc4 = nn.Linear(h1_dims, imsis_size + day_size + hour_size + op_size + msgid_size)

    def forward(self, z):
        h3 = self.relu(self.fc3(z))
        y = self.fc4(h3)
        # print('y size: ', y.size())
        return y


# ----------------------------------------------- LOSS -------------------------------------------------------


def cross_entropy_loss_per_class(recon_x, x, mu, logvar, kld_coef=1):
    global imsis_vocab_size
    # print(" ------------------ LOSS -----------------")
    # recon_x = recon_x.view(5, -1)
    # imsi, d, h, msg_id, op = torch.chunk(recon_x, 5, dim=0)

    #x = torch.transpose(x, 0, 1)

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
    imsi_loss = loss(imsi_in, x[:, 0])
    day_loss = loss(day_in, x[:, 1])
    hour_loss = loss(hour_in, x[:, 2])
    msg_loss = loss(msg_id_in, x[:, 3])
    op_loss = loss(op_in, x[:, 4])

    CE = imsi_loss + day_loss + hour_loss + msg_loss + op_loss
    losses = [imsi_loss, day_loss, hour_loss, msg_loss, op_loss]

    kld = KLD(mu, logvar, D, batch_size)
    return CE, kld, losses


def KLD(mu, logvar, dim, batch_size):

    """
    KLD is Kullback–Leibler divergence -- how much does one learned distribution deviate from another, in this specific
    case the learned distribution from the unit Gaussian

    Parameters
    ----------
    mu: latent mean
    logvar: latent log variance
    """

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Normalise by same number of elements as in reconstruction
    kld /= batch_size * dim
    return kld




# --------------------------------------------------- TRAIN -----------------------------------------------------------

def train(data, encoder, decoder, encoder_optimizer, decoder_optimizer, epoch):

    # toggle model to train mode  (dropout and batchnorm is used)
    encoder.train()
    decoder.train()
    avg_loss = 0

    for i in tqdm(range(0, data.size(0) - 1, 1)):
        # print('DATA SIZE: ', data.size())  # DATA SIZE:  torch.Size([88197, 128, 5])
        # input of shape (seq_len, batch, dimensions)

        x = data[i, :, :].cuda()
        # print('X SIZE: ', x.size())  # X SIZE:  torch.Size([128, 5])

        # Forward pass
        z, mean, logvar = encoder.forward(x)
        recon_batch = decoder.forward(z)

        # CE + KLD Loss
        CE, KLD, losses = cross_entropy_loss_per_class(recon_batch, x, mean, logvar, kld_coef)
        loss = CE + kld_coef * KLD

        # Back prop.
        decoder_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        loss.backward()

        # Update weights
        decoder_optimizer.step()
        encoder_optimizer.step()

        # Get latent space z
        z = encoder.reparametrize(mean, logvar)
        avg_loss += loss

    avg_loss = avg_loss / data.size(0)

    return avg_loss, loss, CE, KLD, losses, z

# --------------------------------------------------- VAL -----------------------------------------------------------


def validate(data_val, encoder, decoder):

    decoder.eval()  # eval mode (no dropout or batchnorm)
    # if encoder is not None:
    encoder.eval()
    avg_loss = 0

    # Loop over batches
    with torch.no_grad():
        for i in range(0, data_val.size(0) - 1, 1):

            x = data_val[i, :, :].cuda()

            # Forward pass
            z, mean, logvar = encoder.forward(x)
            recon_batch = decoder.forward(z)

            # Calculate loss
            CE, KLD, losses = cross_entropy_loss_per_class(recon_batch, x, mean, logvar, kld_coef)
            loss_val = CE + kld_coef * KLD
            avg_loss += loss_val

    avg_loss = avg_loss / data_val.size(0)

    return avg_loss


if __name__ == "__main__":
    main()
