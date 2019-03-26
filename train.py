import os
import pdb
import time
import torch
import argparse
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import defaultdict

from models import VAE


def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    print(device)
    ts = time.time()

    datax = np.load('Data/room_feature.npy')
    datac = np.load('Data/furn_type.npy')
    datas = np.load('Data/buju_feature.npy')
    data_size = len(datax)
    in_dim = args.in_dim
    out_dim = args.out_dim
    MSE_fn = torch.nn.MSELoss(reduce=True, size_average=True)
    def pairwise_distances(x, y=None):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x**2).sum(1).view(-1, 1)
        if y is not None:
            y_norm = (y**2).sum(1).view(1, -1)
        else:
            y = x
            y_norm = x_norm.view(1, -1)

        dist = (x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))) / x.size(0)
        return dist

    def loss_fn(pred_s, s, mean, log_var, z):
        BCE = MSE_fn(pred_s.view(-1, out_dim), s.view(-1, out_dim))
        print(pairwise_distances(z)-pairwise_distances(s))
        pdist_loss = MSE_fn(pairwise_distances(z),pairwise_distances(s))
        # KLD = -0.2 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        print('BCE',BCE)
        # print(KLD)
        print('pdist',pdist_loss)
        # return (BCE + 1e-1 * KLD) / s.size(0)
        return (BCE + 1e0 * pdist_loss)/ s.size(0)

    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        num_labels=12 if args.conditional else 0).to(device)
    # print(vae.encoder)
    # print(vae.decoder)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    batch_size = args.batch_size
    indices = [i for i in range(data_size)]
    batch_num = int(data_size/batch_size)

    while batch_size * batch_num < data_size:
        batch_num += 1

    for epoch in range(args.epochs):
        random.shuffle (indices)
        for i in range(batch_num):
            x = torch.FloatTensor(datax[i*batch_size:min((i+1)*batch_size,data_size),:])
            c = torch.FloatTensor(datac[i*batch_size:min((i+1)*batch_size,data_size),:])
            s = torch.FloatTensor(datas[i*batch_size:min((i+1)*batch_size,data_size),:])
            x, c, s = x.to(device), c.to(device), s.to(device)

            if args.conditional:
                pred_s, mean, log_var, z = vae(x, c)
            else:
                pred_s, mean, log_var, z = vae(x)
            loss = loss_fn(pred_s, s, mean, log_var, z)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % args.print_every == 0 or i == batch_num-1:
                print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                    epoch, args.epochs, i, batch_num, loss.item()))

    indices = [i for i in range(data_size)]
    latent_var = np.zeros((data_size,args.latent_size))
    pred_pos = np.zeros((data_size,args.out_dim))
    for i in range(batch_num):
        x = torch.FloatTensor(datax[i*batch_size:min((i+1)*batch_size,data_size),:])
        c = torch.FloatTensor(datac[i*batch_size:min((i+1)*batch_size,data_size),:])
        s = torch.FloatTensor(datas[i*batch_size:min((i+1)*batch_size,data_size),:])
        pred_s, mean, log_var, z = vae(x, c)
        latent_var[i*batch_size:min((i+1)*batch_size,data_size),:] = mean.data.numpy()
        pred_pos[i*batch_size:min((i+1)*batch_size,data_size),:] = pred_s.data.numpy()
    
    np.save('latent_var.npy',latent_var)
    np.save('pred_pos.npy',pred_pos)

    plt.scatter(latent_var[:,0], latent_var[:,1], label='latent')
    #plt.plot(n_components, aic, label='AIC')
    plt.savefig('latent1.png')
    plt.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--in_dim", type=int, default=14)
    parser.add_argument("--out_dim", type=int, default=36)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[14,32,16])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[16,32,36])
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", default=True)

    args = parser.parse_args()

    main(args)
