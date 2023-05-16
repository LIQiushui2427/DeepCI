# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import numpy as np
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch import optim
from torch.utils.tensorboard.writer import SummaryWriter
from .oadam import OAdam
from torch.optim import Adam, SGD, AdamW, RMSprop
from .rbflayer import RBF
from torch.amp import autocast_mode
from torch.nn import functional as F
# TODO. This epsilon is used only because pytorch 1.5 has an instability in torch.cdist
# when the input distance is close to zero, due to instability of the square root in
# automatic differentiation. Should be removed once pytorch fixes the instability.
# It can be set to 0 if using pytorch 1.4.0
EPSILON = 1e-2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_img(x):
    out = 0.5 * (x + 1)  # 将x的范围由(-1,1)伸缩到(0,1)
    out = out.view(-1, 1, 28, 28)
    return out

def add_weight_decay(net, l2_value, skip_list=()):
    decay, no_decay = [], []
    for name, param in net.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]

def criterion(x, y, label):# x:(bs,11),y (bs), label:(bs,1)
    # x_pred = torch.argmax(x[:,:-1], dim=1)# (bs)
    loss_0 = F.cross_entropy(x[:,:-1], y, reduction='none')# (bs)
    loss_1 = torch.mean(x[:,-1]) * label# (bs)
    return torch.mean(loss_0 + loss_1)
def _kernel(x, y, basis_func, sigma):
    return basis_func(torch.cdist(x, y + EPSILON) * torch.abs(sigma))


class _BaseADeepCI:

    def _pretrain(self, X_1_a_j, X_2_a_j, X_1_a_k, X_2_a_k, 
                  X_1_b_j, X_2_b_j, X_1_b_k, X_2_b_k, Z, X_2_a_j_t,X_2_a_k_t,X_2_b_j_t,X_2_b_k_t,
                  learner_l2, adversary_l2, adversary_norm_reg,
                  learner_lr, adversary_lr, n_epochs, bs, train_learner_every, train_adversary_every,
                  warm_start, logger, model_dir, device = device, verbose = False, add_sample_inds=False):
        """ Prepares the variables required to begin training.
        """
        self.verbose = verbose

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.tempdir = tempfile.TemporaryDirectory(dir=model_dir)
        self.model_dir = self.tempdir.name

        self.n_epochs = n_epochs

        if add_sample_inds:
            sample_inds = torch.tensor(np.arange(X_1_a_j.shape[0]))
            self.train_ds = TensorDataset(X_1_a_j, X_2_a_j, X_1_a_k, X_2_a_k, 
                  X_1_b_j, X_2_b_j, X_1_b_k, X_2_b_k, Z, sample_inds)
        else:
            self.train_ds = TensorDataset(X_1_a_j, X_2_a_j, X_1_a_k, X_2_a_k, 
                  X_1_b_j, X_2_b_j, X_1_b_k, X_2_b_k, Z, X_2_a_j_t,X_2_a_k_t,X_2_b_j_t,X_2_b_k_t)
            
        self.train_dl = DataLoader(self.train_ds, batch_size=bs, shuffle=True)

        self.learner = self.learner.to(device)
        self.adversary = self.adversary.to(device)

        if not warm_start:
            self.learner.apply(lambda m: (
                m.reset_parameters() if hasattr(m, 'reset_parameters') else None))
            self.adversary.apply(lambda m: (
                m.reset_parameters() if hasattr(m, 'reset_parameters') else None))

        beta1 = 0.
        self.optimizerD = SGD(add_weight_decay(self.learner, learner_l2),
                                lr=learner_lr, momentum=0.9)
        self.optimizerG = Adam(add_weight_decay(
                                self.adversary, adversary_l2, skip_list=[]), lr=adversary_lr, betas=(beta1, .01))

        if logger is not None:
            self.writer = SummaryWriter()

        return X_1_a_j, X_2_a_j, X_1_a_k, X_2_a_k, X_1_b_j, X_2_b_j, X_1_b_k, X_2_b_k, Z

    def predict(self, T, model='avg', burn_in=0, alpha=None):
        """
        Parameters
        ----------
        T : treatments
        model : one of ('avg', 'final'), whether to use an average of models or the final
        burn_in : discard the first "burn_in" epochs when doing averaging
        alpha : if not None but a float, then it also returns the a/2 and 1-a/2, percentile of
            the predictions across different epochs (proxy for a confidence interval)
        """
        if model == 'avg':
            preds = np.array([torch.load(os.path.join(self.model_dir,
                                                      "epoch{}".format(i)))(T).cpu().data.numpy()
                              for i in np.arange(burn_in, self.n_epochs)])
            if alpha is None:
                return np.mean(preds, axis=0)
            else:
                return np.mean(preds, axis=0),\
                    np.percentile(
                        preds, 100 * alpha / 2, axis=0), np.percentile(preds, 100 * (1 - alpha / 2), axis=0)
        if model == 'final':
            return torch.load(os.path.join(self.model_dir,
                                           "epoch{}".format(self.n_epochs - 1)))(T).cpu().data.numpy()
        if isinstance(model, int):
            return torch.load(os.path.join(self.model_dir,
                                           "epoch{}".format(model)))(T).cpu().data.numpy()


class _BaseSupLossADeepCI(_BaseADeepCI):

    def fit(self, X_1_a_j, X_2_a_j, X_1_a_k, X_2_a_k,
                  X_1_b_j, X_2_b_j, X_1_b_k, X_2_b_k, Z, X_2_a_j_t,X_2_a_k_t,X_2_b_j_t,X_2_b_k_t,
                learner_l2=2e-5, adversary_l2=2e-5, adversary_norm_reg=1e-3,
                learner_lr=0.00002, adversary_lr=0.00002, n_epochs=100, bs=100, train_learner_every=1, train_adversary_every=8, ols_weight=0.1, warm_start=False, logger=None, model_dir='.', device = device, verbose=False):
        """
        Parameters
        ----------
        Z : instruments
        T : treatments
        Y : outcome
        learner_l2, adversary_l2 : l2_regularization of parameters of learner and adversary
        adversary_norm_reg : adveresary norm regularization weight
        learner_lr : learning rate of the Adam optimizer for learner
        adversary_lr : learning rate of the Adam optimizer for adversary
        n_epochs : how many passes over the data
        bs : batch size
        train_learner_every : after how many training iterations of the adversary should we train the learner
        ols_weight : weight on OLS (square loss) objective
        warm_start : if False then network parameters are initialized at the beginning, otherwise we start
            from their current weights
        logger : a function that takes as input (learner, adversary, epoch, writer) and is called after every epoch
            Supposed to be used to log the state of the learning.
        model_dir : folder where to store the learned models after every epoch
        device : which device to use for training, either 0 or 'cpu'
        """

        X_1_a_j, X_2_a_j, X_1_a_k, X_2_a_k, X_1_b_j, X_2_b_j, X_1_b_k, X_2_b_k, Z = self._pretrain(X_1_a_j, X_2_a_j, X_1_a_k, X_2_a_k, X_1_b_j, X_2_b_j, X_1_b_k, X_2_b_k, Z, X_2_a_j_t,X_2_a_k_t,X_2_b_j_t,X_2_b_k_t, learner_l2, adversary_l2, adversary_norm_reg,learner_lr, adversary_lr, n_epochs, bs, train_learner_every, train_adversary_every,warm_start, logger, model_dir, device, verbose)

        print("BaseSupLossADeepCI: Training learner and adversary")

        for epoch in range(n_epochs):
            self.learner.train()
            self.adversary.train()
            all_D_loss = 0
            all_G_loss = 0
            # if self.verbose > 0:
                # print("Epoch #", epoch, sep="")

            for it, (x_1_a_j_batch, x_2_a_j_batch, x_1_a_k_batch, x_2_a_k_batch, 
                        x_1_b_j_batch, x_2_b_j_batch, x_1_b_k_batch, x_2_b_k_batch, z_batch,X_2_a_j_t,X_2_a_k_t,X_2_b_j_t,X_2_b_k_t) in enumerate(self.train_dl):

                x_1_a_j_batch, x_2_a_j_batch, x_1_a_k_batch, x_2_a_k_batch, x_1_b_j_batch, x_2_b_j_batch, x_1_b_k_batch, x_2_b_k_batch, z_batch = \
                    map(lambda x: x.to(device), (x_1_a_j_batch, x_2_a_j_batch, x_1_a_k_batch, x_2_a_k_batch, x_1_b_j_batch, x_2_b_j_batch, x_1_b_k_batch, x_2_b_k_batch, z_batch))
                X_2_a_j_t,X_2_a_k_t,X_2_b_j_t,X_2_b_k_t = map(lambda x: x.to(device), (X_2_a_j_t,X_2_a_k_t,X_2_b_j_t,X_2_b_k_t))
                # shape of x_1_a_j_batch torch.Size([200]) # 0-1
                # shape of x_2_a_j_batch torch.Size([200, 28, 28])
                # shape of z_batch torch.Size([200]), range is [0,1]
                # print("shape of z_batch", z_batch.shape)
                # Shape of X_2_a_j_t : 200, 1
                
                num_imgs = x_1_a_j_batch.shape[0] # bs
                real_labels = torch.ones(num_imgs, 1).to(device)
                fake_labels = torch.zeros(num_imgs, 1).to(device)
                # print("shape of x_2_a_j_batch", x_2_a_j_batch.shape)# shape of x_2_a_j_batch torch.Size([bs, 784]
                # Train Discriminator
                
                # flatten the images  
                x_2_a_j_batch = torch.flatten(x_2_a_j_batch, start_dim=1)
                x_2_a_k_batch = torch.flatten(x_2_a_k_batch, start_dim=1)
                x_2_b_j_batch = torch.flatten(x_2_b_j_batch, start_dim=1)
                x_2_b_k_batch = torch.flatten(x_2_b_k_batch, start_dim=1)
                
                X_2_a_k_t = torch.flatten(X_2_a_k_t)
                X_2_a_k_t = torch.tensor(X_2_a_k_t, dtype=torch.long)
                X_2_a_j_t = torch.flatten(X_2_a_j_t)
                X_2_a_j_t = torch.tensor(X_2_a_j_t, dtype=torch.long)
                X_2_b_j_t = torch.flatten(X_2_b_j_t)
                X_2_b_j_t = torch.tensor(X_2_b_j_t, dtype=torch.long)
                X_2_b_k_t = torch.flatten(X_2_b_k_t)
                x_2_b_k_t = torch.tensor(X_2_b_k_t, dtype=torch.long)
                
                if (it % train_learner_every == 0):
                    
                    pred_a_k = self.learner(x_2_a_k_batch)# shape of pred_a_k torch.Size([200, 11])
                    pred_a_j = self.learner(x_2_a_j_batch)
                    pred_b_j = self.learner(x_2_b_j_batch)
                    pred_b_k = self.learner(x_2_b_k_batch)
                    
                    # print('shape(pred_a_k)', pred_a_k.shape)
                    # print('X_2_a_k_t shape', X_2_a_k_t.shape)
                    
                    # D_real_loss = criterion(pred_a_k, X_2_a_k_t, real_labels)
                    

                    noise_img = self.adversary(z_batch.to(dtype = torch.float32))
                    
                    # print('shape(noise_img)', noise_img.shape)
                    # print('Noise_img is on device:', noise_img.device)
                    # fake_output = self.learner(noise_img.to(dtype = torch.float32))

                    # D_fake_loss = criterion(fake_output, X_2_a_j_t, fake_labels)
                    
                    # print('shape(fake_img)', fake_img.shape)
                    m = (x_1_a_j_batch - x_1_a_k_batch + \
                            x_1_b_k_batch - x_1_b_j_batch + \
                                torch.argmax(pred_a_j) - torch.argmax(pred_a_k) + \
                                    torch.argmax(pred_b_k) - torch.argmax(pred_b_j)) * \
                                        torch.mean(noise_img**2)
                    
                    print('torch.mean(torch.maximum(torch.tensor(0),-1*(m)))', torch.mean(torch.maximum(torch.tensor(0),-1*(m))).item())
                    # D_loss = D_fake_loss + D_real_loss + 2 * torch.mean(torch.maximum(torch.tensor(0),-1*(m))) - ols_weight * torch.mean((x_1_a_j_batch - x_1_a_k_batch + pred_b_k - pred_b_j)**2) + torch.mean(noise_img**2)
                    # D_loss = (2 * torch.mean(torch.maximum(torch.tensor(0),-1*(m)))) ** 2 - ols_weight * torch.mean((x_1_a_j_batch - x_1_a_k_batch + pred_b_k - pred_b_j)**2) + torch.mean(noise_img**2)
                    D_loss = torch.maximum(torch.tensor(0), (10 * torch.mean(torch.maximum(torch.tensor(0),-1*(m))) ** 2 - ols_weight * torch.mean((x_1_a_j_batch - x_1_a_k_batch + pred_b_k - pred_b_j)**2))) + torch.mean(noise_img**2)
                    # abs
                    self.optimizerD.zero_grad()
                    D_loss.backward(retain_graph=True)
                    self.optimizerD.step()
                    all_D_loss += D_loss.item()

                if (it % train_adversary_every == 0):
                # Train Generator
                    self.learner.eval()
                    
                    pred_a_j = self.learner(x_2_a_j_batch)
                    pred_a_k = self.learner(x_2_a_k_batch)
                    pred_b_j = self.learner(x_2_b_j_batch)
                    pred_b_k = self.learner(x_2_b_k_batch)# shape of pred_a_j is (batch_size, 1)
                    noise_img = self.adversary(z_batch.to(dtype = torch.float32))
                    
                    m = (x_1_a_j_batch - x_1_a_k_batch + \
                            x_1_b_k_batch - x_1_b_j_batch + \
                                pred_a_j - pred_a_k + \
                                pred_b_k - pred_b_j).to(device)

                    # z = torch.rand((num_imgs, 784), device=device)
                    # fake_img = self.adversary(z.to(dtype = torch.float32))
                    fake_img = self.adversary(z_batch.to(dtype = torch.float32))
                    
                    # G_output = self.learner(fake_img)
                    # G_loss = criterion(G_output, X_2_a_j_t, real_labels)
                    # G_loss = criterion(G_output, X_2_a_j_t, fake_labels)
                    G_loss = 2 * torch.mean(torch.maximum(torch.tensor(0),-1*(m)))
                    
                    self.optimizerG.zero_grad()
                    G_loss.backward()
                    self.optimizerG.step()

                    self.adversary.eval()
                    all_G_loss += G_loss.item()
                
                
                print('Epoch {}, d_loss: {:.6f}, g_loss: {:.6f} '.format(epoch, all_D_loss/len(self.train_dl), all_G_loss/len(self.train_dl)))
            
            torch.save(self.learner, os.path.join(
                self.model_dir, "epoch{}".format(epoch)))

            if logger is not None:
                logger(self.learner, self.adversary, epoch, self.writer)
                
        if logger is not None:
            self.writer.flush()
            self.writer.close()
            

        return self


class ADeepCI(_BaseSupLossADeepCI):

    def __init__(self, learner, adversary):
        """
        Parameters
        ----------
        learner : a pytorch neural net module
        adversary : a pytorch neural net module
        """
        print("Initializing ADeepCI")
        self.learner = learner
        self.adversary = adversary
        # whether we have a norm penalty for the adversary
        self.adversary_reg = False
        # which adversary parameters to not ell2 penalize
        self.skip_list = []
