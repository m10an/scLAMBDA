import os
import time
import numpy as np
import scanpy as sc
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import copy

from sclambda.networks import *
from sclambda.utils import *

class Model(object):
    def __init__(self, 
                 adata, # anndata object already splitted
                 gene_emb, # dictionary for gene embeddings
                 split_name = 'split',
                 latent_dim = 30, hidden_dim = 512,
                 training_epochs = 200,
                 batch_size = 500,
                 lambda_MI = 200,
                 eps = 0.001,
                 seed = 1234,
                 model_path = "models",
                 multi_gene = True
                 ):

        # add device
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        self.adata = adata.copy()
        self.gene_emb = gene_emb
        self.x_dim = adata.shape[1]
        self.p_dim = gene_emb[list(gene_emb.keys())[0]].shape[0]
        self.gene_emb.update({'ctrl': np.zeros(self.p_dim)})
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.lambda_MI = lambda_MI
        self.eps = eps
        self.model_path = model_path
        self.multi_gene = multi_gene

        # compute perturbation embeddings
        print("Computing %s-dimentisonal perturbation embeddings for %s cells..." % (self.p_dim, adata.shape[0]))
        self.pert_emb_cells = np.zeros((adata.shape[0], self.p_dim))
        self.pert_emb = {}
        for i in tqdm(np.unique(adata.obs['condition'].values)):
            genes = i.split('+')
            if len(genes) > 1:
                pert_emb_p = self.gene_emb[genes[0]] + self.gene_emb[genes[1]]
            else:
                pert_emb_p = self.gene_emb[genes[0]]
            self.pert_emb_cells[adata.obs['condition'].values == i] += pert_emb_p.reshape(1, -1)
            self.pert_emb[i] = pert_emb_p
        self.adata.obsm['pert_emb'] = self.pert_emb_cells

        # control cells
        ctrl_x = adata[adata.obs['condition'].values == 'ctrl'].X
        self.ctrl_mean = np.mean(ctrl_x, axis=0)
        self.ctrl_x = torch.from_numpy(ctrl_x - self.ctrl_mean.reshape(1, -1)).float().to(self.device)
        self.adata.X = self.adata.X - self.ctrl_mean.reshape(1, -1)

        # split datasets
        print("Spliting data...")
        self.adata_train = self.adata[self.adata.obs[split_name].values == 'train']
        self.adata_val = self.adata[self.adata.obs[split_name].values == 'val']
        self.pert_val = np.unique(self.adata_val.obs['condition'].values)

        self.train_data = PertDataset(torch.from_numpy(self.adata_train.X).float().to(self.device), 
                                      torch.from_numpy(self.adata_train.obsm['pert_emb']).float().to(self.device))
        self.train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

        self.pert_delta = {}
        for i in np.unique(self.adata.obs['condition'].values):
            adata_i = self.adata[self.adata.obs['condition'].values == i]
            delta_i = np.mean(adata_i.X, axis=0)
            self.pert_delta[i] = delta_i

    def loss_function(self, x, x_hat, p, p_hat, mean_z, log_var_z, s, s_marginal, T):
        reconstruction_loss = 0.5 * torch.mean(torch.sum((x_hat - x)**2, axis=1)) + 0.5 * torch.mean(torch.sum((p_hat - p)**2, axis=1))
        KLD_z = - 0.5 * torch.mean(torch.sum(1 + log_var_z - mean_z**2 - log_var_z.exp(), axis=1))
        MI_latent = torch.mean(T(mean_z, s.detach())) - torch.log(torch.mean(torch.exp(T(mean_z, s_marginal.detach()))))
        return reconstruction_loss + KLD_z + self.lambda_MI * MI_latent

    def loss_recon(self, x, x_hat):
        reconstruction_loss = 0.5 * torch.mean(torch.sum((x_hat - x)**2, axis=1))
        return reconstruction_loss

    def loss_MINE(self, mean_z, s, s_marginal, T):
        MI_latent = torch.mean(T(mean_z, s)) - torch.log(torch.mean(torch.exp(T(mean_z, s_marginal))))
        return - MI_latent

    def train(self):
        self.Net = Net(x_dim = self.x_dim, p_dim = self.p_dim, 
                       latent_dim = self.latent_dim, hidden_dim = self.hidden_dim)
        params = list(self.Net.Encoder_x.parameters())+list(self.Net.Encoder_p.parameters())+list(self.Net.Decoder_x.parameters())+list(self.Net.Decoder_p.parameters())
        optimizer = Adam(params, lr=0.0005)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.2)
        optimizer_MINE = Adam(self.Net.MINE.parameters(), lr=0.0005, weight_decay=0.0001)
        scheduler_MINE = StepLR(optimizer_MINE, step_size=30, gamma=0.2)

        corr_val_best = 0
        self.Net.train()
        for epoch in tqdm(range(self.training_epochs)):
            for x, p in self.train_dataloader:
                # adversarial training on p
                p.requires_grad = True 
                self.Net.eval()
                with torch.enable_grad():
                    x_hat, _, _, _, _ = self.Net(x, p)
                    recon_loss = self.loss_recon(x, x_hat)
                    grads = torch.autograd.grad(recon_loss, p)[0]
                    p_ae = p + self.eps * torch.norm(p, dim=1).view(-1, 1) * torch.sign(grads.data) # generate adversarial examples

                self.Net.train()
                x_hat, p_hat, mean_z, log_var_z, s = self.Net(x, p_ae)

                # for MINE
                index_marginal = np.random.choice(np.arange(len(self.train_data)), size=x_hat.shape[0])
                p_marginal = self.train_data.p[index_marginal]
                s_marginal = self.Net.Encoder_p(p_marginal)
                for _ in range(1):
                    optimizer_MINE.zero_grad()
                    loss = self.loss_MINE(mean_z, s, s_marginal, T=self.Net.MINE)
                    loss.backward(retain_graph=True)
                    optimizer_MINE.step()

                optimizer.zero_grad()
                loss = self.loss_function(x, x_hat, p, p_hat, mean_z, log_var_z, s, s_marginal, T=self.Net.MINE)
                loss.backward()
                optimizer.step()
            scheduler.step()
            scheduler_MINE.step()
            if (epoch+1) % 10 == 0:
                print("\tEpoch", (epoch+1), "complete!", "\t Loss: ", loss.item())
                if len(self.pert_val) > 0: # If validating
                    self.Net.eval()
                    corr_ls = []
                    for i in self.pert_val:
                        if self.multi_gene:
                            genes = i.split('+')
                            pert_emb_p = self.gene_emb[genes[0]] + self.gene_emb[genes[1]]
                        else:
                            pert_emb_p = self.gene_emb[i]
                        val_p = torch.from_numpy(np.tile(pert_emb_p, 
                                                         (self.ctrl_x.shape[0], 1))).float().to(self.device)
                        x_hat, p_hat, mean_z, log_var_z, s = self.Net(self.ctrl_x, val_p)
                        x_hat = np.mean(x_hat.detach().cpu().numpy(), axis=0)
                        corr = np.corrcoef(x_hat, self.pert_delta[i])[0, 1]
                        corr_ls.append(corr)

                    corr_val = np.mean(corr_ls)
                    print("Validation correlation delta %.5f" % corr_val)
                    if corr_val > corr_val_best:
                        corr_val_best = corr_val
                        self.model_best = copy.deepcopy(self.Net)
                    self.Net.train()
                else:
                    if epoch == (self.training_epochs-1):
                        self.model_best = copy.deepcopy(self.Net)
        print("Finish training.")
        self.Net = self.model_best
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        state = {'Net': self.Net.state_dict()}
        torch.save(state, os.path.join(self.model_path, "ckpt.pth"))

    def load_pretrain(self):
        self.Net = Net(x_dim = self.x_dim, p_dim = self.p_dim, 
                       latent_dim = self.latent_dim, hidden_dim = self.hidden_dim)
        self.Net.load_state_dict(torch.load(os.path.join(self.model_path, "ckpt.pth"))['Net'])

    def predict(self, 
                pert_test, # perturbation or a list of perturbations
                return_type = 'mean' # return mean or cells
                ):
        self.Net.eval()
        res = {} 
        if isinstance(pert_test, str):
            pert_test = [pert_test]
        for i in pert_test:
            if self.multi_gene:
                genes = i.split('+')
                pert_emb_p = self.gene_emb[genes[0]] + self.gene_emb[genes[1]]
            else:
                pert_emb_p = self.gene_emb[i]
            val_p = torch.from_numpy(np.tile(pert_emb_p, 
                                     (self.ctrl_x.shape[0], 1))).float().to(self.device)
            x_hat, p_hat, mean_z, log_var_z, s = self.Net(self.ctrl_x, val_p)
            if return_type == 'cells':
                adata_pred = ad.AnnData(X=(x_hat.detach().cpu().numpy() + self.ctrl_mean.reshape(1, -1)))
                adata_pred.obs['condition'] = i
                res[i] = adata_pred
            elif return_type == 'mean':
                x_hat = np.mean(x_hat.detach().cpu().numpy(), axis=0) + self.ctrl_mean
                res[i] = x_hat
            else:
                raise ValueError("return_type can only be 'mean' or 'cells'.")
        return res

    def generate(self, 
                 pert_test, # perturbation or a list of perturbations
                 return_type = 'mean', # return mean or cells
                 n_cells = 1000 # number of cells to generate
                 ):
        self.Net.eval()
        res = {} 
        if isinstance(pert_test, str):
            pert_test = [pert_test]
        for i in pert_test:
            if self.multi_gene:
                genes = i.split('+')
                pert_emb_p = self.gene_emb[genes[0]] + self.gene_emb[genes[1]]
            else:
                pert_emb_p = self.gene_emb[i]
            val_p = torch.from_numpy(np.tile(pert_emb_p, 
                                     (n_cells, 1))).float().to(self.device)
            s = self.Net.Encoder_p(val_p)
            z = torch.randn(n_cells, self.latent_dim).to(self.device)
            x_hat = self.Net.Decoder_x(z+s)
            if return_type == 'cells':
                adata_pred = ad.AnnData(X=x_hat.detach().cpu().numpy() + self.ctrl_mean.reshape(1, -1))
                adata_pred.obs['condition'] = i
                res[i] = adata_pred
            elif return_type == 'mean':
                x_hat = np.mean(x_hat.detach().cpu().numpy(), axis=0) + self.ctrl_mean
                res[i] = x_hat
            else:
                raise ValueError("return_type can only be 'mean' or 'cells'.")
        return res

    def get_embedding(self, adata=None):
        if adata == None:
            input_adata = None
            adata = self.adata
        x = torch.from_numpy(adata.X).float().to(self.device)
        p = torch.from_numpy(adata.obsm['pert_emb']).float().to(self.device)
        x_hat, p_hat, mean_z, log_var_z, s = self.Net(x, p)
        adata.obsm['mean_z'] = mean_z.detach().cpu().numpy()
        adata.obsm['z+s'] = adata.obsm['mean_z'] + s.detach().cpu().numpy()

        emb_s = pd.DataFrame(s.detach().cpu().numpy(), index=adata.obs['condition'].values)
        emb_s = emb_s.groupby(emb_s.index, axis=0).mean()
        adata.uns['emb_s'] = emb_s
        if input_adata is None:
            self.adata = adata
        return adata

class PertDataset(Dataset):
    def __init__(self, x, p):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.x = x
        self.p = p

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].to(self.device), self.p[idx].to(self.device)
