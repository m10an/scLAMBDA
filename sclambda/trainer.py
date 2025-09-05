import copy
import os

from tqdm import tqdm
import numpy as np
import torch
import torch.utils.data
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from scipy.sparse._csr import csr_matrix

from sclambda.model import Model


class PertDataset(torch.utils.data.Dataset):
    def __init__(self, x, p, ctrl_mean=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.x = x
        self.p = p
        self.ctrl_mean = ctrl_mean

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx].to(self.device), self.p[idx].to(self.device)


class Trainer:
    def __init__(
        self,
        adata,
        gene_emb,  # dictionary for gene embeddings
        split_name='split',
        latent_dim=30,
        hidden_dim=512,
        training_epochs=200,
        batch_size=500,
        lambda_MI=200,
        eps=0.001,
        model_path="models",
        multi_gene=False,
    ):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.adata = adata
        self.gene_emb = gene_emb
        self.split_name = split_name
        self.x_dim = self.adata.n_vars
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

        # control cells
        ctrl_x = adata[adata.obs['condition'].values == 'ctrl'].X
        self.ctrl_mean = np.mean(ctrl_x, axis=0)
        self.ctrl_x = torch.from_numpy(ctrl_x - self.ctrl_mean.reshape(1, -1)).float().to(self.device) # TODO: keep in cpu
        X = self.adata.X - self.ctrl_mean.reshape(1, -1)
        X = X.toarray() if isinstance(X, csr_matrix) else X

        # split datasets
        print("Spliting data...")
        train_mask = self.adata.obs[self.split_name].values == 'train'
        val_mask = self.adata.obs[self.split_name].values == 'val'
        self.adata_train = self.adata[train_mask]
        self.adata_val = self.adata[val_mask]
        self.pert_val = np.unique(self.adata_val.obs['condition'].values)

        self.model = Model(
            x_dim=self.x_dim,
            p_dim=self.p_dim,
            ctrl_mean=torch.tensor(self.ctrl_mean),
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            batch_size=self.batch_size,
            multi_gene=self.multi_gene,
        )
        self.model.to(self.device)

        self.train_data = PertDataset(
            x=torch.from_numpy(X[train_mask]).float(),
            p=torch.from_numpy(self.pert_emb_cells[train_mask]).float(),
            ctrl_mean=self.model.ctrl_mean,
        )
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )

        self.pert_delta = {}
        for i in np.unique(self.adata.obs['condition'].values):
            delta_i = np.mean(X[self.adata.obs['condition'].values == i], axis=0)
            self.pert_delta[i] = delta_i

    def loss_function(self, x, x_hat, p, p_hat, mean_z, log_var_z, s, s_marginal, T):
        reconstruction_loss = 0.5 * torch.mean(torch.sum((x_hat - x)**2, dim=1)) + 0.5 * torch.mean(torch.sum((p_hat - p)**2, dim=1))
        KLD_z = - 0.5 * torch.mean(torch.sum(1 + log_var_z - mean_z**2 - log_var_z.exp(), dim=1))
        MI_latent = torch.mean(T(mean_z, s.detach())) - torch.log(torch.mean(torch.exp(T(mean_z, s_marginal.detach()))))
        return reconstruction_loss + KLD_z + self.lambda_MI * MI_latent

    def loss_recon(self, x, x_hat):
        reconstruction_loss = 0.5 * torch.mean(torch.sum((x_hat - x)**2, dim=1))
        return reconstruction_loss

    def loss_MINE(self, mean_z, s, s_marginal, T):
        MI_latent = torch.mean(T(mean_z, s)) - torch.log(torch.mean(torch.exp(T(mean_z, s_marginal))))
        return - MI_latent

    def train(
        self,
        *,
        retrain=False,
        seed=1234
    ):
        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

        params = (
            list(self.model.Net.Encoder_x.parameters()) +
            list(self.model.Net.Encoder_p.parameters()) +
            list(self.model.Net.Decoder_x.parameters()) +
            list(self.model.Net.Decoder_p.parameters())
        )
        optimizer = Adam(params, lr=0.0005)
        scheduler = StepLR(optimizer, step_size=30, gamma=0.2)
        optimizer_MINE = Adam(self.model.Net.MINE.parameters(), lr=0.0005, weight_decay=0.0001)
        scheduler_MINE = StepLR(optimizer_MINE, step_size=30, gamma=0.2)

        corr_val_best = 0
        if retrain:
            if len(self.pert_val) > 0: # If validating
                self.model.eval()
                corr_ls = []
                for i in self.pert_val:
                    if self.multi_gene:
                        genes = i.split('+')
                        pert_emb_p = self.gene_emb[genes[0]] + self.gene_emb[genes[1]]
                    else:
                        pert_emb_p = self.gene_emb[i]
                    val_p = torch.from_numpy(np.tile(pert_emb_p, 
                                                     (self.ctrl_x.shape[0], 1))).float().to(self.device)
                    x_hat, p_hat, mean_z, log_var_z, s = self.model(self.ctrl_x, val_p)
                    x_hat = np.mean(x_hat.detach().cpu().numpy(), axis=0)
                    corr = np.corrcoef(x_hat, self.pert_delta[i])[0, 1]
                    corr_ls.append(corr)

                corr_val_best = np.mean(corr_ls)
                print("Previous best validation correlation delta %.5f" % corr_val_best)
        self.model.train()
        for epoch in tqdm(range(self.training_epochs)):
            for x, p in self.train_dataloader:
                # adversarial training on p
                p.requires_grad = True
                self.model.eval()
                with torch.enable_grad():
                    x_hat, _, _, _, _ = self.model(x.to(self.device), p.to(self.device))
                    recon_loss = self.loss_recon(x, x_hat)
                    grads = torch.autograd.grad(recon_loss, p)[0]
                    p_ae = p + self.eps * torch.norm(p, dim=1).view(-1, 1) * torch.sign(grads.data) # generate adversarial examples

                self.model.train()
                x_hat, p_hat, mean_z, log_var_z, s = self.model(x, p_ae)

                # for MINE
                index_marginal = np.random.choice(np.arange(len(self.train_data)), size=x_hat.shape[0])
                p_marginal = self.train_data.p[index_marginal]
                s_marginal = self.model.Net.Encoder_p(p_marginal.to(self.device))
                for _ in range(1):
                    optimizer_MINE.zero_grad()
                    loss = self.loss_MINE(mean_z, s, s_marginal, T=self.model.Net.MINE)
                    loss.backward(retain_graph=True)
                    optimizer_MINE.step()

                optimizer.zero_grad()
                loss = self.loss_function(x, x_hat, p, p_hat, mean_z, log_var_z, s, s_marginal, T=self.model.Net.MINE)
                loss.backward()
                optimizer.step()
            scheduler.step()
            scheduler_MINE.step()
            if (epoch+1) % 10 == 0:
                print("\tEpoch", (epoch+1), "complete!", "\t Loss: ", loss.item())
                if len(self.pert_val) > 0: # If validating
                    self.model.eval()
                    corr_ls = []
                    for i in self.pert_val:
                        if self.multi_gene:
                            genes = i.split('+')
                            pert_emb_p = self.gene_emb[genes[0]] + self.gene_emb[genes[1]]
                        else:
                            pert_emb_p = self.gene_emb[i]
                        val_p = torch.from_numpy(np.tile(pert_emb_p, 
                                                         (self.ctrl_x.shape[0], 1))).float().to(self.device)
                        # TODO: Split on batches? self.ctrl_x, val_p
                        x_hat, p_hat, mean_z, log_var_z, s = self.model.Net(self.ctrl_x, val_p)
                        x_hat = np.mean(x_hat.detach().cpu().numpy(), axis=0)
                        corr = np.corrcoef(x_hat, self.pert_delta[i])[0, 1]
                        corr_ls.append(corr)

                    corr_val = np.mean(corr_ls)
                    print("Validation correlation delta %.5f" % corr_val)
                    if corr_val > corr_val_best:
                        corr_val_best = corr_val
                        self.model_best = copy.deepcopy(self.model)
                    self.model.Net.train()
                else:
                    if epoch == (self.training_epochs-1):
                        self.model_best = copy.deepcopy(self.model)
        print("Finish training.")
        self.model = self.model_best
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        self.model.save(self.model_path)
