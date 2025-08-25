import os
import time
import json

import numpy as np
import anndata as ad
import pandas as pd
import torch
import torch.nn as nn

from sclambda.networks import Net


class Model(nn.Module):
    def __init__(
        self,
        x_dim: int,
        p_dim: int,
        ctrl_mean: torch.Tensor | None = None,
        latent_dim=30,
        hidden_dim=512,
        batch_size=500,
        multi_gene=False,
        gene_emb=None
    ):
        super().__init__()

        # add device
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.x_dim = x_dim
        self.p_dim = p_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.multi_gene = multi_gene

        self.gene_emb = gene_emb or {}
        self.gene_emb.update({'ctrl': np.zeros(self.p_dim)})

        self.Net = Net(
            x_dim=self.x_dim,
            p_dim=self.p_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim
        )
        self.register_buffer(
            "ctrl_mean",
            torch.zeros([1, x_dim]) if ctrl_mean is None else ctrl_mean
        )

    def forward(self, x, p):
        return self.Net(x, p)

    def predict(
        self,
        pert_test,  # perturbation or a list of perturbations
        return_type='mean'  # return mean or cells
    ):
        self.eval()
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
                adata_pred = ad.AnnData(X=(x_hat.detach().cpu().numpy() + self.ctrl_mean.numpy().reshape(1, -1)))
                adata_pred.obs['condition'] = i
                res[i] = adata_pred
            elif return_type == 'mean':
                x_hat = np.mean(x_hat.detach().cpu().numpy(), axis=0) + self.ctrl_mean.numpy()
                res[i] = x_hat
            else:
                raise ValueError("return_type can only be 'mean' or 'cells'.")
        return res

    def generate(
        self,
        pert_test,  # perturbation or a list of perturbations
        return_type='mean',  # return mean or cells
        n_cells=10000  # number of cells to generate
    ):
        self.eval()
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
                res[i] = x_hat.detach().cpu().numpy() + self.ctrl_mean.numpy().reshape(1, -1)
            elif return_type == 'mean':
                x_hat = np.mean(x_hat.detach().cpu().numpy(), axis=0) + self.ctrl_mean.numpy()
                res[i] = x_hat
            else:
                raise ValueError("return_type can only be 'mean' or 'cells'.")
        return res

    def get_embedding(self, adata):
        x = torch.from_numpy(adata.X).float().to(self.device)
        p = torch.from_numpy(adata.obsm['pert_emb']).float().to(self.device)
        x_hat, p_hat, mean_z, log_var_z, s = self.Net(x, p)
        adata.obsm['mean_z'] = mean_z.detach().cpu().numpy()
        adata.obsm['z+s'] = adata.obsm['mean_z'] + s.detach().cpu().numpy()

        emb_s = pd.DataFrame(s.detach().cpu().numpy(), index=adata.obs['condition'].values)
        emb_s = emb_s.groupby(emb_s.index, axis=0).mean()
        adata.uns['emb_s'] = emb_s
        return adata

    def save(self, path):
        torch.save(self.state_dict(), os.path.join(path, "ckpt.pth"))
        args = dict(
            x_dim=self.x_dim,
            p_dim=self.p_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            multi_gene=self.multi_gene,
        )
        with open(os.path.join(path, 'args.json'), 'w') as f:
            json.dump(args, f)

    @classmethod
    def load(cls, path, **kwargs):
        with open(os.path.join(path, 'args.json')) as f:
            args = json.load(f)
        args.update(kwargs)
        model = cls(**args)
        model.load_state_dict(torch.load(os.path.join(path, "ckpt.pth")))
        return model

