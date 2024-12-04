# scLAMBDA

*Modeling and predicting single-cell multi-gene perturbation responses with scLAMBDA*

![scLAMBDA_overview](https://github.com/gefeiwang/scLAMBDA/blob/main/demos/overview.png)

## Installation

## Quick Start

### Preprocessing
scLAMBDA accepts logarithmized data stored in an AnnData object as input. The target genes should be specified in `adata.obs['condition']`. For single-gene perturbation datasets, this field should contain entries such as `"gene_a"` for perturbed cells and `"ctrl"` for control cells. For two-gene perturbation datasets, entries should follow the format `"gene_a+gene_b"` for cells with two target genes, `"gene_a+ctrl"` for cells with one target gene, or `"ctrl"` for control cells.

To run scLAMBDA, a dictionary of gene embeddings in the form `gene_embeddings = {"gene_name": np.array}` must also be provided as input.

The function can be used for simulating a dataset splitting:
```python
n_split = 0
# for single-gene perturbation
adata, split = sclambda.utils.data_split(adata, split_type='single', seed=n_split)
# for two-gene perturbation
adata, split = sclambda.utils.data_split(adata, seed=n_split)
# use all available perturbations in training:
adata, split = sclambda.utils.data_split(adata, split_type='all_train', seed=n_split)
```
### Training
To train the scLAMBDA model, use:
```python
model = sclambda.model.Model(adata, 
                             gene_embeddings,
                             model_path = 'path_to_save_dir/models')

model.train()
```
### Prediction
Once trained, scLAMBDA can be used for *in silico* predictions. With a list of test perturbations `pert_test`, you can predict control cells using:
```python
res = model.predict(pert_test, return_type = 'cells')
```
Alternatively, you can generate new perturbed cells using:
```python
res = model.generate(pert_test, return_type = 'cells')
```
Setting `return_type='mean'` returns the prediction results as the mean gene expression.
## Citation
Gefei Wang, Tianyu Liu, Jia Zhao, Youshu Cheng, Hongyu Zhao. Modeling and predicting single-cell multi-gene perturbation responses with scLAMBDA.
