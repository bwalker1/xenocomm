# Xenocomm

Xenocomm identifies human-to-mouse signaling in xenograft single-cell RNA-seq
data using a Bayesian model of ligand abundance, receptor expression, and
downstream target programs.

## Installation

Use Python 3.11 or 3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install . jupyterlab
```

On Windows, activate with `.venv\Scripts\activate` instead. The package requires
Python 3.11 or newer.

For NVIDIA GPU support on Linux or WSL2, install `'.[gpu]'` instead of `.`.

The CellChat/LIANA comparison also needs R, with `R` and `Rscript` on `PATH`
before installing its dependencies.

On Ubuntu 24.04, install R and the native build dependencies first:

```bash
sudo apt-get update
sudo apt-get install -y python3-dev r-base r-base-dev build-essential gfortran cmake pkg-config \
  libcurl4-openssl-dev libssl-dev libxml2-dev libffi-dev libreadline-dev \
  libfontconfig1-dev libfreetype6-dev libpng-dev libtiff-dev libjpeg-dev \
  libharfbuzz-dev libfribidi-dev libglpk-dev
```

Then, in the activated Python environment, configure a writable R library and
install the comparison packages:

```bash
export R_HOME="$(R RHOME)"
export R_LIBS_USER="$VIRTUAL_ENV/lib/R/library"
mkdir -p "$R_LIBS_USER"
python -m pip install '.[comparison]'
Rscript scripts/install_cellchat.R
```

Set these two environment variables again when starting a new shell to run the
comparison notebook. Other operating systems need equivalent R development
tools and native libraries.

This installs LIANA 1.7.3 and CellChat 2.2.0.9001. Use `'.[gpu,comparison]'`
for the comparison with GPU support.

## Data and notebooks

Download the [PDX data](https://www.dropbox.com/scl/fi/h05y4ythaoucsnvwjb92b/xenocomm-data-melanoma-pdx-10k.tar.gz?rlkey=4pq8309x914x615yxh7x9rs4x&dl=1)
and extract `adata_human.h5ad` and `adata_mouse.h5ad` into
`notebooks/data/melanoma_pdx_10k/`. The archive contains 10,000 cells per
species. All PDX analyses use these distributed files; the 10x data and
interaction databases download automatically.
The notebooks start from raw counts and compute expression preprocessing themselves.

Launch the notebooks from the repository root:

```bash
python -m jupyterlab notebooks
```

| Notebook | Analysis |
| --- | --- |
| [pdx.ipynb](notebooks/pdx.ipynb) | PDX ligand, receptor, and target analyses |
| [10x.ipynb](notebooks/10x.ipynb) | Native 10x human–mouse mixture analysis |
| [ablation.ipynb](notebooks/ablation.ipynb) | Receptor–target ablation across ten seeds |
| [bootstrap.ipynb](notebooks/bootstrap.ipynb) | Cell and paired-sample bootstrap stability |
| [mixtures.ipynb](notebooks/mixtures.ipynb) | PDX and 10x cell-mixture studies |
| [comparison.ipynb](notebooks/comparison.ipynb) | SI Figure 8 comparison workflow with CellChat and LIANA |

Each notebook runs independently. PDX workflows save results under
`notebooks/outputs/<notebook>_10k/`; `10x.ipynb` uses
`notebooks/outputs/10x_hgmm/`. The comparison uses five matched sampling seeds,
100 CellChat bootstrap iterations, and 1,000 LIANA permutations.

The notebooks fix the manuscript batch sizes at 1,024 mouse cells for PDX and
687 for 10x, with 4,096 and 687 validation cells respectively. The PDX UMAPs are
computed from the rebuilt expression using 2,000 variable genes, 30 principal
components, 15 neighbors, and random seed 0.

## Use Xenocomm with your data

When setting up custom human and mouse `AnnData` files, put raw counts in `.X`, gene symbols
in `var_names`, and Ensembl IDs in `var["gene_id"]` or `var["gene_ids"]`.

```python
import anndata as ad
import scanpy as sc
import xenocomm as xc

mouse = ad.read_h5ad("mouse_counts.h5ad")
human = ad.read_h5ad("human_counts.h5ad")
for data in (mouse, human):
    data.layers["counts"] = data.X.copy()
    data.raw = None
    sc.pp.normalize_total(data, target_sum=10_000)
    sc.pp.log1p(data)

network = xc.prepare_network(mouse, human)
abundance = xc.compute_ligand_abundance(
    mouse, human, network["ligands"], network["human_ligands"],
    network["ligand_receptor_matrix"],
)
model = xc.XenocommModel(mouse, **network, mean_ligand=abundance, epochs=5)
model.train()

results = xc.ligand_result_table(
    model.ligands, model.sample(2_000), model.mean_ligand_np,
    model.ligand_receptor_matrix_np,
    xc.get_receptor_sensitivity(model.get_parameters()),
)
results.loc[results["called"]].to_csv("xenocomm_ligands.csv", index=False)
```

The result table reports ligand effects, uncertainty, human signaling fractions,
and enrichment classifications.

The model defaults to batches of 1,024 mouse cells. If you pass `validation_mouse`
to `train`, its cell count must be a positive multiple of the model's batch size.
