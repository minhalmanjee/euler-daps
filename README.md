# DAPS

Code to reproduce **DAPS** (degree-aware post-scoring) results on **LANL** and **OpTC**.

## Layout

- `lanl_experiments/` — main entry (`run.py`, `spinup.py`), `models/`, `loaders/` (includes `edge_dev.py` / DAPS)
- `data/` — placeholders for raw and processed datasets (not shipped)
- `environment.yml` / `requirements.txt` / `install_env.sh` — environment setup

## Setup

```bash
# Option A
conda env create -f environment.yml
conda activate euler

# Option B
bash install_env.sh

cd lanl_experiments
```

### Data

1. **LANL: https://csr.lanl.gov/data/cyber1/**: place `auth.txt` and `redteam.txt` under `data/datasets/lanl/`, then run `python -m loaders.split` (see `lanl_experiments/README.md` and path constants in `loaders/`).
2. **OpTC:: https://github.com/cybermonic/CyberGFM/tree/main/optc_preprocess**: place flow CSV under `data/datasets/optc/cybergfm/`, then run `python -m loaders.split_optc_flow`.

Raw datasets are **not** included.

## Reproduce (5 seeds)

Paper defaults used below: `fpweight=0.6`, DAPS `--edge-dev-alpha 0.1`, LANL `--zscore-cap 2`, OpTC `--zscore-cap 10`. Snapshot `--delta 0.5` hours (= 1800 s).

GPU layout: by default workers use GPUs `0..W-1` and the master uses GPU `W` (so `-w 3` needs **4 GPUs**). On a single GPU use `-w 1 -g 1` only if you have **2 GPUs**, or use `--cpu`.

### GPU (multi-GPU)

```bash
cd lanl_experiments
PY=python

# LANL EULER (GCN+GRU)
for s in 1 2 3 4 5; do
  $PY -u run.py --dataset LANL -d 0.5 -e GCN -r GRU --seed $s -t 1 -w 3 \
    --fpweight 0.6 \
    --save-test-scores figures/baseline_gru_fw0.6_seed${s}.npz
done

# LANL DAPS
for s in 1 2 3 4 5; do
  $PY -u run.py --dataset LANL -d 0.5 -e GCN -r GRU --seed $s -t 1 -w 3 \
    --fpweight 0.6 --edge-dev --edge-dev-alpha 0.1 --zscore-cap 2 \
    --save-test-scores figures/edgedev_gru_a0.1_c2_seed${s}.npz
done

# OpTC EULER
for s in 1 2 3 4 5; do
  $PY -u run.py --dataset OPTC -d 0.5 -e GCN -r GRU --seed $s -t 1 -w 3 \
    --fpweight 0.6 \
    --save-test-scores figures/optc_flow/baseline_gru_fw0.6_seed${s}.npz
done

# OpTC DAPS
for s in 1 2 3 4 5; do
  $PY -u run.py --dataset OPTC -d 0.5 -e GCN -r GRU --seed $s -t 1 -w 3 \
    --fpweight 0.6 --edge-dev --edge-dev-alpha 0.1 --zscore-cap 10 \
    --save-test-scores figures/optc_flow/edgedev_gru_a0.1_c10_seed${s}.npz
done
```

GCN+NONE: replace `-r GRU` with `-r NONE`.

### CPU

Use **`--cpu`**. With `--cpu`, CUDA layout checks are skipped; keep `-w` modest (e.g. 1–3). Example:

```bash
cd lanl_experiments

# LANL DAPS on CPU
python -u run.py --dataset LANL -d 0.5 -e GCN -r GRU --seed 1 -t 1 \
  --cpu -w 1 --fpweight 0.6 \
  --edge-dev --edge-dev-alpha 0.1 --zscore-cap 2 \
  --save-test-scores figures/edgedev_gru_a0.1_c2_seed1.npz

# OpTC DAPS on CPU
python -u run.py --dataset OPTC -d 0.5 -e GCN -r GRU --seed 1 -t 1 \
  --cpu -w 1 --fpweight 0.6 \
  --edge-dev --edge-dev-alpha 0.1 --zscore-cap 10 \
  --save-test-scores figures/optc_flow/edgedev_gru_a0.1_c10_seed1.npz
```

Do **not** omit `--cpu` on a machine without enough GPUs for `-w` workers + 1 master — that raises a GPU layout `RuntimeError`.

## Not included

Raw datasets, `*.npz` score dumps, `model_save.pkl`, and train-degree pickles (generated at runtime).
