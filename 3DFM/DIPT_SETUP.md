# DiPT Setup

DiPT is the only backbone. Its source and commit are recorded in
`src/third_party/dipt/UPSTREAM.md`.

## Configs

| Data | Points | Config |
| --- | ---: | --- |
| PyG ShapeNet | 512 | `configs/dipt_pyg_chair_n512.yaml` |
| PyG ShapeNet | 2048 | `configs/dipt_pyg_chair_n2048.yaml` |
| PointFlow ShapeNet | 512 | `configs/dipt_pointflow_chair_n512.yaml` |
| PointFlow ShapeNet | 2048 | `configs/dipt_pointflow_chair_n2048.yaml` |

PyG uses per-shape `NormalizeScale`; PointFlow uses its global training-set
statistics. Treat them as separate data protocols rather than directly mixing
their raw metric values.

## Install

```powershell
pip install -r 3DFM/requirements.txt
```

Install the `spconv-cuXXX` wheel matching the CUDA runtime used by PyTorch.

## Train

Choose one config and one model variant:

```powershell
# Base
python 3DFM/src/train.py --config 3DFM/configs/dipt_pointflow_chair_n2048.yaml

# Spatial PMA
python 3DFM/src/train.py --config 3DFM/configs/dipt_pointflow_chair_n2048.yaml --arch dipt_spatial_pma --out-dir runs/dipt_spatial_pma_pointflow_n2048

# XHat Anchor PMA with auxiliary supervision
python 3DFM/src/train.py --config 3DFM/configs/dipt_pointflow_chair_n2048.yaml --arch dipt_xhat_anchor_pma --aux-weight 0.5 --out-dir runs/dipt_xhat_anchor_pma_pointflow_n2048

# True previous-step XHat self-conditioning
python 3DFM/src/train.py --config 3DFM/configs/dipt_pointflow_chair_n2048.yaml --arch dipt_xhat_selfcond --self-cond-prob 0.5 --out-dir runs/dipt_xhat_selfcond_pointflow_n2048
```

The self-conditioning model uses a detached preliminary prediction during
training and the previous Euler step's predicted endpoint during sampling.

## Sample

```powershell
python 3DFM/src/sample.py --checkpoint runs/dipt_pointflow_chair_n2048/checkpoint.pt --out-dir runs/dipt_pointflow_chair_n2048/eval --num-samples 32 --batch-size 4 --nfe 1 2 4 8 16 64 --seed 123 --save-ply
```

`eval.py`, `render.py`, `visualize.py`, and the PMA intervention scripts load
the architecture directly from each checkpoint.

## Citation

When using the DiPT backbone, cite Bastico et al., *Rethinking Metrics and
Diffusion Architecture for 3D Point Cloud Generation* (arXiv:2511.05308), and
Point Transformer V3. Source and license details are in
`THIRD_PARTY_NOTICES.md`.
