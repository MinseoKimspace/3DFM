# DiPT Experiments

DiPT is a generative architecture built on Point Transformer V3 blocks. This
project wraps DiPT as a Flow Matching velocity model with input and output
shape `[B, N, 3]`.

## Experiment Scope

| Role | Data | Points | Config |
| --- | --- | ---: | --- |
| Primary thesis experiment | PointFlow ShapeNet | 512 | `3DFM/configs/dipt_pointflow_chair_n512.yaml` |
| Resolution extension | PointFlow ShapeNet | 2048 | `3DFM/configs/dipt_pointflow_chair_n2048.yaml` |
| Smoke test | PyG ShapeNet | 512 | `3DFM/configs/dipt_pyg_chair_n512.yaml` |
| Smoke test | PyG ShapeNet | 2048 | `3DFM/configs/dipt_pyg_chair_n2048.yaml` |

PointFlow and PyG use different normalization and sampling procedures. Do not
compare their raw metric values directly. Use PointFlow for the thesis tables
and PyG for quick implementation checks.

All four configs currently use batch size 8. Keep the batch size, optimizer,
learning rate, epochs, EMA, data split, and seed fixed across models within the
same point-count experiment.

## Install

```powershell
pip install -r 3DFM/requirements.txt
```

Install the `spconv-cuXXX` wheel matching the CUDA runtime used by PyTorch.

## Train

Choose either the 512- or 2048-point PointFlow config and keep it unchanged
across the model comparison.

```powershell
# DiPT Base
python 3DFM/src/train.py --config 3DFM/configs/dipt_pointflow_chair_n512.yaml --arch dipt_base --out-dir runs/dipt_base_pointflow_n512

# DiPT + Spatial PMA
python 3DFM/src/train.py --config 3DFM/configs/dipt_pointflow_chair_n512.yaml --arch dipt_spatial_pma --out-dir runs/dipt_spatial_pma_pointflow_n512

# DiPT + XHat-anchor Spatial PMA
python 3DFM/src/train.py --config 3DFM/configs/dipt_pointflow_chair_n512.yaml --arch dipt_xhat_anchor_pma --out-dir runs/dipt_xhat_anchor_pma_pointflow_n512

# DiPT + XHat self-conditioning
python 3DFM/src/train.py --config 3DFM/configs/dipt_pointflow_chair_n512.yaml --arch dipt_xhat_selfcond --self-cond-prob 0.5 --out-dir runs/dipt_xhat_selfcond_pointflow_n512
```

For 2048 points, replace `n512` with `n2048` in both the config and output
directory. Test the 2048-point Base for memory before launching every variant.

The XHat self-conditioning model uses a detached preliminary prediction during
training and the previous Euler step's predicted endpoint during sampling.

`aux_weight` changes the training objective. Keep it at the config default
(`0.0`) for the primary architecture comparison. If auxiliary supervision is
tested, label it as a separate ablation and compare it with an aux-only control.

## Sample

The sampling batch size only controls inference memory; it does not need to
match the training batch size.

```powershell
python 3DFM/src/sample.py --checkpoint runs/dipt_base_pointflow_n512/checkpoint.pt --out-dir runs/dipt_base_pointflow_n512/eval --num-samples 32 --batch-size 8 --nfe 1 2 4 8 16 64 --seed 123 --save-ply
```

Use 32 samples for quick visual inspection. Quantitative comparisons must use
the same number of generated samples and references, the same sampling seeds,
and the same NFE values for every model.

## Citation

When using this backbone, cite Bastico et al., *Rethinking Metrics and
Diffusion Architecture for 3D Point Cloud Generation* (arXiv:2511.05308), and
Wu et al., *Point Transformer V3: Simpler, Faster, Stronger*. Exact source and
license information is recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
