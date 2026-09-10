# DiPT Setup

DiPT backbone; Flow Matching input/output: `[B, N, 3]`.
[One-line experiment commands](README.md#pointflow-512)

## Install

```powershell
pip install -r 3DFM/requirements.txt
```

Install `spconv-cuXXX` matching PyTorch's CUDA runtime.

## Experiments

- PointFlow: 512 points primary, 2048 points secondary. PyG configs: smoke tests only.
- Keep training settings equal across models; `aux_weight: 0.0`.
- Sampling/eval load EMA weights when saved in `checkpoint.pt["model"]`.
- Self-conditioning: detached first prediction in training; previous-step endpoint, globally pooled, in sampling.

## Citation

Bastico et al., *Rethinking Metrics and Diffusion Architecture for 3D Point Cloud Generation* (arXiv:2511.05308); Wu et al., *Point Transformer V3: Simpler, Faster, Stronger*.
[Sources and licenses](THIRD_PARTY_NOTICES.md)
