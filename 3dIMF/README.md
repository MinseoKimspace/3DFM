# 3dIMF

Minimal research scaffold for unconditional iMF experiments on 3D point
clouds. The first target is ShapeNet single-category point clouds in raw
`[B, N, 3]` space.

The goal is to make the project shape, interfaces, configs, and model variants easy to extend before spending effort on final training performance.

## Research Goal

The main question is whether Set Transformer pooling plus cross-attention conditioning helps an unconditional iMF point-cloud backbone.

The model interface follows the official iMF direction from
`Lyy-iiis/imeanflow`: `model(z, r, t) -> (u, v)`. Here `u` is the average
velocity used by the sampler, and `v` is an instantaneous velocity auxiliary
head used by the iMF training objective.

Initial model variants:

1. `base`: simple point/set backbone.
2. `pool_xattn normal`: early point tokens produce PMA slots, late point tokens
   read their own slots through cross-attention.
3. `pool_xattn shuffle`: same model, but slots are shuffled across the batch.
4. `pool_xattn zero`: same model, but slots are replaced with zeros.

The shuffle and zero variants are meant to test whether the pooled slots carry
useful object-level information.

## Why PyG ShapeNet First

PyTorch Geometric provides a convenient `ShapeNet` dataset wrapper and standard
point-cloud transforms such as `NormalizeScale` and `FixedPoints`. This makes it
useful for early data and training-loop validation without building a custom
mesh preprocessing pipeline on day one.

Later, this can move to ShapeNetCore plus `trimesh` preprocessing for tighter
control over categories, splits, surface sampling, normalization, caching, and
paper-comparison protocols.

## Not Included Yet

- CFG.
- Class conditioning.
- Latent space.
- Point Transformer.
- PyTorch3D.
- Full paper-comparison preprocessing.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

The training, sampling, and evaluation scripts are skeletons right now. They
define the intended CLIs and wiring points, but the core functions raise
`NotImplementedError` until the first implementation pass.

Future baseline command:

```bash
python src/train.py --config configs/shapenet_chair_base.yaml
```

Future pooled cross-attention commands:

```bash
python src/train.py --config configs/shapenet_chair_pool_xattn.yaml
python src/train.py --config configs/shapenet_chair_pool_xattn_shuffle.yaml
python src/train.py --config configs/shapenet_chair_pool_xattn_zero.yaml
```

Sample from a checkpoint:

```bash
python src/sample.py --checkpoint runs/shapenet_chair_base/checkpoints/latest.pt
```

Evaluate with a simple Chamfer skeleton:

```bash
python src/eval.py --checkpoint runs/shapenet_chair_base/checkpoints/latest.pt
```

## Notes

PyG installation can depend on your local CUDA and PyTorch versions. If
`torch-geometric` fails to install from the default index, follow the official
PyG install selector for your environment.

Open3D is only used for `.ply` export in `utils/visualization.py`. If it is not
installed, training still runs until sample export is requested.
