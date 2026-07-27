# PTv3 Setup

The vendored PTv3 source is pinned in `src/third_party/ptv3/UPSTREAM.md`.
Project-specific Flow Matching code lives in `src/models/ptv3_backbone.py`.

## Dependencies

First inspect the installed PyTorch and CUDA versions:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

Install the common Python dependencies:

```bash
pip install -r 3DFM/requirements.txt
```

If `torch-scatter` does not have a matching default wheel, install it from the
PyG wheel index. Then install the matching `spconv-cuXXX` package.

Verify the optional PTv3 environment:

```bash
python -c "import addict, timm, torch_scatter, spconv.pytorch; print('PTv3 dependencies OK')"
```

## Training

```bash
python 3DFM/src/train.py --config 3DFM/configs/ptv3_base_chair_n8192.yaml
```

The initial config disables FlashAttention and uses local patch size 128.
It keeps the same optimizer, EMA, data, and epoch protocol as the optimized
Base FM experiment.

## Sampling

```bash
python 3DFM/src/sample.py --checkpoint runs/ptv3_base_chair_n8192/checkpoint.pt --out-dir runs/ptv3_base_chair_n8192/eval --num-samples 32 --batch-size 1 --nfe 1 2 4 8 16 64 --save-ply
```

Rendering and visualization consume the generated `.pt` or `.ply` files and
do not require PTv3-specific changes.

## Evaluation

```bash
python 3DFM/src/eval.py --refs runs/refs/chair_test_all_n8192_fps.pt --checkpoints runs/ptv3_base_chair_n8192/checkpoint.pt --labels PTv3Base --num-samples 128 --batch-size 1 --nfe 4 8 64 --cd-only --pointflow-denorm --one-nna-cd --cd-points 64 256 1024 full --density --density-k 4 --output runs/ptv3_base_chair_n8192/metrics.json
```

## Render

```bash
python 3DFM/src/render.py --run-dir runs/ptv3_base_chair_n8192/eval/nfe_064 --output runs/ptv3_base_chair_n8192/render_nfe64.png
```
