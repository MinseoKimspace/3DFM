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

## Visual Sampling

Use 32 samples for a quick visual NFE sweep. `sample.py` creates the noise
once, so every NFE below uses the same 32 initial point clouds.

```bash
python 3DFM/src/sample.py --checkpoint runs/ptv3_base_chair_n8192/checkpoint.pt --out-dir runs/ptv3_base_chair_n8192/eval --num-samples 32 --batch-size 1 --nfe 1 2 4 8 16 64 --seed 123 --save-ply
```

Rendering and visualization consume the generated `.pt` or `.ply` files and
do not require PTv3-specific changes.

## Quick Evaluation

Use a balanced 128 generated / 128 reference evaluation to check that the
checkpoint and metric pipeline work. This is a screening run, not the final
comparison.

```bash
python 3DFM/src/eval.py --refs runs/refs/chair_test_all_n8192_fps.pt --checkpoints runs/ptv3_base_chair_n8192/checkpoint.pt --labels PTv3Base --num-samples 128 --max-refs 128 --batch-size 1 --nfe 4 8 64 --seed 123 --cd-only --pointflow-denorm --one-nna-cd --cd-points 64 256 1024 full --density --density-k 4 --output runs/ptv3_base_chair_n8192/metrics_quick.json
```

## Formal Base Comparison

The formal comparison uses 512 generated samples and 512 references. Put Base
and PTv3 in the same command so they use identical noise at every NFE. Run the
same evaluation with seeds 123, 456, and 789.

```bash
python 3DFM/src/eval.py --refs runs/refs/chair_test_all_n8192_fps.pt --checkpoints runs/base_chair_n8192/checkpoint.pt runs/ptv3_base_chair_n8192/checkpoint.pt --labels Base PTv3Base --num-samples 512 --max-refs 512 --batch-size 1 --nfe 4 8 64 --seed 123 --cd-only --pointflow-denorm --one-nna-cd --cd-points 64 256 1024 full --density --density-k 4 --output runs/ptv3_base_chair_n8192/metrics_seed123.json
```

Repeat the command with:

```text
--seed 456 --output runs/ptv3_base_chair_n8192/metrics_seed456.json
--seed 789 --output runs/ptv3_base_chair_n8192/metrics_seed789.json
```

Keep the raw values in these JSON files. Record the mean and standard deviation
across the three seeds in `RUNG0_RESULTS.md`. `METRIC_PROTOCOL.md` contains only
the fixed evaluation method, not numeric results.

## Render

```bash
python 3DFM/src/render.py --run-dir runs/ptv3_base_chair_n8192/eval/nfe_064 --output runs/ptv3_base_chair_n8192/render_nfe64.png
```
