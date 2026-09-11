# 3D Point Cloud Flow Matching

[Setup](DIPT_SETUP.md) | [Results](RESULTS.md) | [Third-party notices](THIRD_PARTY_NOTICES.md)

PowerShell, repository root. Skip completed steps. Every command is independent.
NFE 64; evaluation uses 512 generated chairs and 512 references.
Training: batch 8, 300 epochs, Adam LR 2e-4, gamma 0.99, EMA 0.9999, seed 42.
The 2048-point commands retain your 2048-chair export/sampling request; evaluation takes the first 512.

## PointFlow 512

### References

```powershell
python 3DFM/src/export_pointflow_refs.py --data-root 3DFM/src/data/ShapeNetCore.v2.PC15k --category Chair --split test --part test --stats-split train --shape-index 0 --num-shapes 512 --num-points 512 --subsample random --seed 123 --normalize global --output runs/refs/chair_test_S512_n512_random.pt
```

### Train

```powershell
python 3DFM/src/train.py --config 3DFM/configs/dipt_pointflow_chair_n512.yaml --arch dipt_base --seed 42 --out-dir runs/dipt_base_pointflow_n512
python 3DFM/src/train.py --config 3DFM/configs/dipt_pointflow_chair_n512.yaml --arch dipt_spatial_pma --seed 42 --out-dir runs/dipt_spatial_pma_pointflow_n512
python 3DFM/src/train.py --config 3DFM/configs/dipt_pointflow_chair_n512.yaml --arch dipt_xhat_anchor_pma --seed 42 --out-dir runs/dipt_xhat_anchor_pma_pointflow_n512
python 3DFM/src/train.py --config 3DFM/configs/dipt_pointflow_chair_n512.yaml --arch dipt_xhat_selfcond --seed 42 --out-dir runs/dipt_xhat_selfcond_pointflow_n512
python 3DFM/src/train.py --config 3DFM/configs/dipt_pointflow_chair_n512.yaml --arch dipt_xhat_selfcond_pma --seed 42 --out-dir runs/dipt_xhat_selfcond_pma_pointflow_n512
```

### Sample

```powershell
python 3DFM/src/sample.py --checkpoint runs/dipt_base_pointflow_n512/checkpoint.pt --out-dir runs/dipt_base_pointflow_n512/eval --num-samples 512 --batch-size 8 --nfe 64 --seed 123
python 3DFM/src/sample.py --checkpoint runs/dipt_spatial_pma_pointflow_n512/checkpoint.pt --out-dir runs/dipt_spatial_pma_pointflow_n512/eval --num-samples 512 --batch-size 8 --nfe 64 --seed 123
python 3DFM/src/sample.py --checkpoint runs/dipt_xhat_anchor_pma_pointflow_n512/checkpoint.pt --out-dir runs/dipt_xhat_anchor_pma_pointflow_n512/eval --num-samples 512 --batch-size 8 --nfe 64 --seed 123
python 3DFM/src/sample.py --checkpoint runs/dipt_xhat_selfcond_pointflow_n512/checkpoint.pt --out-dir runs/dipt_xhat_selfcond_pointflow_n512/eval --num-samples 512 --batch-size 8 --nfe 64 --seed 123
python 3DFM/src/sample.py --checkpoint runs/dipt_xhat_selfcond_pma_pointflow_n512/checkpoint.pt --out-dir runs/dipt_xhat_selfcond_pma_pointflow_n512/eval --num-samples 512 --batch-size 8 --nfe 64 --seed 123
```

### Eval

```powershell
python 3DFM/src/eval.py --refs runs/refs/chair_test_S512_n512_random.pt --samples runs/dipt_base_pointflow_n512/eval/nfe_064/samples.pt --max-refs 512 --max-samples 512 --cd-only --cd-points full --one-nna-cd --pointflow-denorm --cd-batch-size 4 --output runs/dipt_base_pointflow_n512/eval/metrics_nfe64.json
python 3DFM/src/eval.py --refs runs/refs/chair_test_S512_n512_random.pt --samples runs/dipt_spatial_pma_pointflow_n512/eval/nfe_064/samples.pt --max-refs 512 --max-samples 512 --cd-only --cd-points full --one-nna-cd --pointflow-denorm --cd-batch-size 4 --output runs/dipt_spatial_pma_pointflow_n512/eval/metrics_nfe64.json
python 3DFM/src/eval.py --refs runs/refs/chair_test_S512_n512_random.pt --samples runs/dipt_xhat_anchor_pma_pointflow_n512/eval/nfe_064/samples.pt --max-refs 512 --max-samples 512 --cd-only --cd-points full --one-nna-cd --pointflow-denorm --cd-batch-size 4 --output runs/dipt_xhat_anchor_pma_pointflow_n512/eval/metrics_nfe64.json
python 3DFM/src/eval.py --refs runs/refs/chair_test_S512_n512_random.pt --samples runs/dipt_xhat_selfcond_pointflow_n512/eval/nfe_064/samples.pt --max-refs 512 --max-samples 512 --cd-only --cd-points full --one-nna-cd --pointflow-denorm --cd-batch-size 4 --output runs/dipt_xhat_selfcond_pointflow_n512/eval/metrics_nfe64.json
python 3DFM/src/eval.py --refs runs/refs/chair_test_S512_n512_random.pt --samples runs/dipt_xhat_selfcond_pma_pointflow_n512/eval/nfe_064/samples.pt --max-refs 512 --max-samples 512 --cd-only --cd-points full --one-nna-cd --pointflow-denorm --cd-batch-size 4 --output runs/dipt_xhat_selfcond_pma_pointflow_n512/eval/metrics_nfe64.json
```

### Render: Each Model + Comparison Grid

```powershell
python 3DFM/src/render_mitsuba_points.py --run-dirs runs/dipt_base_pointflow_n512/eval runs/dipt_spatial_pma_pointflow_n512/eval runs/dipt_xhat_anchor_pma_pointflow_n512/eval runs/dipt_xhat_selfcond_pointflow_n512/eval --labels Base SpatialPMA XHatAnchorPMA XHatSelfCond --nfe 64 --index 2 --out-dir runs/figures/pointflow_n512_nfe64_idx2 --view side --up-axis y --camera-distance 3.2 --fov 35 --pad 1.12 --radius 0.2 --height-color --ambient 0.9 --key-light 3 --spp 128 --width 900 --height 900 --grid-cols 4
python 3DFM/src/render_mitsuba_points.py --run-dirs runs/dipt_xhat_selfcond_pma_pointflow_n512/eval --labels XHatSelfCondPMA --nfe 64 --index 2 --out-dir runs/figures/selfcond_pma_n512_nfe64_idx2 --view side --up-axis y --camera-distance 3.2 --fov 35 --pad 1.12 --radius 0.2 --height-color --ambient 0.9 --key-light 3 --spp 128 --width 900 --height 900
```

## PointFlow 2048

### References

```powershell
python 3DFM/src/export_pointflow_refs.py --data-root 3DFM/src/data/ShapeNetCore.v2.PC15k --category Chair --split test --part test --stats-split train --shape-index 0 --num-shapes 2048 --num-points 2048 --subsample random --seed 123 --normalize global --output runs/refs/chair_test_n2048_random.pt
```

### Train

```powershell
python 3DFM/src/train.py --config 3DFM/configs/dipt_pointflow_chair_n2048.yaml --arch dipt_base --seed 42 --out-dir runs/dipt_base_pointflow_n2048
python 3DFM/src/train.py --config 3DFM/configs/dipt_pointflow_chair_n2048.yaml --arch dipt_spatial_pma --seed 42 --out-dir runs/dipt_spatial_pma_pointflow_n2048
python 3DFM/src/train.py --config 3DFM/configs/dipt_pointflow_chair_n2048.yaml --arch dipt_xhat_anchor_pma --seed 42 --out-dir runs/dipt_xhat_anchor_pma_pointflow_n2048
python 3DFM/src/train.py --config 3DFM/configs/dipt_pointflow_chair_n2048.yaml --arch dipt_xhat_selfcond --seed 42 --out-dir runs/dipt_xhat_selfcond_pointflow_n2048
python 3DFM/src/train.py --config 3DFM/configs/dipt_pointflow_chair_n2048.yaml --arch dipt_xhat_selfcond_pma --seed 42 --out-dir runs/dipt_xhat_selfcond_pma_pointflow_n2048
```

### Sample

```powershell
python 3DFM/src/sample.py --checkpoint runs/dipt_base_pointflow_n2048/checkpoint.pt --out-dir runs/dipt_base_pointflow_n2048/eval --num-samples 2048 --batch-size 8 --nfe 64 --seed 123
python 3DFM/src/sample.py --checkpoint runs/dipt_spatial_pma_pointflow_n2048/checkpoint.pt --out-dir runs/dipt_spatial_pma_pointflow_n2048/eval --num-samples 2048 --batch-size 8 --nfe 64 --seed 123
python 3DFM/src/sample.py --checkpoint runs/dipt_xhat_anchor_pma_pointflow_n2048/checkpoint.pt --out-dir runs/dipt_xhat_anchor_pma_pointflow_n2048/eval --num-samples 2048 --batch-size 8 --nfe 64 --seed 123
python 3DFM/src/sample.py --checkpoint runs/dipt_xhat_selfcond_pointflow_n2048/checkpoint.pt --out-dir runs/dipt_xhat_selfcond_pointflow_n2048/eval --num-samples 2048 --batch-size 8 --nfe 64 --seed 123
python 3DFM/src/sample.py --checkpoint runs/dipt_xhat_selfcond_pma_pointflow_n2048/checkpoint.pt --out-dir runs/dipt_xhat_selfcond_pma_pointflow_n2048/eval --num-samples 2048 --batch-size 8 --nfe 64 --seed 123
```

### Eval

```powershell
python 3DFM/src/eval.py --refs runs/refs/chair_test_n2048_random.pt --samples runs/dipt_base_pointflow_n2048/eval/nfe_064/samples.pt --max-refs 512 --max-samples 512 --cd-only --cd-points full --one-nna-cd --pointflow-denorm --cd-batch-size 4 --output runs/dipt_base_pointflow_n2048/eval/metrics_nfe64.json
python 3DFM/src/eval.py --refs runs/refs/chair_test_n2048_random.pt --samples runs/dipt_spatial_pma_pointflow_n2048/eval/nfe_064/samples.pt --max-refs 512 --max-samples 512 --cd-only --cd-points full --one-nna-cd --pointflow-denorm --cd-batch-size 4 --output runs/dipt_spatial_pma_pointflow_n2048/eval/metrics_nfe64.json
python 3DFM/src/eval.py --refs runs/refs/chair_test_n2048_random.pt --samples runs/dipt_xhat_anchor_pma_pointflow_n2048/eval/nfe_064/samples.pt --max-refs 512 --max-samples 512 --cd-only --cd-points full --one-nna-cd --pointflow-denorm --cd-batch-size 4 --output runs/dipt_xhat_anchor_pma_pointflow_n2048/eval/metrics_nfe64.json
python 3DFM/src/eval.py --refs runs/refs/chair_test_n2048_random.pt --samples runs/dipt_xhat_selfcond_pointflow_n2048/eval/nfe_064/samples.pt --max-refs 512 --max-samples 512 --cd-only --cd-points full --one-nna-cd --pointflow-denorm --cd-batch-size 4 --output runs/dipt_xhat_selfcond_pointflow_n2048/eval/metrics_nfe64.json
python 3DFM/src/eval.py --refs runs/refs/chair_test_n2048_random.pt --samples runs/dipt_xhat_selfcond_pma_pointflow_n2048/eval/nfe_064/samples.pt --max-refs 512 --max-samples 512 --cd-only --cd-points full --one-nna-cd --pointflow-denorm --cd-batch-size 4 --output runs/dipt_xhat_selfcond_pma_pointflow_n2048/eval/metrics_nfe64.json
```

### Render: Each Model + Comparison Grid

```powershell
python 3DFM/src/render_mitsuba_points.py --run-dirs runs/dipt_base_pointflow_n2048/eval runs/dipt_spatial_pma_pointflow_n2048/eval runs/dipt_xhat_anchor_pma_pointflow_n2048/eval runs/dipt_xhat_selfcond_pointflow_n2048/eval --labels Base SpatialPMA XHatAnchorPMA XHatSelfCond --nfe 64 --index 2 --out-dir runs/figures/pointflow_n2048_nfe64_idx2 --view side --up-axis y --camera-distance 3.2 --fov 35 --pad 1.12 --radius 0.1 --height-color --ambient 0.9 --key-light 3 --spp 128 --width 900 --height 900 --grid-cols 4
python 3DFM/src/render_mitsuba_points.py --run-dirs runs/dipt_xhat_selfcond_pma_pointflow_n2048/eval --labels XHatSelfCondPMA --nfe 64 --index 2 --out-dir runs/figures/selfcond_pma_n2048_nfe64_idx2 --view side --up-axis y --camera-distance 3.2 --fov 35 --pad 1.12 --radius 0.1 --height-color --ambient 0.9 --key-light 3 --spp 128 --width 900 --height 900
```

## Previous-Prediction PMA

`dipt_xhat_selfcond_pma`: previous endpoint coordinates -> coordinate embedding + FPS/KNN spatial PMA -> cross-attention into current DiPT features. Requires new training; existing models are unchanged. Reuses the four configs (`aux_weight: 0`, `self_cond_prob: 0.5`). First step and `zero` mode skip the conditioning branch. Training uses a detached preliminary prediction at the same noisy state; sampling reuses the previous step's prediction. `--save-xhat` remains exclusive to the same-pass XHatAnchorPMA model below.

PyG training alternatives (use PyG test references for evaluation, without `--pointflow-denorm`):

```powershell
python 3DFM/src/train.py --config 3DFM/configs/dipt_pyg_chair_n512.yaml --arch dipt_xhat_selfcond_pma --seed 42 --out-dir runs/dipt_xhat_selfcond_pma_pyg_n512
python 3DFM/src/train.py --config 3DFM/configs/dipt_pyg_chair_n2048.yaml --arch dipt_xhat_selfcond_pma --seed 42 --out-dir runs/dipt_xhat_selfcond_pma_pyg_n2048
```

Optional diagnostics after training: separate normal/shuffle/zero rollouts; same-state velocity comparison along the normal rollout. Shuffle needs batch size > 1.

```powershell
python 3DFM/src/sample_intervention.py --checkpoint runs/dipt_xhat_selfcond_pma_pointflow_n512/checkpoint.pt --out-dir runs/dipt_xhat_selfcond_pma_pointflow_n512/intervention --num-samples 32 --batch-size 8 --nfe 16 --seed 123
python 3DFM/src/diagnose_guidance.py --checkpoint runs/dipt_xhat_selfcond_pma_pointflow_n512/checkpoint.pt --out runs/dipt_xhat_selfcond_pma_pointflow_n512/diagnostic/guidance_nfe16.json --num-samples 32 --batch-size 8 --nfe 16 --seed 123
```

## Anchor Intermediate Predictions

No retraining. Saves current states and predicted final shapes; prints checkpoint `aux_weight`. Diagnostic timings are not benchmarks. Rendering uses a shared camera/scale: current state left, prediction right, generated final at the bottom (not ground truth).

```powershell
python 3DFM/src/sample.py --checkpoint runs/dipt_xhat_anchor_pma_pointflow_n512/checkpoint.pt --out-dir runs/dipt_xhat_anchor_pma_pointflow_n512/diagnostic_xhat --num-samples 8 --batch-size 8 --nfe 16 --seed 123 --save-xhat
python 3DFM/src/render_mitsuba_points.py --xhat-dir runs/dipt_xhat_anchor_pma_pointflow_n512/diagnostic_xhat/nfe_016 --index 2 --out-dir runs/figures/anchor_xhat_n512_idx2 --view side --up-axis y --radius 0.2 --height-color --ambient 0.9 --key-light 3 --spp 64 --width 600 --height 600
```

## Notes

- Render another chair: change `--index 2` and the output directory's `idx2`; try `--view iso` for a three-quarter view.
- Eval must print 512 samples and 512 refs. Caps do not create missing samples.
- Use the same sampling seeds and refs across models.
- Record each JSON's `NFE=64` row in [RESULTS.md](RESULTS.md): MMD-CD (lower), COV-CD (higher), 1-NNA-CD (real/real reference near 50%), Time/sample.
