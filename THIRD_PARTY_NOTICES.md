# Third-Party Notices

- **Diffusion Point Transformer (DiPT)**: required model files are vendored in
  `3DFM/src/third_party/dipt` from
  [DiffusionPointTransformer](https://github.com/matteo-bastico/DiffusionPointTransformer)
  at commit `cff9fe070ddd073ca5dc4dbdd0ebf2ccb25b7225`, under the MIT License.
  Imports and registry dependencies were minimally adapted for a self-contained
  package; project-specific Flow Matching integration is in
  `3DFM/src/models/dipt_backbone.py`.
- **Point Transformer V3**: DiPT includes PTv3 blocks and required Pointcept
  utilities. Original project:
  [PointTransformerV3](https://github.com/Pointcept/PointTransformerV3), MIT License.
- **PointFlow**: `3DFM/src/data/pointflow_pc15k.py` adapts the
  [PointFlow dataset loader](https://github.com/stevenygd/PointFlow), MIT License.
- **Set Transformer**: MAB/PMA in `3DFM/src/models/spatial_pma.py` adapts
  [set_transformer](https://github.com/juho-lee/set_transformer), MIT License.
- **PointNet++ utilities**: `3DFM/src/models/point_ops.py` adapts utilities from
  [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch),
  MIT License.
- **LION metrics**: `3DFM/src/metrics/lion_backend.py` optionally imports an external
  [LION](https://github.com/nv-tlabs/LION) checkout; LION code is not vendored.

The complete DiPT license is retained in `3DFM/src/third_party/dipt/LICENSE`.
