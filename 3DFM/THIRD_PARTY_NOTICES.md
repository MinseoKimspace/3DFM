# Third-Party Notices

- **DiPT / Point Transformer V3**: vendored in `src/third_party/dipt` from
  [DiffusionPointTransformer](https://github.com/matteo-bastico/DiffusionPointTransformer)
  at commit `cff9fe070ddd073ca5dc4dbdd0ebf2ccb25b7225`, under the MIT License.
- **PointFlow**: `src/data/pointflow_pc15k.py` adapts the
  [PointFlow dataset loader](https://github.com/stevenygd/PointFlow), MIT License.
- **Set Transformer**: MAB/PMA in `src/models/spatial_pma.py` adapts
  [set_transformer](https://github.com/juho-lee/set_transformer), MIT License.
- **PointNet++ utilities**: `src/models/point_ops.py` adapts utilities from
  [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch),
  MIT License.
- **LION metrics**: `src/metrics/lion_backend.py` optionally imports an external
  [LION](https://github.com/nv-tlabs/LION) checkout; LION code is not vendored.

The complete DiPT license is retained in `src/third_party/dipt/LICENSE`.
