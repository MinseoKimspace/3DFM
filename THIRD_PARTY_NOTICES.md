# Third-party Notices

This project adapts small utility components from open-source research code.
The adapted code is kept minimal and modified for this project's tensor shapes
and experiment interface.

## Set Transformer MAB/PMA

Source: https://github.com/juho-lee/set_transformer  
Paper: Lee et al., "Set Transformer: A Framework for Attention-based
Permutation-Invariant Neural Networks", ICML 2019.  
License: MIT License  
Copyright: Copyright (c) 2020 Juho Lee

Used for: MAB/PMA-style attention pooling.  
Notes: The implementation in this project is modified for point-cloud Flow
Matching experiments.

## PointNet++ PyTorch Utilities

Source: https://github.com/yanx27/Pointnet_Pointnet2_pytorch  
License: MIT License  
Copyright: Copyright (c) 2019 benny

Used for: `square_distance`, `index_points`, `farthest_point_sample`,
and `query_ball_point`.

Notes: These low-level point cloud utility functions are copied or lightly
reformatted for this project's `[B, N, C]` tensor layout.

## LION Metrics Backend

Source: https://github.com/nv-tlabs/LION

This project does not vendor LION metric source code. Evaluation code may import
LION as an external backend when the user provides a local LION checkout.

## PointFlow ShapeNetCore.v2.PC15k Loader

Source: https://github.com/stevenygd/PointFlow  
File: `datasets.py`  
License: MIT License

Used for: ShapeNetCore.v2.PC15k loading conventions and the
`Uniform15KPC` / `ShapeNet15kPointClouds` dataset structure.

Notes: The implementation in this project is copied/adapted to return tensors
for the local Flow Matching training loop while preserving PointFlow's split,
normalization, and 10k/5k point-pool conventions.
