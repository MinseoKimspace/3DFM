from __future__ import annotations

# ShapeNetCore.v2.PC15k loader copied/adapted from PointFlow:
# https://github.com/stevenygd/PointFlow/blob/master/datasets.py
#
# Keep this file intentionally close to PointFlow's dataset conventions:
# - category/split/*.npy files with 15000 surface-sampled points
# - deterministic object order shuffle with seed 38383
# - optional global vs per-shape normalization
# - first 10000 points as train_points, last 5000 as test_points
#
# See THIRD_PARTY_NOTICES.md for source/license attribution.

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


synsetid_to_cate = {
    "02691156": "airplane",
    "02773838": "bag",
    "02801938": "basket",
    "02808440": "bathtub",
    "02818832": "bed",
    "02828884": "bench",
    "02876657": "bottle",
    "02880940": "bowl",
    "02924116": "bus",
    "02933112": "cabinet",
    "02747177": "can",
    "02942699": "camera",
    "02954340": "cap",
    "02958343": "car",
    "03001627": "chair",
    "03046257": "clock",
    "03207941": "dishwasher",
    "03211117": "monitor",
    "04379243": "table",
    "04401088": "telephone",
    "02946921": "tin_can",
    "04460130": "tower",
    "04468005": "train",
    "03085013": "keyboard",
    "03261776": "earphone",
    "03325088": "faucet",
    "03337140": "file",
    "03467517": "guitar",
    "03513137": "helmet",
    "03593526": "jar",
    "03624134": "knife",
    "03636649": "lamp",
    "03642806": "laptop",
    "03691459": "speaker",
    "03710193": "mailbox",
    "03759954": "microphone",
    "03761084": "microwave",
    "03790512": "motorcycle",
    "03797390": "mug",
    "03928116": "piano",
    "03938244": "pillow",
    "03948459": "pistol",
    "03991062": "pot",
    "04004475": "printer",
    "04074963": "remote_control",
    "04090263": "rifle",
    "04099429": "rocket",
    "04225987": "skateboard",
    "04256520": "sofa",
    "04330267": "stove",
    "04530566": "vessel",
    "04554684": "washer",
    "02992529": "cellphone",
    "02843684": "birdhouse",
    "02871439": "bookshelf",
}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}


@dataclass
class PointFlowStats:
    mean: np.ndarray
    std: np.ndarray
    normalize: str
    normalize_std_per_axis: bool


class Uniform15KPC(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        subdirs: list[str],
        tr_sample_size: int = 10000,
        te_sample_size: int = 5000,
        split: str = "train",
        scale: float = 1.0,
        normalize_per_shape: bool = False,
        normalize_std_per_axis: bool = False,
        all_points_mean: np.ndarray | None = None,
        all_points_std: np.ndarray | None = None,
        input_dim: int = 3,
    ) -> None:
        self.root_dir = str(root_dir)
        self.split = split
        self.in_tr_sample_size = tr_sample_size
        self.in_te_sample_size = te_sample_size
        self.scale = scale
        self.normalize_per_shape = normalize_per_shape
        self.normalize_std_per_axis = normalize_std_per_axis
        self.input_dim = input_dim

        self.all_points = []
        self.cate_idx_lst = []
        self.all_cate_mids = []

        for cate_idx, subdir in enumerate(subdirs):
            sub_path = os.path.join(self.root_dir, subdir, self.split)
            if not os.path.isdir(sub_path):
                raise FileNotFoundError(f"Missing PointFlow split directory: {sub_path}")

            mids = sorted(
                os.path.join(self.split, name[:-4])
                for name in os.listdir(sub_path)
                if name.endswith(".npy")
            )
            if not mids:
                raise FileNotFoundError(f"No .npy files found in {sub_path}")

            for mid in mids:
                obj_fname = os.path.join(self.root_dir, subdir, mid + ".npy")
                point_cloud = np.load(obj_fname).astype("float32")
                if point_cloud.shape != (15000, input_dim):
                    raise ValueError(
                        f"Expected (15000, {input_dim}), got {point_cloud.shape} "
                        f"from {obj_fname}"
                    )

                self.all_points.append(point_cloud[None])
                self.cate_idx_lst.append(cate_idx)
                self.all_cate_mids.append((subdir, mid))

        self.all_points = np.concatenate(self.all_points, axis=0)
        self.raw_points = self.all_points.copy()
        self.cate_idx_lst = np.array(self.cate_idx_lst)

        shuffle_idx = list(range(self.all_points.shape[0]))
        random.Random(38383).shuffle(shuffle_idx)
        self.all_points = self.all_points[shuffle_idx]
        self.raw_points = self.raw_points[shuffle_idx]
        self.cate_idx_lst = self.cate_idx_lst[shuffle_idx]
        self.all_cate_mids = [self.all_cate_mids[i] for i in shuffle_idx]

        if normalize_per_shape:
            bsz = self.all_points.shape[0]
            self.all_points_mean = self.all_points.mean(axis=1).reshape(
                bsz, 1, input_dim
            )
            if normalize_std_per_axis:
                self.all_points_std = self.all_points.std(axis=1).reshape(
                    bsz, 1, input_dim
                )
            else:
                self.all_points_std = self.all_points.reshape(bsz, -1).std(
                    axis=1
                ).reshape(bsz, 1, 1)
        else:
            self.all_points_mean = (
                all_points_mean
                if all_points_mean is not None
                else self.all_points.reshape(-1, input_dim).mean(axis=0).reshape(
                    1, 1, input_dim
                )
            )
            if all_points_std is not None:
                self.all_points_std = all_points_std
            elif normalize_std_per_axis:
                self.all_points_std = self.all_points.reshape(
                    -1, input_dim
                ).std(axis=0).reshape(1, 1, input_dim)
            else:
                self.all_points_std = self.all_points.reshape(-1).std().reshape(
                    1, 1, 1
                )

        self.all_points_std = np.maximum(self.all_points_std, 1e-8).astype("float32")
        self.all_points_mean = self.all_points_mean.astype("float32")
        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std
        self.train_points = self.all_points[:, :10000]
        self.test_points = self.all_points[:, 10000:]
        self.tr_sample_size = min(10000, tr_sample_size)
        self.te_sample_size = min(5000, te_sample_size)

    def __len__(self) -> int:
        return self.all_points.shape[0]

    def __getitem__(self, idx: int) -> dict[str, object]:
        tr_out = self.train_points[idx]
        if self.tr_sample_size < tr_out.shape[0]:
            tr_idx = np.random.choice(tr_out.shape[0], self.tr_sample_size)
            tr_out = tr_out[tr_idx]

        te_out = self.test_points[idx]
        if self.te_sample_size < te_out.shape[0]:
            te_idx = np.random.choice(te_out.shape[0], self.te_sample_size)
            te_out = te_out[te_idx]

        return {
            "idx": idx,
            "train_points": tr_out.astype("float32"),
            "test_points": te_out.astype("float32"),
            "mean": self.all_points_mean[idx]
            if self.normalize_per_shape
            else self.all_points_mean.reshape(1, self.input_dim),
            "std": self.all_points_std[idx]
            if self.normalize_per_shape
            else self.all_points_std.reshape(1, -1),
            "cate_idx": self.cate_idx_lst[idx],
            "sid": self.all_cate_mids[idx],
        }


class ShapeNet15kPointClouds(Uniform15KPC):
    def __init__(
        self,
        root_dir: str | Path,
        cates: list[str] | str,
        tr_sample_size: int = 10000,
        te_sample_size: int = 5000,
        split: str = "train",
        scale: float = 1.0,
        normalize_per_shape: bool = False,
        normalize_std_per_axis: bool = False,
        all_points_mean: np.ndarray | None = None,
        all_points_std: np.ndarray | None = None,
    ) -> None:
        if cates == "all":
            subdirs = sorted(synsetid_to_cate)
        else:
            if isinstance(cates, str):
                cates = [cates]
            subdirs = [category_to_synset(cate) for cate in cates]

        super().__init__(
            root_dir=root_dir,
            subdirs=subdirs,
            tr_sample_size=tr_sample_size,
            te_sample_size=te_sample_size,
            split=split,
            scale=scale,
            normalize_per_shape=normalize_per_shape,
            normalize_std_per_axis=normalize_std_per_axis,
            all_points_mean=all_points_mean,
            all_points_std=all_points_std,
        )


def category_to_synset(category: str) -> str:
    key = category.lower()
    if key in cate_to_synsetid:
        return cate_to_synsetid[key]
    if category.isdigit():
        return category
    raise ValueError(f"Unknown PointFlow category: {category}")


def load_pointflow_15k(root: str | Path, category: str, split: str) -> np.ndarray:
    dataset = ShapeNet15kPointClouds(
        root_dir=root,
        cates=[category],
        split=split,
        tr_sample_size=10000,
        te_sample_size=5000,
    )
    return dataset.raw_points.copy()


def compute_stats(
    clouds: np.ndarray,
    normalize: str,
    normalize_std_per_axis: bool,
) -> PointFlowStats:
    normalize_per_shape = normalize == "per_shape"
    dataset = Uniform15KPC.__new__(Uniform15KPC)
    dataset.all_points = clouds.astype("float32")
    dataset.normalize_per_shape = normalize_per_shape
    dataset.normalize_std_per_axis = normalize_std_per_axis
    dataset.input_dim = clouds.shape[-1]

    if normalize_per_shape:
        bsz = clouds.shape[0]
        mean = clouds.mean(axis=1).reshape(bsz, 1, clouds.shape[-1])
        if normalize_std_per_axis:
            std = clouds.std(axis=1).reshape(bsz, 1, clouds.shape[-1])
        else:
            std = clouds.reshape(bsz, -1).std(axis=1).reshape(bsz, 1, 1)
    elif normalize == "global":
        mean = clouds.reshape(-1, clouds.shape[-1]).mean(axis=0).reshape(
            1, 1, clouds.shape[-1]
        )
        if normalize_std_per_axis:
            std = clouds.reshape(-1, clouds.shape[-1]).std(axis=0).reshape(
                1, 1, clouds.shape[-1]
            )
        else:
            std = clouds.reshape(-1).std().reshape(1, 1, 1)
    else:
        raise ValueError(f"Unknown normalization: {normalize}")

    return PointFlowStats(
        mean=mean.astype("float32"),
        std=np.maximum(std, 1e-8).astype("float32"),
        normalize=normalize,
        normalize_std_per_axis=normalize_std_per_axis,
    )


def stats_to_torch(stats: PointFlowStats) -> dict[str, object]:
    return {
        "mean": torch.from_numpy(stats.mean.copy()),
        "std": torch.from_numpy(stats.std.copy()),
        "normalize": stats.normalize,
        "normalize_std_per_axis": stats.normalize_std_per_axis,
    }


def _part_points(dataset: Uniform15KPC, part: str) -> np.ndarray:
    if part == "train":
        return dataset.train_points
    if part == "test":
        return dataset.test_points
    if part == "all":
        return dataset.all_points
    raise ValueError(f"Unknown PointFlow part: {part}")


def load_pointflow_shapes(
    root: str | Path,
    category: str,
    split: str,
    part: str,
    start_index: int,
    num_shapes: int,
    normalize: str = "global",
    normalize_std_per_axis: bool = False,
    stats: PointFlowStats | None = None,
) -> tuple[torch.Tensor, PointFlowStats]:
    normalize_per_shape = normalize == "per_shape"
    if normalize not in ("global", "per_shape"):
        raise ValueError(f"Unknown normalization: {normalize}")

    dataset = ShapeNet15kPointClouds(
        root_dir=root,
        cates=[category],
        split=split,
        tr_sample_size=10000,
        te_sample_size=5000,
        normalize_per_shape=normalize_per_shape,
        normalize_std_per_axis=normalize_std_per_axis,
        all_points_mean=None if stats is None else stats.mean,
        all_points_std=None if stats is None else stats.std,
    )
    points = _part_points(dataset, part=part)

    end_index = len(points) if num_shapes <= 0 else start_index + num_shapes
    points = points[start_index:end_index]
    if len(points) == 0:
        raise ValueError("No PointFlow shapes selected.")

    used_stats = PointFlowStats(
        mean=dataset.all_points_mean,
        std=dataset.all_points_std,
        normalize=normalize,
        normalize_std_per_axis=normalize_std_per_axis,
    )
    return torch.from_numpy(points.copy()).float().contiguous(), used_stats
