# Point cloud utility functions copied from:
# https://github.com/yanx27/Pointnet_Pointnet2_pytorch
#
# Original code licensed under the MIT License.
# See THIRD_PARTY_NOTICES.md for attribution and license details.
#
# Copied functions:
# - square_distance
# - index_points
# - farthest_point_sample
from __future__ import annotations
import torch

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint, random_start):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    if npoint > N:
        raise ValueError("npoint must be <= number of points")
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    if random_start:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    else:
        farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def knn_point(k, xyz, query_xyz):
    # idx: [B, M, K]
    if k > xyz.shape[1]:
        raise ValueError("k must be <= number of points")
    dist = square_distance(xyz, query_xyz)    
    idx = dist.topk(k=k, dim=1, largest=False).indices
    idx = idx.transpose(1, 2).contiguous()
    return idx
