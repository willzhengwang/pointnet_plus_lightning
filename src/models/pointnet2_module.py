#!/usr/bin/env python
"""
PointNet++
Reference: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""
import numpy as np
import torch
from src.models.pointnet_module import FeatureNet, TNet


def pc_normalize(pc: np.ndarray):
    """
    Normalizes a point cloud by subtracting the mean and scaling to unit radius.
    @param pc: num_points * num_features array of a point clout
    @return:
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid

    # Find the furthest point from the origin to use as the radius
    radius = max(np.max(np.linalg.norm(pc, axis=1)), np.finfo(float).eps)  # Avoid division by zero
    pc = pc / radius  # Scale the point cloud to unit radius
    return pc


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Calculate the squared Euclidean distance for every pair of points between two point clouds.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    @param src: (batch_size, N, channels), N is the number of points in the source point cloud
    @param dst: (batch_size, M, channels), M is the number of points in the destination point cloud
    @return:
        dist: [batch_size, N, M], per-point square distance
    """
    B, N, _ = src.shape  # B: batch_size
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))  # (B, N, M)
    dist += torch.sum(src ** 2, -1).view(B, N, 1)  # (B, N, M)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)  # (B, N, M)
    return dist


def index_points(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Indexing points
    @param points: (batch_size, num_points, num_channels)
    @param indices: (batch_size, num_indices)
    @return:
        indexed_points: (batch_size, num_indices, num_channels)
    """
    device = points.device
    indices_expanded = indices.unsqueeze(-1).repeat(1, 1, points.shape[-1]).to(device)
    new_points = torch.gather(points, 1, indices_expanded)
    return new_points


def farthest_point_sample(xyz: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS)
    @param xyz:
    @param num_samples: the number of sampled points
    @return:
        sample_inds: (batch_size, num_samples) of sample point indices
    """
    device = xyz.device
    B, N, C = xyz.shape  # batch_size, num_points, num_channels
    sample_inds = torch.zeros(B, num_samples, dtype=torch.long).to(device)

    # Initialize the distances between points to the selected sample points with max inf.
    distance = torch.zeros(B, N).to(device) + float('inf')

    # For each point cloud, randomly select a point as the farthest point
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(num_samples):
        # set the new centroid as the last farthest point
        sample_inds[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)

        # calculate the distances between the points with the centroid (i.e. the newly selected sample)
        dist = torch.sum((xyz - centroid) ** 2, -1)

        # Keep updating the distance array, which enables the selection of farthest point to all existing centroids.
        mask = dist < distance
        distance[mask] = dist[mask]

        farthest = torch.max(distance, -1)[1]
    return sample_inds
