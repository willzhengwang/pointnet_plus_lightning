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


def farthest_point_sample(xyz: torch.Tensor, num_centroids: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) for selecting a number of centroids from the input point clouds.
    @param xyz: (batch_size, num_points, num_channels=3) - a batch of point clouds
    @param num_centroids: the number of sampled points
    @return:
        centroid_inds: (batch_size, num_samples) of sample point indices
    """
    device = xyz.device
    B, N, C = xyz.shape  # batch_size, num_points, num_channels
    centroid_inds = torch.zeros(B, num_centroids, dtype=torch.long).to(device)

    # Initialize the distances between points to the selected sample points with max inf.
    distance = torch.zeros(B, N).to(device) + float('inf')

    # For each point cloud, randomly select a point as the farthest point
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_inds = torch.arange(B, dtype=torch.long).to(device)

    for i in range(num_centroids):
        # set the new centroid as the last farthest point
        centroid_inds[:, i] = farthest
        centroid = xyz[batch_inds, farthest, :].view(B, 1, C)

        # calculate the distances between the points with the centroid (i.e. the newly selected sample)
        dist = torch.sum((xyz - centroid) ** 2, -1)

        # Keep updating the distance array, which enables the selection of farthest point to all existing centroids.
        mask = dist < distance
        distance[mask] = dist[mask]

        farthest = torch.max(distance, -1)[1]
    return centroid_inds


def query_ball(radius, num_samples, xyz, query_xyz):
    """
    Find sample points within a radius for each query point (centroid).
    @param radius: local region radius
    @param num_samples: max sample number in local region
    @param xyz: (batch_size, num_points, num_channels=3) - a batch of point clouds
    @param query_xyz: query points, [B, S, 3]
    @return:
        group_inds: (batch_size, num_centroids, num_samples) the centroid index of each point
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = query_xyz.shape
    group_inds = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])

    # calculate the squared distance for every pair of points between two point clouds.
    sqr_dists = square_distance(query_xyz, xyz)  # (batch_size, S, N)
    group_inds[sqr_dists > radius ** 2] = N
    group_inds = group_inds.sort(dim=-1)[0][:, :, :num_samples]

    # if the number of sample points within the radius is < num_samples, fill the remaining with the first point
    group_first = group_inds[:, :, 0].view(B, S, 1).repeat([1, 1, num_samples])
    mask = group_inds == N
    group_inds[mask] = group_first[mask]
    return group_inds


