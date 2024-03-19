#!/usr/bin/env python
"""
PointNet++
Reference: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""
from typing import List, Optional
import numpy as np
import torch
from torch import nn
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
    @param num_centroids: the number of sampled points (centroids)
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


def sample_and_group(num_centroids: int, radius: float, num_samples: int, points_xyz: torch.Tensor,
                     points_features: Optional[torch.Tensor] = None, return_fps: bool = False):
    """
    FPS sampling + Grouping for a batch of point clouds.
    @param num_centroids: number of centroids in FPS
    @param radius: local region radius
    @param num_samples: max sample number in a local region
    @param points_xyz: (batch_size, num_points, 3) - coordinates of point clouds
    @param points_features: (batch_size, num_points, D) - features of point clouds
    @param return_fps: True - return FPS results. False - return
    @return:
        centroids_xyz: (batch_size, num_centroids, 3)
        new_points: (batch_size, num_centroids, num_samples, C+D)

    """
    B, N, C = points_xyz.shape
    S = num_centroids

    # find centroids of local regions
    fps_inds = farthest_point_sample(points_xyz, num_centroids)  # [B, num_centroids, C]

    # centroid points
    centroids_xyz = index_points(points_xyz, fps_inds)

    # query a fixed number of samples for each centroid
    idx = query_ball(radius, num_samples, points_xyz, centroids_xyz)

    # each point in the xyz has a centroid label
    grouped_xyz = index_points(points_xyz, idx)  # [B, num_centroids, num_samples, C]

    # centralize the coordinates of the points by subtracting the local centroids
    grouped_xyz_norm = grouped_xyz - centroids_xyz.view(B, S, 1, C)

    if points_features is not None:
        grouped_points = index_points(points_features, idx)
        # concatenate the xyz coordinates and features of the point clouds
        # [batch_size, num_centroids, num_samples, C+D]
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz_norm
    if return_fps:
        return centroids_xyz, new_points, grouped_xyz, fps_inds
    else:
        return centroids_xyz, new_points


def sample_and_group_all(points_xyz, points_features):
    """
    Sample and group all points into one local region. There is only one centroid.
    @param points_xyz: (batch_size, num_points, 3) - coordinates of point clouds
    @param points_features: (batch_size, num_points, D) - features of point clouds
    @return:
        centroid_xyz: (batch_size, 1, 3)
        new_points:  (batch_size, 1, num_samples, C+D)
    """
    device = points_xyz.device
    B, N, C = points_xyz.shape
    centroid_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = points_xyz.view(B, 1, N, C)
    if points_features is not None:
        new_points = torch.cat([grouped_xyz, points_features.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return centroid_xyz, new_points


class PointNetSetAbstraction(nn.Moudule):
    """
    A hierarchical feature learning framework that aims to capture local context at different scales.
    A set abstraction block is made of three key layers: Sampling layer, Grouping layer and PointNet layer.
    Sampling layer: use iterative farthest point sampling (FPS) to choose a subset of query points (centroids).
    Grouping layer: use Ball Query to finds all points that are within a radius to the query points (centroids).
    PointNet layer: for local pattern learning.
    """

    def __init__(self, num_centroids: torch.Tensor, radius: float, num_samples: int, in_channels,
                 mlp_channels: List[int], group_all: bool):
        super().__init__()
        self.num_centroids = num_centroids
        self.radius = radius
        self.num_samples = num_samples
        self.in_channels = in_channels

        self.mlp_layers = nn.ModuleList()
        for out_channels in mlp_channels:
            self.mlp_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                )
            )
            in_channels = out_channels
        self.group_all = group_all

    def forward(self, points_xyz, points_features):
        """
        @param points_xyz: (B, C, N), i.e. (batch_size, num_coordinates, num_points) - coordinates of point clouds
        @param points_features: (B, D, N) i.e. (batch_size, num_features, num_points) - features of point clouds
        @return: 
        """
        points_xyz = points_xyz.permute(0, 2, 1)  # (batch_size, num_points, 3). 3=num_coordinates
        if points_features is not None:
            points_features = points_features.permute(0, 2, 1)  # (batch_size, num_points, num_features)

        if self.group_all:
            centroid_xyz, new_points = sample_and_group_all(points_xyz, points_features)
        else:
            centroid_xyz, new_points = sample_and_group(self.num_centroids, self.radius, self.num_samples,
                                                        points_xyz, points_features)
        # centroid_xyz: (batch_size, num_samples, 3)
        # new_points: (batch_size, num_centroids, num_samples, 3 + num_features)
        new_points = new_points.permute(0, 3, 2, 1)  # (batch_size, 3+num_features, num_samples, num_centroids)

        for layer in self.mlp_layers:
            new_points = layer(new_points)
        # new_points: (batch_size, mlp_channels[-1], num_samples, num_centroids)
        new_points = torch.max(new_points, 2)[0]  # (batch_size, mlp_channels[-1], 1, num_centroids)
        centroid_xyz = centroid_xyz.permute(0, 2, 1)  # (batch_size, 3, num_samples)

        return centroid_xyz, new_points


class PointNetSetAbstractionMSG(nn.Module):
    """
    Multi-scale grouping (MSG).
    A simple but effective way to capture multiscale patterns is to apply grouping layers with different scales
    followed by according PointNets to extract features of each scale.
    Features at different scales are concatenated to form a multi-scale feature.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
