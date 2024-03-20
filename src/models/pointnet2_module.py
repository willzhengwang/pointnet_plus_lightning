#!/usr/bin/env python
"""
PointNet++
Reference: https://github.com/yanx27/Pointnet_Pointnet2_pytorch
"""
from typing import List, Optional
import numpy as np
import torch
from torch import nn


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
    @param indices: (batch_size, idx_dim0, idx_dim1, ...)
    @return:
        indexed_points: (batch_size, idx_dim0, idx_dim1, ..., num_channels)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(indices.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(indices.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, indices, :]
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
    group_inds = query_ball(radius, num_samples, points_xyz, centroids_xyz)  # (batch_size, num_centroids, num_samples)

    # each point in the xyz has a centroid label
    grouped_xyz = index_points(points_xyz, group_inds)  # [B, num_centroids, num_samples, C]

    # centralize the coordinates of the points by subtracting the local centroids
    grouped_xyz_norm = grouped_xyz - centroids_xyz.view(B, S, 1, C)

    if points_features is not None:
        grouped_points = index_points(points_features, group_inds)
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


class PointNetSetAbstraction(nn.Module):
    """
    A hierarchical feature learning framework that aims to capture local context at different scales.
    A set abstraction block is made of three key layers: Sampling layer, Grouping layer and PointNet layer.
    Sampling layer: use iterative farthest point sampling (FPS) to choose a subset of query points (centroids).
    Grouping layer: use Ball Query to finds all points that are within a radius to the query points (centroids).
    PointNet layer: for local pattern learning.
    """

    def __init__(self, num_centroids: int, radius: float, num_samples: int, in_channels: int,
                 mlp_channels: List[int], group_all: bool):
        super().__init__()
        self.num_centroids = num_centroids
        self.radius = radius
        self.num_samples = num_samples
        self.in_channels = in_channels

        self.mlp = nn.ModuleList()
        for out_channels in mlp_channels:
            self.mlp.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
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

        for layer in self.mlp:
            new_points = layer(new_points)
        # new_points: (batch_size, mlp_channels[-1], num_samples, num_centroids)
        # extract the global features of each local region (centroid) == local features
        new_points = torch.max(new_points, 2)[0]
        centroid_xyz = centroid_xyz.permute(0, 2, 1)  # (batch_size, 3, num_samples)
        return centroid_xyz, new_points


class PointNetSetAbstractionMSG(nn.Module):
    """
    Multi-scale Grouping (MSG).
    A simple but effective way to capture multiscale patterns is to apply grouping layers with different scales
    followed by according PointNets to extract features of each scale.
    Features at different scales are concatenated to form a multi-scale feature.
    """
    def __init__(self, num_centroids: int, radius_list: List[float], num_samples_list: List[int], in_channels: int,
                 mlp_channels_list: List[List[int]]):
        # 512, [0.1, 0.2, 0.4], [16, 32, 128], 3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        super().__init__()
        self.num_centroids = num_centroids
        self.radius_list = radius_list
        self.num_samples_list = num_samples_list
        self.mlp_blocks = nn.ModuleList()
        for mlp_channels in mlp_channels_list:
            mlp = nn.ModuleList()
            for out_channels in mlp_channels:
                mlp.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                    )
                )
                in_channels = out_channels
            self.mlp_blocks.append(mlp)

    def forward(self, points_xyz, points_features):
        """
        @param points_xyz: (B, C, N), i.e. (batch_size, num_coordinates, num_points) - coordinates of point clouds
        @param points_features: (B, D, N) i.e. (batch_size, num_features, num_points) - features of point clouds
        @return:
        """
        points_xyz = points_xyz.permute(0, 2, 1)  # (batch_size, num_points, 3). 3=num_coordinates
        if points_features is not None:
            points_features = points_features.permute(0, 2, 1)  # (batch_size, num_points, num_features)

        B, N, C = points_xyz.shape  # batch_size, num_points, num_coordinates
        S = self.num_centroids

        # Sampling: select/sample a number of centroids with FPS
        centroid_inds = farthest_point_sample(points_xyz, self.num_centroids)

        # Get the xyz coordinate of the centroids
        centroids_xyz = index_points(points_xyz, centroid_inds)  # new_xyz

        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.num_samples_list[i]  # num_samples in a local region

            # Grouping: sample K points from a local region of each centroid
            # Each point is marked with a centroid index
            group_inds = query_ball(radius, K, points_xyz, centroids_xyz)
            grouped_xyz = index_points(points_xyz, group_inds)  # [B, num_centroids, num_samples, C]

            # Centralize the point coordinates to their associated centroids
            grouped_xyz -= centroids_xyz.view(B, S, 1, C)

            if points_features is not None:
                groupped_features = index_points(points_features, group_inds)
                grouped_points = torch.cat([groupped_features, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, C+D, K, S]

            # Get the global features for each local region
            for mlp in self.mlp_blocks:
                grouped_points = mlp(grouped_points)

            # Get the final global features
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        centroids_xyz = centroids_xyz.permute(0, 2, 1)
        # Concatenate features at different scales to form a multi-scale feature
        return centroids_xyz, torch.cat(new_points_list, dim=1)


class PointNetFeaturePropagation(nn.Module):
    """
    In PointNetSetAbstraction (PointNetSetAbstractionMSG) layer, the original point set is subsampled.
    However, in set segmentation task, we want to obtain point features for all the original points.

    This PointNetFeaturePropagation aims to propagate features from subsampled points to the original points.
    It's achieved by inverse distance weighted interpolation.
    """
    def __init__(self, in_channels, mlp_channels):
        super().__init__()

        self.mlp = nn.ModuleList()
        for out_channels in mlp_channels:
            self.mlp.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                )
            )
            in_channels = out_channels

    def forward(self, original_xyz, sampled_xyz, original_features, sampled_features):
        """
        # TODO:
        Upsample points data.
        @param original_xyz:
        @param sampled_xyz:
        @param original_features:
        @param sampled_features:
        @return:
            features
        """
        original_xyz = original_xyz.permute(0, 2, 1)
        sampled_xyz = sampled_xyz.permute(0, 2, 1)

        sampled_features = sampled_features.permute(0, 2, 1)  # ?!

        B, N, C = original_xyz.shape
        _, S, _ = sampled_xyz.shape

        if S == 1:
            interp_features = sampled_features.permute(1, N, 1)
        else:
            # interpolating features by means of inverse distance weighted average
            dists = square_distance(original_xyz, sampled_xyz)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interp_features = torch.sum(index_points(sampled_features, idx) * weight.view(B, N, 3, 1), dim=2)

        if original_features is not None:
            original_features = original_features.permute(0, 2, 1)
            new_features = torch.cat([original_features, interp_features], dim=-1)
        else:
            new_features = interp_features

        # Apply mlp to concatenated features.
        for mlp in self.mlp_blocks:
            new_features = mlp(new_features)

        return new_features


class PointNet2SSGCls(nn.Module):
    """
    PointNet++ with SSG Single-scale grouping) for classification
    """
    def __init__(self, num_classes, normal_channel=True):
        """
        Init function
        @param num_classes: number of classes
        @param normal_channel: True - input point clouds with additional normal vectors.
                               False: input only coordinates
        """
        super().__init__()
        in_channels = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        # self, num_centroids: int, radius: float, num_samples: int, in_channels: int,
        #                  mlp_channels: List[int], group_all: bool
        self.sa1 = PointNetSetAbstraction(512, 0.2, 32, in_channels, [64, 64, 128], group_all=False)
        self.sa2 = PointNetSetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256], group_all=False)
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], group_all=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, points: torch.Tensor):
        """
        @param points: [batch_size, num_points, num_channels] of point clouds.
        The first 3 channels are (x, y, z) coordinates. The later are features such as normal vector.
        @return:
        """
        B, _, _ = points.shape
        xyz = points[:, :3, :]
        if self.normal_channel and points.shape[1] > 3:
            norm = points[:, 3:, :]
        else:
            norm = None
        l1_xyz, l1_features = self.sa1(xyz, norm)
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)
        x = l3_features.view(B, 1024)
        x = self.drop1(nn.functional.relu(self.bn1(self.fc1(x))))
        x = self.drop2(nn.functional.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = nn.functional.log_softmax(x, -1)
        return x, l3_features


class PointNet2MSGCls(nn.Module):
    """
    PointNet++ with MSG (multi-scale grouping) for classification
    """
    def __init__(self, num_classes, normal_channel=True):
        super().__init__()
        in_channels = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMSG(512,
                                             [0.1, 0.2, 0.4],
                                             [16, 32, 128],
                                             in_channels,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMSG(128,
                                             [0.2, 0.4, 0.8],
                                             [32, 64, 128],
                                             320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None,
                                          640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, points):
        B, _, _ = points.shape
        if self.normal_channel:
            norm = points[:, 3:, :]
            xyz = points[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(nn.functional.relu(self.bn1(self.fc1(x))))
        x = self.drop2(nn.functional.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = nn.functional.log_softmax(x, -1)
        return x, l3_points


if __name__ == '__main__':
    from src.data.shapenet_datamodule import ShapenetCoreDataModule

    data_dir = 'data/shapenetcore_subset'
    dm = ShapenetCoreDataModule(data_dir, batch_size=4, augmentation=True, classification=False)
    test_loader = dm.test_dataloader()

    num_classes = 5
    model = PointNet2SSGCls(num_classes, normal_channel=False)

    for i, batch in enumerate(test_loader):
        if i >= 1:
            break
        print(f"Batch {i} points:", points.shape)
        print(f"Batch {i} labels:", labels.shape)
        points, labels = batch
        points = points.permute(0, 2, 1)
        out = model(points)
        print(f"Output log_softmax: {out[1].shape}")
    print("Total number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
