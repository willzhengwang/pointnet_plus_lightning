#!/usr/bin/env python
import torch
import numpy as np
import pytest
from src.models.pointnet2_module import (pc_normalize, square_distance, index_points, farthest_point_sample,
                                         query_ball, PointNet2MSGCls)


def test_pc_normalize():
    np.random.seed(42)
    pc = np.random.randn(50, 3)
    pc[:, 0] * 2 + 3.0
    pc[:, 1] * (-2) - 1.0

    normalized_pc = pc_normalize(pc)
    radius = np.max(np.linalg.norm(normalized_pc, axis=1))
    assert radius == pytest.approx(1.0, abs=1e-8)


def test_square_distance():
    # case 1: test the output size
    src = torch.randn([8, 50, 3])
    dst = torch.randn([8, 60, 3])
    dist = square_distance(src, dst)
    assert dist.shape == torch.Size([8, 50, 60])

    # case 2: check the results
    src = torch.ones([8, 10, 3])
    src[:, :, 1] = -1
    dst = torch.zeros([8, 15, 3])
    dist = square_distance(src, dst)
    assert torch.all(torch.eq(dist, 3.0))


def test_index_points():
    batch_size, num_pts = 4, 10
    points = torch.zeros([batch_size, num_pts, 3])
    for i in range(num_pts):
        points[:, i, :] = i
    inds = torch.tensor([[0, 1], [0, 5], [6, 7], [9, 8]], dtype=torch.long)
    indexed_points = index_points(points, inds)
    for i in range(batch_size):
        expect = inds[i].view(len(inds[i]), -1).repeat(1, 3).float()
        assert torch.eq(indexed_points[i], expect).all()


def test_fps_and_ball_query():
    torch.manual_seed(0)
    batch_size, num_centroids, num_channels = 2, 3, 3
    batch_xyz = torch.randn([batch_size, 100, num_channels], dtype=torch.float32) * 0.05
    batch_xyz[:, 50:, :] += 10.0
    centroid_inds = farthest_point_sample(batch_xyz, num_centroids)

    centroids = torch.gather(batch_xyz, 1, centroid_inds.unsqueeze(-1).repeat([1, 1, num_channels]))
    squared_dis = torch.norm(centroids[0, 0, :] - centroids[0, 1, :]) ** 2
    assert pytest.approx(squared_dis.item(), abs=20) == 300.0

    group_inds = query_ball(0.1, 10, batch_xyz, centroids)
    assert group_inds.shape == torch.Size([batch_size, num_centroids, 10])


def test_pointnet2_msg_cls():
    batch_size, num_classes = 2, 5
    model = PointNet2MSGCls(num_classes, normal_channel=False)
    points = torch.randn([batch_size, 3, 1000])  # [batch_size, num_channels, num_points]
    out = model(points)
    assert out[0].shape == torch.Size([batch_size, num_classes])
    assert out[1].shape == torch.Size([batch_size, 1024, 1])
