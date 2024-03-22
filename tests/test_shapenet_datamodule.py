#!/usr/bin/env python
import numpy as np
import pytest
import torch

from src.data.shapenet_datamodule import pc_normalize, ShapenetCoreDataModule


def test_pc_normalize():
    np.random.seed(42)
    pc = np.random.randn(50, 3)
    pc[:, 0] * 2 + 3.0
    pc[:, 1] * (-2) - 1.0

    normalized_pc = pc_normalize(pc)
    radius = np.max(np.linalg.norm(normalized_pc, axis=1))
    assert radius == pytest.approx(1.0, abs=1e-8)


def test_shapenet_datamodule():
    data_dir = 'data/shapenetcore_normals'
    dm = ShapenetCoreDataModule(data_dir, batch_size=2, classification=False, augmentation=True)
    test_loader = dm.test_dataloader()
    for i, batch in enumerate(test_loader):
        if i >= 1:
            break
        points, labels = batch
        assert points.shape == torch.Size([2, 2500, 6])
        assert labels.shape == torch.Size([2, 2500])
