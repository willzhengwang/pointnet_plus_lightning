#!/usr/bin/env python
from typing import List
import json
import os
from os import path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


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


class ShapenetCoreDataset(Dataset):
    """
    ShapeNetCore dataset - coordinates with normals.
    ShapeNetCore is a subset of the full ShapeNet dataset with single clean 3D models and manually
    verified category and alignment annotations. It covers 55 common object categories with about 51,300
    unique 3D models.

    Dataset structure:
    <data_dir> contains a number of sub folders. Each sub folder is named by <category_id> such as 02691156 and contains
    a list of txt file.
    Each text file contains N * 7 float array. In each row,  (x, y, z, nx, ny, nz, segment_label), like the following
    0.090740 -0.131890 -0.081920 -0.116500 -0.033230 -0.992600 1.000000
    """
    def __init__(self, data_dir: str, dataset_type: str, classification: bool, augmentation: bool,
                 num_points: int = 2500):
        """
        Initialize a dataset instance
        @param data_dir: data/shapenetcore_normals
        @param dataset_type: choices=['train', 'val', 'test']
        @param classification: True for classification; False for segmentation.
        @param augmentation: True - apply data augmentation.
        @param num_points: The number of points in each point cloud is different, a batch requires each sample
        (point cloud) with the same length. Therefore, we need to resample each point cloud to the same length
        """
        super().__init__()

        self.data_dir = data_dir
        self.augmentation = augmentation
        self.classification = classification
        self.num_points = num_points
        self.cat2id, self.id2cat = self.get_categories()

        self.json_file = os.path.join(data_dir, 'train_test_split', 'shuffled_{}_file_list.json'.format(dataset_type))
        with open(self.json_file) as f:
            self.file_list = json.load(f)

    def get_files(self, category: str) -> List:
        """
        Get files of a certain category
        @param category:  name
        @return:
        """
        cat_id = self.cat2id[category]
        cat_files = []
        for entry in self.file_list:
            _, str_id, item = entry.split('/')
            if str_id == cat_id:
                cat_files.append(entry)
        return cat_files

    def get_categories(self):
        cat2id = {}  # {name: [str_id, category_index]}
        id2cat = {}  # {str_id: [name, category_index]}
        cat_idx = 0
        with open(path.join(self.data_dir, "synsetoffset2category.txt"), "r") as f:
            lines = f.readlines()
            for line in lines:
                if line:
                    name, str_id = line.strip().split("\t")
                    if name not in cat2id and str_id not in id2cat:
                        cat2id[name] = [str_id, cat_idx]
                        id2cat[str_id] = [name, cat_idx]
                        cat_idx += 1
        return cat2id, id2cat

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        """
        Return the input x and label/target for the i-th sample
        @param idx: index
        @return:
            torch.Tensor points: n_pts * 3
            torch.Tensor label: when classification=True: a single class label
                                when classification=False: n_pts * 1 of segmentation labels
        """
        _, cat_id, item = self.file_list[idx].split('/')
        pts_file = path.join(self.data_dir, cat_id, item + '.txt')
        points = np.loadtxt(pts_file, dtype=np.float32, delimiter=' ')
        points[:, 0:3] = pc_normalize(points[:, 0:3])

        if self.classification:
            cls_label = np.array([self.id2cat[cat_id][1]], dtype=np.int64)
        else:
            seg_labels = np.array(points[:, -1], dtype=np.int64)

        if points.shape[0] >= self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[choice, :]
        else:  # if points.shape[0] < self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points)
            points = points[choice, :]

        points = np.array(points[:, :6], dtype=np.float32)
        if self.augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            points[:, [0, 2]] = points[:, [0, 2]].dot(rotation_matrix)  # random rotation of (x, y, z)
            points[:, [3, 5]] = points[:, [3, 5]].dot(rotation_matrix)  # random rotation of (nx, ny, nz)

        # for classification
        if self.classification:
            return torch.from_numpy(points), torch.from_numpy(cls_label)
        # for segmentation
        return torch.from_numpy(points), torch.from_numpy(seg_labels[choice].squeeze())


class ShapenetCoreDataModule(LightningDataModule):
    """
    ShapenetCore Data module
    """
    def __init__(self, data_dir, batch_size=8, classification=True, augmentation=False,
                 num_workers=4, num_points=2500):
        """
        Initialize a ShapenetCoreDataModule instance

        @param data_dir: data folder path
        @param batch_size: batch size
        @param classification: True for classification; False for segmentation.
        @param augmentation: data augmentation, such as rotation.
        @param num_workers: number of workers on the data loading.
        @param num_points: The number of points in each point cloud is different, a batch requires each sample
        (point cloud) with the same length. Therefore, we need to resample each point cloud to the same length
        """
        super().__init__()
        self.data_dir = data_dir
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = ShapenetCoreDataset(data_dir, 'train',
                                                 classification, augmentation, num_points=num_points)
        self.val_dataset = ShapenetCoreDataset(data_dir, 'val',
                                               classification, False, num_points=num_points)
        self.test_dataset = ShapenetCoreDataset(data_dir, 'test',
                                                classification, False, num_points=num_points)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True)
        return data_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        data_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=False,
            persistent_workers=True)
        return data_loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        data_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=False,
            persistent_workers=True
        )
        return data_loader


if __name__ == '__main__':
    data_dir = 'data/shapenetcore_normals'
    dm = ShapenetCoreDataModule(data_dir, batch_size=2, classification=True, augmentation=True)
    test_loader = dm.test_dataloader()
    for i, batch in enumerate(test_loader):
        if i >= 1:
            break
        points, labels = batch
        print(f"Batch {i} points:", points.shape)
        print(f"Batch {i} labels:", labels.shape)

    print('Done')
