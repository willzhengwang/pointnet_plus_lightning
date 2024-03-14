#!/usr/bin/env python
from typing import List
import json
import os
from os import path
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


class ShapenetCoreDataset(LightningDataModule):
    """
    ShapeNetCore is a subset of the full ShapeNet dataset with single clean 3D models and manually
    verified category and alignment annotations. It covers 55 common object categories with about 51,300
    unique 3D models.
    """
    def __init__(self, data_dir: str, dataset_type: str, classification: bool, augmentation: bool,
                 num_points: int = 2500):
        """
        Initialize a dataset instance
        @param data_dir: data/shapenetcore_subset
        @param dataset_type: choices=['train', 'val', 'test']
        @param classification: True - classification. False: segmentation.
        @param augmentation: True - apply data augmentation.
        @param num_points: The number of points in each point cloud is different, a batch requires each sample
        (point cloud) with the same length. Therefore, we need to resample each point cloud to the same length
        """
        super().__init__()

        self.data_dir = data_dir
        self.classification = classification
        self.augmentation = augmentation
        self.num_points = num_points
        self.cat2id, self.id2cat = self.get_categories()

        self.json_file = os.path.join(data_dir, 'train_test_split', 'shuffled_{}_file_list.json'.format(dataset_type))
        with open(self.json_file) as f:
            self.file_list = json.load(f)

    def get_files(self, category: str) -> List:
        """
        Get files of a certain category
        @param category: category name
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

    def __getitem__(self, i: int):
        """
        Return the input x and label/target for the i-th sample
        @param i: index
        @return:
            torch.tensor points: n_pts * 3
            torch.tensor label: when classification=True: a single class label
                                when classification=False: n_pts * 1 of segmentation labels
        """
        _, cat_id, item = self.file_list[i].split('/')
        pts_file = path.join(self.data_dir, cat_id, 'points', item + '.pts')
        points = np.loadtxt(pts_file, dtype=np.float32, delimiter=' ')

        if points.shape[0] > self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[choice, :]
        elif points.shape[0] < self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points)
            points = points[choice, :]

        if self.augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            points[:, [0, 2]] = points[:, [0, 2]].dot(rotation_matrix)  # random rotation
            points += np.random.normal(0, 0.02, size=points.shape)  # random jitter

        # for classification
        if self.classification:
            label = np.array([self.id2cat[cat_id][1]], dtype=np.int64)
            return torch.from_numpy(points), torch.from_numpy(label.squeeze())
        # for segmentation:
        seg_file = path.join(self.data_dir, cat_id, 'points_label', item + '.seg')
        label = np.loadtxt(seg_file, dtype=np.int64, delimiter=' ')
        label = label[choice]
        return torch.from_numpy(points), torch.from_numpy(label.squeeze())


class ShapenetCoreDataModule(LightningDataModule):
    """
    ShapenetCore Data module
    """
    def __init__(self, data_dir, batch_size=8, classification=True, augmentation=False, num_workers=4, num_points=2500):
        super().__init__()
        self.data_dir = data_dir
        self.classification = classification
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
            pin_memory=False)
        return data_loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        data_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=False)
        return data_loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        data_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            pin_memory=False)
        return data_loader


def show_pcd(points, labels):
    """
    Visualize a point cloud sample
    @param points:
    @param labels:
    @return:
    """
    import open3d as o3d

    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Map label values to colors
    num_labels = int(np.max(labels))
    label_colors = np.random.rand(num_labels + 1, 3)  # Generate random colors for each label
    colors = label_colors[labels.astype(int)]  # Map labels to colors

    # Assign colors to the point cloud based on the labels
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    data_dir = 'data/shapenetcore_subset'
    dm = ShapenetCoreDataModule(data_dir, batch_size=2, augmentation=True, classification=False)
    train_loader = dm.train_dataloader()

    for i, batch in enumerate(train_loader):
        if i > 2:
            break
        points, labels = batch
        for j in range(points.shape[0]):
            show_pcd(points[j].numpy(), labels[j].numpy())
        print(f"Batch {i} points:", points.shape)
        print(f"Batch {i} labels:", labels.shape)

    print('Done')
