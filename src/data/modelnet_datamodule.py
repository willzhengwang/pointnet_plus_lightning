#!/usr/bin/env python
import os
from os import path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
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


class ModelnetDataset(Dataset):
    """
    Modelnet dataset (for the classification purpose) - coordinates with normals.
    The download url: https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip

    Dataset structure:
    In the <data_dir>,
        filelist.txt: all point cloud files in <data_dir> and its sub-folders.
        modelnet_id.txt: class name - index.
        modelnet<N>_shape_names.txt: contains N (10 or 40) shape class names.
        modelnet<N>_test.txt: testing samples.
        modelnet<N>_train.txt: training samples.
        <shape_name> folders, such as airplane, bathtub, etc. Each shape folder contains a list of txt files.
        A "class_idx.txt" file has 10000 lines and each line is (x, y, z, nx, ny, nz).
    """
    def __init__(self, data_dir: str, dataset_type: str, num_classes: int = 40, augmentation: bool = True,
                 num_points: int = 10000):
        """
        Initialize a dataset instance
        @param data_dir: data/modelnet40_normal_resampled
        @param dataset_type: choices=['train', 'test']
        @param num_classes: number of classes
        @param augmentation: True - apply data augmentation.
        @param num_points: if the number of points is not num_points, resample it to the length num_points.
        """
        super().__init__()

        self.data_dir = data_dir
        self.num_classes = num_classes
        self.augmentation = augmentation
        self.num_points = num_points
        self.cat2id, self.id2cat = self.get_categories()

        self.txt_file = os.path.join(data_dir, 'modelnet{}_{}.txt'.format(num_classes, dataset_type))
        with open(self.txt_file, "r") as f:
            self.sample_files = []  # file_path, cls_idx
            for line in f.readlines():
                if line:
                    cls_name = line.strip()[:-5]
                    self.sample_files.append((path.join(data_dir, cls_name, line.strip() + '.txt'),
                                              self.cat2id[cls_name]))

    def get_categories(self):
        cat2id = {}  # {name: idx}
        id2cat = {}  # {idx: name}
        with open(path.join(self.data_dir, f"modelnet{self.num_classes}_shape_names.txt"), "r") as f:
            lines = f.readlines()
            cat_idx = 0
            for line in lines:
                if line:
                    cls_name = line.strip()
                    if cls_name not in cat2id and cat_idx not in id2cat:
                        cat2id[cls_name] = cat_idx
                        id2cat[cat_idx] = cls_name
                        cat_idx += 1
        return cat2id, id2cat

    def __len__(self):
        """ return the number of samples in the dataset """
        return len(self.sample_files)

    def __getitem__(self, idx: int):
        """
        Return the input x and label/target for the i-th sample
        @param idx: index
        @return:
            torch.Tensor points: n_pts * 6
            torch.Tensor label: when classification=True: a single class label
        """
        pts_file, cls_label = self.sample_files[idx]
        points = np.loadtxt(pts_file, dtype=np.float32, delimiter=',')
        points[:, 0:3] = pc_normalize(points[:, 0:3])

        # just in case the number of points is not equal to self.num_points
        if points.shape[0] > self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[choice, :]
        elif points.shape[0] < self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points)
            points = points[choice, :]

        if self.augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            points[:, [0, 2]] = points[:, [0, 2]].dot(rotation_matrix)  # random rotation of (x, y, z)
            points[:, [3, 5]] = points[:, [3, 5]].dot(rotation_matrix)  # random rotation of (nx, ny, nz)

        return torch.from_numpy(points), torch.from_numpy(np.array([cls_label]).squeeze().astype(np.int64))


class ModelnetDataModule(LightningDataModule):
    """
    Modelnet Data module
    """
    def __init__(self, data_dir, batch_size=16, num_classes: int = 40,
                 augmentation=True, num_workers=8, num_points=10000):
        """
        Initialize a ShapenetCoreDataModule instance

        @param data_dir: data folder path
        @param batch_size: batch size
        @param num_classes: number of classes
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

        dataset = ModelnetDataset(data_dir, 'train', num_classes=num_classes,
                                  augmentation=augmentation, num_points=num_points)
        train_size = int(0.9 * len(dataset))
        torch.manual_seed(42)  # ensure the split consistency
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

        self.test_dataset = ModelnetDataset(data_dir, 'test', num_classes=num_classes,
                                            augmentation=augmentation, num_points=num_points)

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

    data_dir = 'data/modelnet40_normal_resampled'
    dm = ModelnetDataModule(data_dir, batch_size=2, num_classes=40)
    test_loader = dm.test_dataloader()
    for i, batch in enumerate(test_loader):
        if i >= 1:
            break
        points, labels = batch
        print(f"Batch {i} points:", points.shape)
        print(f"Batch {i} labels:", labels.shape)
    print('Done')
