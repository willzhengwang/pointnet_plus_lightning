#!/usr/bin/env python
from typing import List
import json
import os
from os import path
from collections import defaultdict
import numpy as np
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


class ShapenetCoreDataset(LightningDataModule):
    """
    ShapeNetCore is a subset of the full ShapeNet dataset with single clean 3D models and manually
    verified category and alignment annotations. It covers 55 common object categories with about 51,300
    unique 3D models.
    """
    def __init__(self, data_dir: str, dataset_type: str, classification: bool, augmentation: bool):
        """
        Initialize a dataset instance
        @param data_dir: data/shapenetcore_subset
        @param dataset_type: choices=['train', 'val', 'test']
        @param classification: True - classification. False: segmentation.
        @param augmentation: True - apply data augmentation.
        """
        super.__init__()

        self.data_dir = data_dir
        self.classification = classification
        self.augmentation = augmentation
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

        if self.augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            points[:, [0, 2]] = points[:, [0, 2]].dot(rotation_matrix)  # random rotation
            points += np.random.normal(0, 0.02, size=points.shape)  # random jitter

        # for classification
        if self.classification:
            label = np.array([self.id2cat[cat_id][1]], dtype=np.int32)
            return torch.from_numpy(points), torch.from_numpy(label)
        # for segmentation:
        seg_file = path.join(self.data_dir, cat_id, 'points_label', item + '.seg')
        label = np.loadtxt(seg_file, dtype=np.int32, delimiter=' ')
        return torch.from_numpy(points), torch.from_numpy(label)


class ShapenetCoreDataModule(LightningDataModule):
    """

    """
    def __init__(self, data_dir, batch_size=8, classification=True, augmentation=False):
        super.__init__()
        self.data_dir = data_dir
        self.classification = classification
        self.augmentation = augmentation
        self.batch_size = batch_size

        self.train_dataset = ShapenetCoreDataset(data_dir, 'train', classification, augmentation)
        self.val_dataset = ShapenetCoreDataset(data_dir, 'val', classification, False)
        self.test_dataset = ShapenetCoreDataset(data_dir, 'test', classification, False)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        data_loader = torch.utils.da



        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn)
        return train_loader
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass


if __name__ == '__main__':
    data_dir = 'data/shapenetcore_subset'
    train_dataset = ShapenetCoreDataset(data_dir, 'train', augmentation=True)
    n_files = len(train_dataset)
    print(f"Number of files: {n_files}")

    idx = np.random.randint(0, n_files-1)
    print(f"The {idx}-th sample is picked.")
    points, label = train_dataset[idx]
    print(points.shape)
    print(label)
    print('Done')
