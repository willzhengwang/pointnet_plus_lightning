#!/usr/bin/env python
from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from lightning import LightningModule


class TNet(nn.Module):
    """
    T-Net for transformation in the PointNet paper.
    """

    def __init__(self, k: int):
        """
        @param k: input dimension.
        """
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(
            nn.Conv1d(k, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        # max(x, 2) + view(-1, 1024)
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(256, k*k)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = torch.max(x, 2, keepdim=True)[0]  # (batch_size, 1024, num_pts)
        x = x.view(-1, 1024)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)  # (B, k*k)
        x = x.view(-1, self.k, self.k)  # (B, k, k)

        identity = torch.eye(self.k).view(1, self.k, self.k).repeat(batch_size, 1, 1)
        if x.cuda():
            identity = identity.cuda()
        return x + identity


class FeatureNet(nn.Module):
    """
    Extract local and global features
    """
    def __init__(self, classification=True, feature_transform=False):
        """
        @param classification: True - for classification. Extract global feature only.
                               False - for segmentation. Cat([global_feature, local_feature]
        @param feature_transform:
        """
        super().__init__()
        self.classification = classification
        self.feature_transform = feature_transform
        self.tnet3 = TNet(3)
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        if feature_transform:
            self.tnet64 = TNet(64)
        self.mlp = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (batch_size, 3, num_pts)
        trans3 = self.tnet3(x)  # (batch_size, 3, 3)
        x = torch.transpose(x, 1, 2)  # (batch_size, num_pts, 3)
        x = torch.bmm(x, trans3)  # (batch_size, num_pts, 3)
        x = torch.transpose(x, 1, 2)  # (batch_size, 3, num_pts)
        x = self.conv1(x)  # (batch_size, 64, num_pts)
        local_feat = x

        if self.feature_transform:
            trans64 = self.tnet64(x)
            x = torch.transpose(x, 1, 2)
            x = torch.bmm(x, trans64)  # (batch_size, num_pts, 64)
            x = torch.transpose(x, 1, 2)  # (batch_size, 64, num_pts)
        else:
            trans64 = None

        x = self.mlp(x)  # (batch_size, 1024, num_pts)
        x = torch.max(x, 2, keepdim=True)[0]  # (batch_size, 1024, 1)
        global_feat = x.view(-1, 1024)  # (batch_size, 1024)

        if self.classification:
            return global_feat, trans3, trans64
        # segmentation
        return torch.cat([local_feat, global_feat], dim=1), trans3, trans64


def regularize_feat_transform(feat_trans):
    """
    Regularization loss over the feature transformation matrix
    @param feat_trans: (batch_size, 64, 64)
    @return:
    """
    k = feat_trans.shape[-1]
    I = torch.eye(k)
    if feat_trans.cuda():
        I = I.cuda()
    tmp = (torch.bmm(feat_trans, torch.transpose(feat_trans, 1, 2)) - I)
    reg_loss = torch.mean(torch.norm(tmp, dim=(1, 2)))
    return reg_loss


class PointNetCls(nn.Module):
    """
    PointNet for classification
    """
    def __init__(self, num_classes, feature_transform=False):
        super().__init__()

        self.feat_net = FeatureNet(classification=True, feature_transform=feature_transform)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256, kernel_size=1, bias=False),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, num_classes, kernel_size=1),
        )

    def forward(self, x):
        x, tran3, trans64 = self.feat_net(x)
        x = self.mlp(x)  # (batch_size, num_classes)
        return nn.functional.log_softmax(x), tran3, trans64


class PointNetClsModule(LightningModule):
    """
    PontNet Classifier - Lightning Module
    """
    def __init__(self,
                 net: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 compling: bool = False
                 ):
        """
        Init function of the LightningModule
        @param net: The model to train.
        @param optimizer: The optimizer to use for training.
        @param scheduler: The learning rate scheduler to use for training.
        @param compling: True - compile model for faster training with pytorch 2.0.
        """
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        self.loss = nn.functional.nll_loss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        points, labels = batch
        points = torch.transpose(points, 1, 2)  # (batch_size, 3, num_point)

        preds, tran3, trans64 = self.forward(points)  # preds are log_softmax
        total_loss = self.loss(preds, labels)
        if trans64 is not None:
            total_loss += regularize_feat_transform(trans64)
