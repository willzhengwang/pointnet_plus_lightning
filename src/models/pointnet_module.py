#!/usr/bin/env python
import torch
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics import MeanMetric, MaxMetric
from lightning import LightningModule


class TNet(nn.Module):
    """
    T-Net for (input / feature) transformation in the PointNet paper.
    Structure: mlp --> maxpool --> mlp.
    TNet is data-dependent , so it's size is (batch_size, dim, dim).
    TNet provides a new viewpoint of a pcd of an object.
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
            nn.BatchNorm1d(512),
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
        if x.is_cuda:
            identity = identity.cuda()
        return x + identity


class FeatureNet(nn.Module):
    """
    Extract point embeddings and global features.
    Structure: input TNet --> optional feature TNet --> MLP --> max pool
    """
    def __init__(self, classification=True, feature_transform=False):
        """
        @param classification: True - for classification. Extract global feature only.
                               False - for segmentation. Cat([point_embedding, global_feature]
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
        point_feat = x

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
        return torch.cat([point_feat, global_feat], dim=1), trans3, trans64


def regularize_feat_transform(feat_trans):
    """
    Regularization loss over the feature transformation matrix
    @param feat_trans: (batch_size, 64, 64)
    @return:
    """
    k = feat_trans.shape[-1]
    I = torch.eye(k)
    if feat_trans.is_cuda:
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
        self.num_classes = num_classes

        self.feat_net = FeatureNet(classification=True, feature_transform=feature_transform)
        self.mlp = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 256, bias=False),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x, tran3, trans64 = self.feat_net(x)
        logits = self.mlp(x)  # (batch_size, num_classes)
        return logits, tran3, trans64


class PointNetClsModule(LightningModule):
    """
    PontNet Classifier - Lightning Module
    """
    def __init__(self,
                 net: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler,
                 compile: bool = False
                 ):
        """
        Init function of the LightningModule
        @param net: The model to train.
        @param optimizer: The optimizer to use for training.
        @param scheduler: The learning rate scheduler to use for training.
        @param compile: True - compile model for faster training with pytorch 2.0.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        self.criterion = nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=net.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=net.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=net.num_classes)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        @param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step on a batch of data from the training set.

        @param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        @param batch_idx: The index of the current batch.
        @return: A tensor of losses between model predictions and targets.
        """
        points, labels = batch
        points = torch.transpose(points, 1, 2)  # (batch_size, 3, num_point)

        logits, tran3, trans64 = self.forward(points)  # preds are logits

        loss = self.criterion(logits, labels)
        if trans64 is not None:
            loss += regularize_feat_transform(trans64)

        preds = torch.argmax(logits, dim=1)  # (batch_size, num_classes)
        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, labels)

        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step on a batch of data from the validation set.
        @param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        @param batch_idx: The index of the current batch.
        """
        points, labels = batch
        points = torch.transpose(points, 1, 2)  # (batch_size, 3, num_point)

        logits, tran3, trans64 = self.forward(points)  # preds are logits
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)  # (batch_size, num_classes)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, labels)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Perform a single test step on a batch of data from the test set.

        @param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        @param batch_idx: The index of the current batch.
        """
        points, labels = batch
        points = torch.transpose(points, 1, 2)  # (batch_size, 3, num_point)

        logits, _, _ = self.forward(points)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)  # (batch_size, num_classes)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, labels)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        @return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == '__main__':
    batch_size, num_classes = 4, 5
    sim_data = torch.rand(batch_size, 3, 2000)

    pointfeat = PointNetCls(num_classes)
    out, _, _ = pointfeat(sim_data)
    print(f"Expect out shape: {batch_size} * {num_classes}")
    print('Point feat', out.shape)
    print("Total number of parameters:", sum(p.numel() for p in pointfeat.parameters() if p.requires_grad))
