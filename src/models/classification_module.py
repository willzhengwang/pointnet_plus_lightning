#!/usr/bin/env python
"""
Pytorch Lightning implementation of Classification on point clouds
"""
import torch
from torch import nn
from lightning import LightningModule
from torchmetrics.classification import Accuracy
from torchmetrics.aggregation import MeanMetric, MaxMetric


class PointsClsModule(LightningModule):
    """
    Classification on Point Clouds - Lightning Module
    """
    def __init__(self, net: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler,
                 compile: bool = False):
        """
        Init function of the LightningModule
        @param net: The model to train. Either PointNet2MSGCls or PointNet2SSGCls.
        @param optimizer: The optimizer to use for training.
        @param scheduler: The learning rate scheduler to use for training.
        @param compile: True - compile model for faster training with pytorch 2.0.
        """
        super().__init__()

        # This line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt.
        # logger=True: send the hyperparameters to the logger, which results in a large log file.
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
        """
        Lightning hook that is called at the beginning of fit (train + validate),
        validate, test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        @param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def on_train_start(self) -> None:
        """
        Lightning hook that is called when training begins
        """
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    def forward(self, x):
        return self.net(x)

    def model_step(self, batch):
        """
        Forward + Loss + Predict a batch of data
        """
        points, labels = batch
        points = points.permute(0, 2, 1)  # (batch_size, 3+num_features, num_point)

        logits, _ = self.forward(points)  # logits: batch_size * num_classes
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

    def training_step(self, batch: tuple, batch_idx: int):
        """
        Perform a single training step on a batch of data from the training set.

        @param batch: A batch of data (a tuple) containing the input tensor of images and target labels.
        @param batch_idx: The index of the current batch.
        @return: A tensor of losses between model predictions and targets.
        """
        loss, preds, labels = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, labels)

        self.log("train/loss", self.train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step on a batch of data from the validation set
        """
        loss, preds, labels = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, labels)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """
        Perform a single test step on a batch of data from the test set.
        """
        loss, preds, labels = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, labels)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        @return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val/loss',
                'interval': 'epoch',
                'frequency': 1,
            }
        return {'optimizer': optimizer}
