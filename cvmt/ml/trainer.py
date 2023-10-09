""" The trainer classes and functions mostly written in PytorchLightning API. """

import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from easydict import EasyDict
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from torchmetrics import MeanSquaredError
from torchvision import transforms

from .models import MultiTaskLandmarkUNetCustom, load_model
from .utils import (HDF5MultitaskDataset, MultitaskCollator, TransformsMapping,
                    load_loss, load_optimizer, load_scheduler, LogLearningRateToWandb,)
from collections import OrderedDict
from typing import *


class MultitaskTrainOnlyLandmarks(pl.LightningModule):
    def __init__(self, model, *args, **kwargs,):
        super().__init__(*args, **kwargs,)
        # model
        self.model = model
        
        # metrics
        self.train_mse_task_1 = MeanSquaredError_(name="train_mse_1")
        self.train_mse_task_2 = MeanSquaredError_(name="train_mse_2")
        self.val_mse_task_1 = MeanSquaredError_(name="val_mse_1")
        self.val_mse_task_2 = MeanSquaredError_(name="val_mse_2")

    def training_step(self, batch, batch_idx):
        assert isinstance(batch, dict)
        task_ids = list(batch.keys()).sort()
        # training for task 1
        task_id = task_ids[0]
        x, y = batch[task_id]
        y = y.to(torch.float32)
        preds = self.model(x, task_id=task_id)
        loss_1 = nn.CrossEntropyLoss()(preds, y)
        self.log(f'train_loss_{task_id}', loss_1)
        self.train_mse_task_1.update(preds, y)
        self.log(f'train_mse_{task_id}', self.train_mse_task_1)
        
        # training for task 2
        task_id = task_ids[0]
        x, y = batch[task_id]
        y = y.to(torch.float32)
        preds = self.model(x, task_id=task_id)
        loss_2 = nn.CrossEntropyLoss()(preds, y)
        self.log(f'train_loss_{task_id}', loss_2)
        self.train_mse_task_2.update(preds, y)
        self.log(f'train_mse_{task_id}', self.train_mse_task_2)

        # run the optimizers
        # Perform the first optimizer step
        self.optimizer.step(0)
        self.optimizer.zero_grad()

        # Perform the second optimizer step
        self.optimizer.step(1)
        self.optimizer.zero_grad()
        return loss_1, loss_2

    def configure_optimizers(self):
        optimizer1 = torch.optim.SGD(self.parameters(), lr=0.001)
        optimizer2 = torch.optim.SGD(self.parameters(), lr=0.001)
        return [optimizer1, optimizer2]
    
    def validation_step(self, batch, batch_idx):
        assert isinstance(batch, dict)
        task_ids = list(batch.keys()).sort()
        # validation for task 1
        task_id = task_ids[0]
        x, y = batch[task_id]
        y = y.to(torch.float32)
        preds = self.model(x, task_id=task_id)
        loss_1 = nn.CrossEntropyLoss()(preds, y)
        self.log(f'val_loss_{task_id}', loss_1)
        self.val_mse_task_1.update(preds, y)
        self.log(f'val_mse_{task_id}', self.val_mse_task_1)

        # validation for task 2
        task_id = task_ids[0]
        x, y = batch[task_id]
        y = y.to(torch.float32)
        preds = self.model(x, task_id=task_id)
        loss_2 = nn.CrossEntropyLoss()(preds, y)
        self.log(f'val_loss_{task_id}', loss_2)
        self.val_mse_task_2.update(preds, y)
        self.log(f'val_mse_{task_id}', self.val_mse_task_2)


class SingletaskTraining(pl.LightningModule):
    """ PL module for training a UNET model for vertebral landmark detection.
    
    See the issue below for why a pytorch lightning model cannot be loaded from checkpoint with hparams saved
    and how to use pytorch lighning module to enable using a model that is defined outside a pl module.

    https://github.com/Lightning-AI/lightning/issues/3629#issue-707536217
    """
    def __init__(
            self,
            model: nn.Module,
            task_id: int,
            optim_params: Union[EasyDict, Dict, None] = None,
            scheduler_params: Union[EasyDict, Dict, None] = None,
            loss_name: Union[str, None] = None,
            checkpoint_path: Union[str, None] = None,
        ):
        super().__init__()
        # model
        self.model = model
        # if chekpoint is supplied
        if checkpoint_path:
            self.model = self.load_from_checkpoint(checkpoint_path,).model
        # input args
        self.task_id = task_id
        self.save_hyperparameters(logger=True,)
        self.hparams.update(self.model.hparams)

        # metrics
        self.train_mse = MeanSquaredError_(name="train_mse")
        self.val_mse = MeanSquaredError_(name="val_mse")

        # set the input and outputs
        self.loss_name = loss_name
        self.optim_params = dict(optim_params) if optim_params else None
        self.scheduler_params = dict(scheduler_params) if scheduler_params else None

        self._setup()

    def _setup(self):
        if self.task_id == 1:
            self.input_key_name = self.output_key_name = 'image'
            if self.loss_name:
                self.loss = load_loss(self.loss_name)
            else:
                self.loss = nn.MSELoss
        elif self.task_id == 2:
            self.input_key_name = 'image'
            self.output_key_name = 'edges'
            if self.loss_name:
                self.loss = load_loss(self.loss_name)
            else:
                self.loss = nn.MSELoss
        elif self.task_id == 3:
            self.input_key_name = 'image'
            self.output_key_name = 'v_landmarks'
            if self.loss_name:
                self.loss = load_loss(self.loss_name)
            else:
                self.loss = nn.CrossEntropyLoss
        elif self.task_id == 4:
            self.input_key_name = 'image'
            self.output_key_name = 'f_landmarks'
            if self.loss_name:
                self.loss = load_loss(self.loss_name)
            else:
                self.loss = nn.CrossEntropyLoss
        else:
            raise ValueError("The inserted `task_id` is incorrect. Accepted values are int 1 to 4.")

    def training_step(self, batch, batch_idx):
        x, y = batch[self.input_key_name], batch[self.output_key_name]
        y = y.to(torch.float32)
        x = x.to(torch.float32)
        preds = self.model(x, task_id=self.task_id)
        loss = self.loss(preds, y)
        self.log(f'train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.train_mse.update(preds, y)
        self.log(f'train_mse', self.train_mse, on_step=True, on_epoch=True, logger=True)
        # check if task_id is either 3 or 4, then compute metric mre
        if self.task_id in [3,4]: # either v or f landmark detection
            train_mre = mean_radial_error(preds, y)
            self.log("train_mre", train_mre, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        # optimizer and StepLR scheduler
        if self.optim_params:
            optimizer = load_optimizer(**self.optim_params, model_parameters=self.parameters())
            if self.scheduler_params:
                scheduler = load_scheduler(**self.scheduler_params, optimizer=optimizer,)
                return {"optimizer": optimizer, "lr_scheduler": scheduler}
            else:
                return optimizer
        else:
            raise ValueError("`optim_params` cannot be None or empty dict!")

    def validation_step(self, batch, batch_idx):
        x, y = batch[self.input_key_name], batch[self.output_key_name]
        y = y.to(torch.float32)
        x = x.to(torch.float32)
        preds = self.model(x, task_id=self.task_id)
        loss = self.loss(preds, y)
        self.log(f'val_loss', loss, prog_bar=True, logger=True)
        self.val_mse.update(preds, y)
        self.log(f'val_mse', self.val_mse, prog_bar=True, logger=True)
        # check if task_id is either 3 or 4, then compute metric mre
        if self.task_id in [3,4]: # either v or f landmark detection
            val_mre = mean_radial_error(preds, y)
            self.log("val_mre", val_mre, prog_bar=True, logger=True)


def create_dataloader(
    task_id: int,
    params: EasyDict,
    split: str,
    batch_size: int,
    shuffle: bool,
    sampler_n_samples: Union[int, None],
) -> torch.utils.data.DataLoader:
    # load metadata
    metadata_table = pd.read_hdf(
        os.path.join(params.PRIMARY_DATA_DIRECTORY, params.TRAIN.METADATA_TABLE_NAME),
        key='df',
    )
    # create the right list of paths
    if task_id == 3:
        train_file_list = metadata_table.loc[
            (metadata_table['split']==split) & (metadata_table['valid_v_annots']==True), ['harmonized_id']
        ].to_numpy().ravel().tolist()
    if task_id == 2:
        train_file_list = metadata_table.loc[
            (metadata_table['split']==split) & (metadata_table['edges_present']==True), ['harmonized_id']
        ].to_numpy().ravel().tolist()
    if task_id == 4:
        train_file_list = metadata_table.loc[
            (metadata_table['split']==split) & (metadata_table['f_annots_present']==True), ['harmonized_id']
        ].to_numpy().ravel().tolist()
    train_file_list = [
        os.path.join(params.PRIMARY_DATA_DIRECTORY, file_path+'.hdf5') for file_path in train_file_list
    ]
    # instantiate the transforms
    transforms_mapping = TransformsMapping()
    if split == "train":
        transforms_config = OrderedDict(params.TRAIN.TRANSFORMS.TRAIN)
    elif split == "val":
        transforms_config = OrderedDict(params.TRAIN.TRANSFORMS.VAL)
    my_transforms = [transforms_mapping.get(t_name, **t_args) for t_name, t_args in transforms_config.items()]
    my_transforms = transforms.Compose(my_transforms)
    # instantiate the dataset and dataloader objects
    dataset = HDF5MultitaskDataset(
        file_paths=train_file_list,
        task_id=task_id,
        transforms=my_transforms,
    )
    if sampler_n_samples:
        assert isinstance(sampler_n_samples, int)
        sampler = RandomSampler(dataset, replacement=True, num_samples=sampler_n_samples)
        shuffle=False
    else:
        sampler = None
        shuffle = shuffle
    collator_task = MultitaskCollator(
        task_id=task_id,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator_task,
        num_workers=params.TRAIN.N_WORKERS_DATA_LOADER,
        sampler=sampler,
    )
    return dataloader


def trainer_multitask_v_and_f_landmarks(params: EasyDict):
    # initialize the model
    model_params = params.MODEL.PARAMS
    model = MultiTaskLandmarkUNetCustom(**model_params)
    # initialize dataloader and trainer objects
    train_dataloaders = {}
    val_dataloaders = {}

    for task_config in params.TRAIN.MULTIPLE_TASKS:
        # unpack parameters for easier readability
        task_id = task_config.TASK_ID
        batch_size = task_config.BATCH_SIZE
        shuffle = task_config.SHUFFLE
        # initialize dataloader
        train_dataloaders[task_id] = create_dataloader(
            task_id=task_id,
            batch_size=batch_size,
            split='train',
        )
        val_dataloaders[task_id] = create_dataloader(
            task_id=task_id,
            batch_size=batch_size,
            split='val',
        )
    # initialize trainer
    pl_model = MultitaskTrainOnlyLandmarks(
        model=model,
    )
    trainer = pl.Trainer(
        max_epochs=params.TRAIN.MAX_EPOCHS,
        log_every_n_steps=1,
    )
    # run the training
    trainer.fit(
        pl_model,
        train_dataloaders=train_dataloaders,
        val_dataloaders=val_dataloaders,
    )
    return None


def trainer_v_landmarks_single_task(params: EasyDict, checkpoint_path: Union[str, None] = None):
    # get the parameters
    task_config = params.TRAIN.V_LANDMARK_TASK
    task_id = task_config.TASK_ID
    batch_size = task_config.BATCH_SIZE
    shuffle = task_config.SHUFFLE
    devices = params.TRAIN.DEVICES
    accelerator = params.TRAIN.ACCELERATOR
    sampler_n_samples = params.TRAIN.SAMPLER_N_SAMPLES
    loss_name = params.TRAIN.LOSS_NAME
    optim_params = params.TRAIN.OPTIMIZER
    scheduler_params = params.TRAIN.SCHEDULER
    # initialize the model
    model_params = params.MODEL.PARAMS
    model = load_model(**model_params)
    # initialize dataloader and trainer objects
    # train dataloader
    train_dataloader = create_dataloader(
        task_id=task_id,
        batch_size=batch_size,
        split='train',
        shuffle=shuffle,
        params=params,
        sampler_n_samples=sampler_n_samples,
    )
    # val dataloader
    val_dataloader = create_dataloader(
        task_id=task_id,
        batch_size=batch_size,
        split='val',
        shuffle=False,
        params=params,
        sampler_n_samples=None,
    )
    # initialize trainer
    pl_model = SingletaskTraining(
        model=model,
        task_id=task_id,
        checkpoint_path=checkpoint_path,
        loss_name=loss_name,
        optim_params=optim_params,
        scheduler_params=scheduler_params if scheduler_params.scheduler_name else {}, # if name is null, empty dict
    )
    wandb_logger = WandbLogger(
        log_model='all',
        save_dir=params.WANDB.CHECKPOINTING.dir,
    )
    run_name = wandb_logger.experiment.name
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mre', 
        mode='min',
        dirpath=params.WANDB.CHECKPOINTING.dir,
        save_top_k=1,
        filename=f'{run_name}-{{epoch}}-{{step}}-{{val_loss:.3f}}-{{val_mre:.1f}}',
    )
    trainer = pl.Trainer(
        default_root_dir=params.WANDB.CHECKPOINTING.dir, 
        max_epochs=params.TRAIN.MAX_EPOCHS,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, LogLearningRateToWandb()],
        log_every_n_steps=5,
        accelerator=accelerator,
        devices=devices,
    )
    # run the training
    trainer.fit(
        pl_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    return None


def trainer_edge_detection_single_task(params: EasyDict,):
    # get the parameters
    task_config = params.TRAIN.EDGE_DETECT_TASK
    task_id = task_config.TASK_ID
    batch_size = task_config.BATCH_SIZE
    shuffle = task_config.SHUFFLE
    devices = params.TRAIN.DEVICES
    accelerator = params.TRAIN.ACCELERATOR
    sampler_n_samples = params.TRAIN.SAMPLER_N_SAMPLES
    loss_name = params.TRAIN.LOSS_NAME
    lr = params.TRAIN.LR
    # initialize the model
    model_params = params.MODEL.PARAMS
    model = load_model(**model_params)
    # initialize dataloader and trainer objects
    # train dataloader
    train_dataloader = create_dataloader(
        task_id=task_id,
        batch_size=batch_size,
        split='train',
        shuffle=shuffle,
        params=params,
        sampler_n_samples=sampler_n_samples,
    )
    # val dataloader
    val_dataloader = create_dataloader(
        task_id=task_id,
        batch_size=batch_size,
        split='val',
        shuffle=False,
        params=params,
        sampler_n_samples=None,
    )
    # initialize trainer
    pl_model = SingletaskTraining(
        model=model,
        task_id=task_id,
        loss_name=loss_name,
        lr=lr,
    )
    wandb_logger = WandbLogger(
        log_model='all',
        save_dir=params.WANDB.CHECKPOINTING.dir,
    )
    run_name = wandb_logger.experiment.name
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mse', 
        mode='min',
        dirpath=params.WANDB.CHECKPOINTING.dir,
        save_top_k=1,
        filename=f'{run_name}-{{epoch}}-{{step}}-{{val_loss:.3f}}-{{val_mse:.1f}}',
    )
    trainer = pl.Trainer(
        default_root_dir=params.WANDB.CHECKPOINTING.dir, 
        max_epochs=params.TRAIN.MAX_EPOCHS,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, LogLearningRateToWandb()],
        log_every_n_steps=5,
        accelerator=accelerator,
        devices=devices,
    )
    # run the training
    trainer.fit(
        pl_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    return None


class MeanSquaredError_(MeanSquaredError):
    def __init__(self, name):
        super().__init__()
        self.name = name


def mean_radial_error(preds: torch.Tensor, targets: torch.Tensor) -> int:
    with torch.no_grad():
        # Find the indices of the maximum value in each channel
        pred_indices = max_indices_4d_tensor(preds)
        target_indices = max_indices_4d_tensor(targets)

        # calculate the euclidean distances and then their mean
        distances = torch.sqrt(torch.sum((target_indices - pred_indices)**2, dim=2))
        mre = torch.mean(distances)
    return mre


def max_indices_4d_tensor(inp_tensor: torch.Tensor):
    # Get the maximum values along the height and width dimensions
    max_vals, max_inds = torch.max(inp_tensor.view(inp_tensor.shape[0], inp_tensor.shape[1], -1), dim=2)

    # Convert the indices into 2D coordinates
    max_inds_2d = torch.stack((max_inds // inp_tensor.shape[3], max_inds % inp_tensor.shape[3]), dim=2)

    return max_inds_2d
