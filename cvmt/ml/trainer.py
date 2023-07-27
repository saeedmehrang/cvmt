""" The trainer classes and functions mostly written in PytorchLightning API. """

import os

import pandas as pd
import pytorch_lightning as pl
import torch
from easydict import EasyDict
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError
from torchvision import transforms

from .models import MultiTaskLandmarkUNetCustom
from .utils import (Coord2HeatmapTransform, CustomToTensor,
                    HDF5MultitaskDataset, MultitaskCollator, ResizeTransform)


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


class SingletaskTrainLandmarks(pl.LightningModule):
    def __init__(self, model, *args, **kwargs,):
        super().__init__(*args, **kwargs,)
        # model
        self.model = model    

        # metrics
        self.train_mse = MeanSquaredError_(name="train_mse")
        self.val_mse = MeanSquaredError_(name="val_mse")

    def training_step(self, batch, batch_idx):
        #assert isinstance(batch, dict)
        #task_ids = list(batch.keys()).sort()
        # training for task
        #task_id = task_ids[0]
        task_id = 3
        x, y = batch['image'], batch['v_landmarks']
        y = y.to(torch.float32)
        x = x.to(torch.float32)
        preds = self.model(x, task_id=task_id)
        loss = nn.CrossEntropyLoss()(preds, y)
        self.log(f'train_loss_{task_id}', loss)
        self.train_mse.update(preds, y)
        self.log(f'train_mse_{task_id}', self.train_mse)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def validation_step(self, batch, batch_idx):
        #assert isinstance(batch, dict)
        #task_ids = list(batch.keys()).sort()
        # validation for task
        #task_id = task_ids[0]
        task_id = 3
        x, y = batch['image'], batch['v_landmarks']
        y = y.to(torch.float32)
        x = x.to(torch.float32)
        preds = self.model(x, task_id=task_id)
        loss = nn.CrossEntropyLoss()(preds, y)
        self.log(f'val_loss_{task_id}', loss)
        self.val_mse.update(preds, y)
        self.log(f'val_mse_{task_id}', self.val_mse)


def create_dataloader(
    task_id: int,
    params: EasyDict,
    split: str,
    batch_size: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    # load metadata
    metadata_table = pd.read_hdf(
        os.path.join(params.PRIMARY_DATA_DIRECTORY, params.TRAIN.METADATA_TABLE_NAME),
        key='df',
    )
    # create the right list of paths
    train_file_list = metadata_table.loc[
        (metadata_table['split']==split) & (metadata_table['v_annots_present']==True), ['harmonized_id']
    ].to_numpy().ravel().tolist()
    train_file_list = [
        os.path.join(params.PRIMARY_DATA_DIRECTORY, file_path+'.hdf5') for file_path in train_file_list
    ]
    # instantiate the transforms
    my_transforms = transforms.Compose([
        ResizeTransform(tuple(params.TRAIN.TARGET_IMAGE_SIZE)),
        Coord2HeatmapTransform(
            tuple(params.TRAIN.TARGET_IMAGE_SIZE),
            params.TRAIN.GAUSSIAN_COORD2HEATMAP_STD,
        ),
        CustomToTensor(),
    ])
    # instantiate the dataset and dataloader objects
    train_dataset = HDF5MultitaskDataset(
        file_paths=train_file_list,
        task_id=task_id,
        transforms=my_transforms,
    )
    collator_task = MultitaskCollator(
        task_id=task_id,
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator_task,
        num_workers=1,
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


def trainer_v_landmarks_single_task(params: EasyDict):
    # get the parameters
    task_config = params.TRAIN.SINGLE_TASK
    task_id = task_config.TASK_ID
    batch_size = task_config.BATCH_SIZE
    shuffle = task_config.SHUFFLE
    # initialize the model
    model_params = params.MODEL.PARAMS
    model = MultiTaskLandmarkUNetCustom(**model_params)
    # initialize dataloader and trainer objects
    # train dataloader
    train_dataloader = create_dataloader(
        task_id=task_id,
        batch_size=batch_size,
        split='train',
        shuffle=shuffle,
        params=params,
    )
    # val dataloader
    val_dataloader = create_dataloader(
        task_id=task_id,
        batch_size=batch_size,
        split='val',
        shuffle=shuffle,
        params=params,
    )
    # initialize trainer
    pl_model = SingletaskTrainLandmarks(
        model=model,
    )
    trainer = pl.Trainer(        
        max_epochs=params.TRAIN.MAX_EPOCHS,
        log_every_n_steps=1,
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
