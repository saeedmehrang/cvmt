import os
import random
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple, Union

import cv2
import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import yaml
from easydict import EasyDict
from numpy import unravel_index
from pytorch_lightning.loggers import TensorBoardLogger
from segmentation_models_pytorch.base.modules import Activation
from segmentation_models_pytorch.encoders import get_encoder
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MeanSquaredError
from torchvision import transforms

SUPPORTED_ARCHS = {"Unet", "FPN"}


def nested_dict_to_easydict(nested_dict):
    if isinstance(nested_dict, dict):
        for key, value in nested_dict.items():
            nested_dict[key] = nested_dict_to_easydict(value)
        return EasyDict(nested_dict)
    elif isinstance(nested_dict, list):
        return [nested_dict_to_easydict(item) for item in nested_dict]
    else:
        return nested_dict


with open("../../code_configs/params.yaml") as f:
    PARAMS = yaml.safe_load(f)
    PARAMS = nested_dict_to_easydict(PARAMS)


class HDF5MultitaskDataset(Dataset):
    def __init__(
        self,
        file_paths: List[str],
        transforms: Callable = None,
        landmark_shapes: Dict = {'v_landmarks': (13,2), 'f_landmarks': (19,2)},
        task_id: int = 1,
    ):
        self.file_paths = file_paths
        self.transforms = transforms
        self.landmark_shapes = landmark_shapes
        self.task_id = task_id      

    def __len__(self):
        return len(self.file_paths)

    def _task_one(
        self,
        index,
    ) -> Dict[str, Any]:
        # read the hdf5 file
        f = h5py.File(self.file_paths[index], "r")
        # read image data
        image = f['image']['data'][:]
        # task one - input: image, output: image! this is input image reconstruction
        data_dict = {'image': image}
        return data_dict

    def _task_two(
        self,
        index,
    ) -> Dict[str, Any]:
        # read the hdf5 file
        f = h5py.File(self.file_paths[index], "r")
        # read image data
        image = f['image']['data'][:]
        # task one - input: image, output: image edges
        # we avoid duplicating the data here in the interest of RAM consumption.
        data_dict = {'image': image, 'edges': None}
        if 'edges' in f.keys():
            edges = f['edges']['data'][:]
            if edges.shape == image.shape:
                data_dict['edges'] = edges
        return data_dict

    def _task_three(
        self,
        index,
    ) -> Dict[str, Any]:
        # read the hdf5 file
        f = h5py.File(self.file_paths[index], "r")
        # read image data
        image = f['image']['data'][:]
        # task three - input: image, output: vertebral landmarks
        data_dict = {'image': image, 'v_landmarks': None}
        if 'v_landmarks' in f.keys():
            v_landmarks = []
            vertebrate_ids = list(f['v_landmarks']['shapes'].keys())
            for vertebrate_id in vertebrate_ids:
                landmarks = f['v_landmarks']['shapes'][vertebrate_id]['points'][:]
                v_landmarks.append(landmarks)
            # turn it into numpy array
            v_landmarks = np.concatenate(v_landmarks)
            if self.landmark_shapes['v_landmarks'] == v_landmarks.shape:
                data_dict['v_landmarks'] = v_landmarks
        f.close()
        return data_dict
    
    def _task_four(
        self,
        index,
    ) -> Dict[str, Any]:
        # read the hdf5 file
        f = h5py.File(self.file_paths[index], "r")
        # read image data
        image = f['image']['data'][:]
        # task four - input: image, output: facial landmarks
        data_dict = {'image': image, 'f_landmarks': None}
        if 'f_landmarks' in f.keys():
            f_landmarks = f['f_landmarks']['points'][:]
            if self.landmark_shapes['f_landmarks'] == f_landmarks.shape:
                data_dict['f_landmarks'] = f_landmarks
        f.close()
        return data_dict

    def __getitem__(self, index) -> Dict[str, Any]:
        # task one: input image reconstruction
        if self.task_id == 1:
            data_dict = self._task_one(index=index)
        # task two: input image edge detection
        elif self.task_id == 2:
            data_dict = self._task_two(index=index)
        # task three - input: image, output: vertebral landmarks
        elif self.task_id == 3:
            data_dict = self._task_three(index=index)
        # task four - input: image, output: facial landmarks
        elif self.task_id == 4:
            data_dict = self._task_four(index=index)
        else:
            raise ValueError("Invalid Task ID (task_id) variable. The supported values are 1,2,3, and 4!")
        # Apply transforms
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        return data_dict


class MultitaskCollator(object):
    def __init__(
        self,
        task_id: int = 1,
    ):
        """
        A collator class for creating minibatches in multitask learning scenarios.

        Args:
            task_id (int): The task ID corresponding to the specific task.
                Supported task IDs:
                    - 1: Auxiliary unsupervised image reconstruction
                    - 2: Auxiliary supervised edge detection
                    - 3: Supervised landmark detection with vertebral landmarks
                    - 4: Supervised landmark detection with facial landmarks

        Raises:
            ValueError: If the task_id is not supported.
        """
        if task_id not in [1, 2, 3, 4]:
            raise ValueError(
                "the inserted value for `task_id` variable is not supported."
                "The supported values are {}".format([1, 2, 3, 4])
            )
        self.supervised_training_keys = {
            2: 'edges', 3: 'v_landmarks', 4: 'f_landmarks'
        }
        self.task_id = task_id
    
    def __supervised_training_collation(
        self,
        batch,
    ) -> Dict[str, torch.Tensor]:
        """
        Collates the minibatch for supervised training tasks.

            Args:
                batch (list): A list of samples in the minibatch.

            Returns:
                dict: A dictionary containing the collated minibatch.
                    Keys:
                        - 'image': Tensor representing the input images.
                        - <label_key>: Tensor representing the specific task labels.
        """
        self.label_key = self.supervised_training_keys[self.task_id]
        # filter out samples with missing labels, in reality only very few samples should be dropped
        batch = [sample for sample in batch if sample[self.label_key] != None]
        if len(batch) == 0:
            return None

        # collate remaining samples as before
        images = torch.stack([sample['image'] for sample in batch])
        labels = torch.stack([sample[self.label_key] for sample in batch])
        # ... collate other labels as needed
        return {'image': images, self.label_key: labels}
    
    def __unsupervised_training_collation(
        self,
        batch,
    ) -> Dict[str, torch.Tensor]:
        """
        Collates the minibatch for unsupervised training tasks.

        Args:
            batch (list): A list of samples in the minibatch.

        Returns:
            dict: A dictionary containing the collated minibatch.
                Keys:
                    - 'image': Tensor representing the input images.
        """
        images = torch.stack([sample['image'] for sample in batch])
        return {'image': images}

    def __call__(
        self,
        batch,
    ) -> Dict[str, torch.Tensor]:
        """
        Collates the minibatch based on the task ID.

        Args:
            batch (list): A list of samples in the minibatch.

        Returns:
            dict: A dictionary containing the collated minibatch.

        """
        if self.task_id == 1:
            out = self.__unsupervised_training_collation(batch)
        else:
            out = self.__supervised_training_collation(batch)
        return out


def split_filenames(
    filenames: List[str],
    grouping_factor: List[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = None
) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits a list of filenames into train, validation, and test sets, stratified by a grouping factor.

    Args:
        filenames (List[str]): A list of filenames to be split.
        grouping_factor (List[int]): A list of grouping factors corresponding to each filename.
        train_ratio (float): Ratio of data to be used for training.
        val_ratio (float): Ratio of data to be used for validation.
        test_ratio (float): Ratio of data to be used for testing.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[List[str], List[str], List[str]]: A tuple containing the lists of filenames for
        train, validation, and test sets, respectively.
    """
    assert len(filenames) == len(grouping_factor), "filenames and grouping_factor must have the same length"
    assert train_ratio + val_ratio + test_ratio == 1, "train_ratio, val_ratio, and test_ratio must add up to 1"

    # shuffle the inputs
    combined = list(zip(filenames, grouping_factor))
    random.shuffle(combined)
    filenames, grouping_factor = zip(*combined)
    
    # Create a list of unique grouping factors
    groups = list(set(grouping_factor))

    # Calculate the number of samples for each set
    num_samples = len(filenames)
    num_train = int(num_samples * train_ratio)
    num_val = int(num_samples * val_ratio)
    num_test = num_samples - num_train - num_val

    # Initialize empty lists for the train, val, and test sets
    train_filenames, val_filenames, test_filenames = [], [], []

    # Iterate over the grouping factors and split the filenames accordingly
    for group in groups:
        group_filenames = [f for f, g in zip(filenames, grouping_factor) if g == group]
        group_num_samples = len(group_filenames)
        group_num_train = int(group_num_samples * train_ratio)
        group_num_val = int(group_num_samples * val_ratio)

        # Assign filenames to the train, val, and test sets
        train_filenames.extend(group_filenames[:group_num_train])
        val_filenames.extend(group_filenames[group_num_train:group_num_train+group_num_val])
        test_filenames.extend(group_filenames[group_num_train+group_num_val:])

    # Shuffle the train, val, and test sets
    if seed is not None:
        random.seed(seed)
    random.shuffle(train_filenames)
    random.shuffle(val_filenames)
    random.shuffle(test_filenames)

    return train_filenames, val_filenames, test_filenames


class ResizeTransform(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        # create the new height and width
        new_h, new_w = int(new_h), int(new_w)
        # resize the image
        resized_image = cv2.resize(image, (new_h, new_w), interpolation = cv2.INTER_LINEAR)
        # create the new output dict
        resized_sample = {'image': resized_image,}
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        if 'v_landmarks' in sample and sample['v_landmarks'] is not None:
            v_landmarks = sample['v_landmarks']
            v_landmarks = v_landmarks * [new_w / w, new_h / h]
            resized_sample['v_landmarks'] = v_landmarks
        if 'f_landmarks' in sample and sample['f_landmarks'] is not None:
            f_landmarks = sample['f_landmarks']
            f_landmarks = f_landmarks * [new_w / w, new_h / h]
            resized_sample['f_landmarks'] = f_landmarks
        if 'edges' in sample and sample['edges'] is not None:
            edges = sample['edges']
            resized_edges = cv2.resize(edges, (new_h, new_w), interpolation = cv2.INTER_LINEAR)
            resized_sample['edges'] = resized_edges
        return resized_sample


class Coord2HeatmapTransform(object):
    """Transform the coordinates of landmarks to heatmaps of
    the same size as input image. The heatmap is smoothed by
    a Gaussian filter.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
        gauss_std: standard deviation of the Gaussian kernel convolved with
            the landmark heatmap.
    """

    def __init__(self, output_size: Tuple[int], gauss_std: float):
        assert isinstance(output_size, (tuple))
        self.output_size = output_size
        self.gauss_std = gauss_std

    def __call__(self, sample):
        # unpack the image
        image = sample['image']
        # create the output dict
        transformed_data = {'image': image,}
        # transform coordinates to heatmaps
        if 'v_landmarks' in sample and sample['v_landmarks'] is not None:
            v_landmarks = sample['v_landmarks']
            v_landmarks = self.coord2heatmap(
                landmarks=v_landmarks,
                output_size=self.output_size,
                std=self.gauss_std,
            )
            transformed_data['v_landmarks'] = v_landmarks
        if 'f_landmarks' in sample and sample['f_landmarks'] is not None:
            f_landmarks = sample['f_landmarks']
            f_landmarks = self.coord2heatmap(
                landmarks=f_landmarks,
                output_size=self.output_size,
                std=self.gauss_std,
            )
            transformed_data['f_landmarks'] = f_landmarks
        if 'edges' in sample and sample['edges'] is not None:
            edges = sample['edges']
            transformed_data['edges'] = edges
        return transformed_data

    @staticmethod
    def coord2heatmap(
        landmarks: np.ndarray, 
        output_size: Tuple[int],
        std: float = 2.0,
    ):
        """
        Convert landmark coordinates to a heatmap.

        Args:
            landmarks (ndarray): An array of landmark coordinates.
            output_size (tuple): The size of the output heatmap (height, width).
            std (float, optional): The standard deviation for the Gaussian kernel. Defaults to 2.

        Returns:
            ndarray: The heatmap representation of the landmarks.

        """
        # get the size of the output image
        h, w = output_size
        c = landmarks.shape[0]
        # Create a black image of size (w, h)
        heatmap = np.zeros((c, h, w), dtype=np.uint8)
        # Convert the point coordinates to integer values
        landmarks = landmarks.astype(int)
        # Create a Gaussian kernel
        kernel_size = int(4 * std + 0.5)  # Set the kernel size based on the standard deviation
        kernel = cv2.getGaussianKernel(kernel_size, std)
        # normalize the kernel to have a peak at 1.0
        kernel /= kernel.max()
        # form a 2-D kernel
        kernel = np.outer(kernel.T, kernel.T)
        # Set the pixel at the point of interest to white
        for c, coord in enumerate(landmarks):
            x, y = coord
            heatmap_ = heatmap[c, :, :]
            heatmap_[y, x] = 255
            # Apply the Gaussian filter to the image
            heatmap_gauss = cv2.filter2D(heatmap_, -1, kernel)
            heatmap[c, ...] = heatmap_gauss
        return heatmap


class CustomToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # unpack the image
        image = sample['image']
        image = self.convert_to_bw(image)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        # create the output dict
        transformed_data = {'image': torch.from_numpy(image),}
        # transform coordinates to heatmaps
        if 'v_landmarks' in sample and sample['v_landmarks'] is not None:
            v_landmarks = sample['v_landmarks']
            transformed_data['v_landmarks'] = torch.from_numpy(v_landmarks)
        if 'f_landmarks' in sample and sample['f_landmarks'] is not None:
            f_landmarks = sample['f_landmarks']
            transformed_data['f_landmarks'] = torch.from_numpy(f_landmarks)
        if 'edges' in sample and sample['edges'] is not None:
            edges = sample['edges']
            edges = self.convert_to_bw(edges)
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C x H x W
            edges = edges.transpose((2, 0, 1))
            transformed_data['edges'] = torch.from_numpy(edges)
        return transformed_data
    
    @staticmethod
    def convert_to_bw(
        image: np.ndarray
    ) -> np.ndarray:
        """
        Convert an image to black and white.

        Args:
            image (ndarray): The input image.

        Returns:
            ndarray: The black and white version of the image.

        Raises:
            ValueError: If the image shape is invalid.

        """
        if len(image.shape) == 2:
            # image is black and white, add a channel axis to the end
            image = np.expand_dims(image, 2)
            return image
        elif len(image.shape) == 3 and image.shape[2] == 1:
            # image has a single channel, already black and white
            return image
        elif len(image.shape) == 3 and image.shape[2] > 1:
            # image has multiple channels, convert to black and white
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
            image = np.expand_dims(image, 2)
            return image
        else:
            raise ValueError("Invalid image shape")
            return None


def plot_image_landmarks(
    image,
    v_landmarks=None,
    f_landmarks=None,
):
    """
    Plot an image with landmarks.

    Args:
        image: The input image to be plotted.
        v_landmarks (ndarray, optional): An array of landmarks for visible points. Defaults to None.
        f_landmarks (ndarray, optional): An array of landmarks for fiducial points. Defaults to None.

    Returns:
        None

    """
    fig, ax = plt.subplots(figsize=(5,5),)
    ax.imshow(image, cmap='gray')
    # Plot points on top of image
    if v_landmarks is not None:
        for i in range(v_landmarks.shape[0]):
            ax.scatter(v_landmarks[i,0], v_landmarks[i,1], color='red', s=10)
    if f_landmarks is not None:
        for i in range(f_landmarks.shape[0]):
            ax.scatter(f_landmarks[i,0], f_landmarks[i,1], color='orange', s=10)
    # set the ticker
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(100))
    # get the current tick locations and labels
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()

    # set the rotation angle for the tick labels
    ax.set_xticklabels(xticks, rotation=45)

    # Show plot
    plt.show()
    return None


class MultiTaskLandmarkUNetCustom(nn.Module):
    """Multi-task U-Net architecture for landmark detection and image processing tasks"""

    def __init__(
        self, 
        in_channels: int=1, 
        out_channels1: int=1,
        out_channels2: int=1,
        out_channels3: int=13,
        out_channels4: int=19,
        enc_chan_multiplier: int=1,
        dec_chan_multiplier: int=1,
        backbone_encoder: Union[None, str] = None,
        backbone_weights: Union[str, None] = "imagenet",
        freeze_backbone: bool = True,
        enc_out_chans: np.ndarray = np.array([4, 8, 16, 32, 64]),
        dec_out_chans: np.ndarray = np.array([64, 32, 16, 8, 4]),
    ) -> None:
        """
        Initialize the MultiTaskLandmarkUNet1 model

        Parameters:
            in_channels (int): Number of input channels (default: 3)
            out_channels1 (int): Number of output channels for image reconstruction (default: 1)
            out_channels2 (int): Number of output channels for edge detection (default: 1)
            out_channels3 (int): Number of output channels for v landmark detection (default: 13)
            out_channels4 (int): Number of output channels for f landmark detection (default: 19)
            enc_chan_multiplier (int): Multiplication factor that increases the hidden channels of the encoder (default: 1)
            dec_chan_multiplier (int): Multiplication factor that increases the hidden channels of the decoder (default: 1)
            backbone_encoder: Union[None, str] = None,
            backbone_weights: Union[str, None] = "imagenet",
        """
        super().__init__()
        
        assert isinstance(enc_chan_multiplier, int)
        assert isinstance(dec_chan_multiplier, int)
        
        # encoder
        self.backbone_encoder = backbone_encoder
        if backbone_encoder is not None:
            self.encoder = get_encoder(
                backbone_encoder,
                in_channels=1,
                depth=5,
                weights=backbone_weights,
            )
            if freeze_backbone:
                for param in self.encoder.parameters():
                    param.requires_grad = False
            # infer the encoder output channels from the model itself
            out = self.encoder(torch.randn(1, 1, 256, 256))
            if not isinstance(out, list) and len(out)<5:
                raise ValueError(
                    "The selected backbone for the encoder does not "
                    "have sufficient depth! The encoder has to output "
                    "at least 5 feature maps!"
                )
            out.reverse()
            enc_out_chans = [out[i].shape[1] for i in range(len(out))]
            enc_out_chans = enc_out_chans[:5]
            enc_out_chans.reverse()

        else:
            # set the number of output channels in the encoder
            self.dconv1 = DoubleConv(in_channels, enc_out_chans[0])
            self.dconv2 = DoubleConv(enc_out_chans[0], enc_out_chans[1])
            self.dconv3 = DoubleConv(enc_out_chans[1], enc_out_chans[2])
            self.dconv4 = DoubleConv(enc_out_chans[2], enc_out_chans[3])
            self.dconv5 = DoubleConv(enc_out_chans[3], enc_out_chans[4])
            
        # define the number of channels of encoder and the decoder
        enc_out_chans = enc_out_chans*enc_chan_multiplier    
        dec_out_chans = dec_out_chans*dec_chan_multiplier
        
        # bridge
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # decoder
        self.upconv4 = DoubleConv(enc_out_chans[4] + enc_out_chans[3], dec_out_chans[0])
        self.upconv3 = DoubleConv(dec_out_chans[0] + enc_out_chans[2], dec_out_chans[1])
        self.upconv2 = DoubleConv(dec_out_chans[1] + enc_out_chans[1], dec_out_chans[2])
        self.upconv1 = DoubleConv(dec_out_chans[2] + enc_out_chans[0], dec_out_chans[3])
        
        # outputs
        # task 1
        self.conv1_1 = DoubleConv(dec_out_chans[3], dec_out_chans[4],)
        self.conv1_2 = nn.Conv2d(dec_out_chans[4], out_channels1, kernel_size=1)
        
        # task 2
        self.conv2_1 = DoubleConv(dec_out_chans[3], dec_out_chans[4],)
        self.conv2_2 = nn.Conv2d(dec_out_chans[4], out_channels2, kernel_size=1)

        # task 3
        self.conv3_1 = DoubleConv(dec_out_chans[3], dec_out_chans[4],)
        self.conv3_2 = nn.Conv2d(dec_out_chans[4], out_channels3, kernel_size=1)
        
        # task 4
        self.conv4_1 = DoubleConv(dec_out_chans[3], dec_out_chans[4],)
        self.conv4_2 = nn.Conv2d(dec_out_chans[4], out_channels4, kernel_size=1)

    def forward(self, x: torch.Tensor, task_id: int):
        # store the input shape
        inp_shape = x.shape
        # Down sampling
        if self.backbone_encoder is not None:
            _, x1, x3, x5, x7, x9 = self.encoder(x)
        else:
            x1 = self.dconv1(x)
            x2 = self.maxpool(x1)
            x3 = self.dconv2(x2)
            x4 = self.maxpool(x3)
            x5 = self.dconv3(x4)
            x6 = self.maxpool(x5)
            x7 = self.dconv4(x6)
            x8 = self.maxpool(x7)
            x9 = self.dconv5(x8)

        # Up sampling
        x = self.upsample(x9)
        x = torch.cat([x, x7], dim=1)
        x = self.upconv4(x)
        x = self.upsample(x)
        x = torch.cat([x, x5], dim=1)
        x = self.upconv3(x)
        x = self.upsample(x)
        x = torch.cat([x, x3], dim=1)
        x = self.upconv2(x)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.upconv1(x)

        # upsample once more if we have a backbone encoder
        if self.backbone_encoder is not None:
            x = self.upsample(x)

        # return the outputs based on the task
        if task_id == 1:
            # Output for task 1: unsupervised image reconstruction
            x = self.conv1_1(x)
            x = self.conv1_2(x)
        elif task_id == 2:
            # Output for task 2: supervised edge detection
            x = self.conv2_1(x)
            x = self.conv2_2(x)
        elif task_id == 3:
            # Output for task 3: supervised vertebral landmark detection
            x = self.conv3_1(x)
            x = self.conv3_2(x)
        elif task_id == 4:
            # Output for task 4: supervised facial landmark detection
            x = self.conv4_1(x)
            x = self.conv4_2(x)
        else:
            raise ValueError('Bad Task ID passed')
        return x


class DoubleConv(nn.Module):
    """Double Convolution Layer"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

    
def plot_many_heatmaps(
    heatmap_tensor: torch.Tensor,
    cols: int = 4,
):
    """
    Plot multiple heatmaps in a grid layout.

    Args:
        heatmap_tensor (torch.Tensor): A tensor containing the heatmaps to be plotted.
        cols (int, optional): The number of columns in the grid layout. Defaults to 4.

    Returns:
        None

    """
    # change to numpy
    heatmap_tensor = heatmap_tensor.numpy()
    # number of heatmaps
    n_heatmaps = heatmap_tensor.shape[0]
    rows = n_heatmaps//cols
    if n_heatmaps%cols != 0:
        rows += 1
    # create a new figure
    fig = plt.figure(tight_layout=True, figsize=(10,10))
    gs = gridspec.GridSpec(rows, cols)
    
    c = 0
    r = 0
    for i, heatmap in enumerate(heatmap_tensor):
        if i != 0 and i % 4 == 0:
            r += 1
            c = 0
        # find the location of Gaussian peak
        max_index = unravel_index(heatmap.argmax(), heatmap.shape)
        # show the image
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(heatmap, cmap='gray')
        ax.set_title("Gauss peak coord: {}".format(','.join(map(str, max_index))))
        c += 1
    # title
    # plt.suptitle("dataset is {}".format(dataset))
    # display the plot
    plt.tight_layout()
    plt.show()
    plt.pause(1)
    return None


class MultitaskTrainOnlyLandmarks(pl.LightningModule):
    def __init__(self, model, *args, **kwargs,):
        super().__init__(*args, **kwargs,)
        # model
        self.model = model
        
        # metrics
        self.train_mse_task_1 = MeanSquaredError(name="train_mse_1")
        self.train_mse_task_2 = MeanSquaredError(name="train_mse_2")
        self.val_mse_task_1 = MeanSquaredError(name="val_mse_1")
        self.val_mse_task_2 = MeanSquaredError(name="val_mse_2")

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
        self.train_mse = MeanSquaredError(name="train_mse")
        self.val_mse = MeanSquaredError(name="val_mse")

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
    batch_size: int = 16,
    split: str = 'train',
) -> torch.utils.data.DataLoader:
    # load metadata
    metadata_table = pd.read_hdf(
        os.path.join(PARAMS.PRIMARY_DATA_DIRECTORY, PARAMS.TRAIN.METADATA_TABLE_NAME),
        key='df',
    )
    # create the right list of paths
    train_file_list = metadata_table.loc[
        (metadata_table['split']==split) & (metadata_table['v_annots_present']==True), ['harmonized_id']
    ].to_numpy().ravel().tolist()
    train_file_list = [
        os.path.join(PARAMS.PRIMARY_DATA_DIRECTORY, file_path+'.hdf5') for file_path in train_file_list
    ]
    # instantiate the transforms
    my_transforms = transforms.Compose([
        ResizeTransform(tuple(PARAMS.TRAIN.TARGET_IMAGE_SIZE)),
        Coord2HeatmapTransform(
            tuple(PARAMS.TRAIN.TARGET_IMAGE_SIZE),
            PARAMS.TRAIN.GAUSSIAN_COORD2HEATMAP_STD,
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
    shuffle = True if split == 'train' else False
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator_task,
        num_workers=4,
    )
    return dataloader


def trainer_multitask_v_and_f_landmarks():
    # initialize the model
    model_params = PARAMS.MODEL.PARAMS
    model = MultiTaskLandmarkUNetCustom(**model_params)
    # initialize dataloader and trainer objects
    train_dataloaders = {}
    val_dataloaders = {}

    for task_config in PARAMS.TRAIN.MULTIPLE_TASKS:
        # unpack parameters for easier readability
        task_id = task_config.TASK_ID
        batch_size = task_config.BATCH_SIZE
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
        max_epochs=PARAMS.TRAIN.MAX_EPOCHS,
        log_every_n_steps=1,
    )
    # run the training
    trainer.fit(
        pl_model,
        train_dataloaders=train_dataloaders,
        val_dataloaders=val_dataloaders,
    )
    return None


def trainer_v_landmarks_single_task():
    # get the parameters
    task_config = PARAMS.TRAIN.SINGLE_TASK
    task_id = task_config.TASK_ID
    batch_size = task_config.BATCH_SIZE
    # initialize the model
    model_params = PARAMS.MODEL.PARAMS
    model = MultiTaskLandmarkUNetCustom(**model_params)
    # initialize dataloader and trainer objects
    # train dataloader
    train_dataloader = create_dataloader(
        task_id=task_id,
        batch_size=batch_size,
        split='train',
    )
    # val dataloader
    val_dataloader = create_dataloader(
        task_id=task_id,
        batch_size=batch_size,
        split='val',
    )
    # initialize trainer
    pl_model = SingletaskTrainLandmarks(
        model=model,
    )
    trainer = pl.Trainer(        
        max_epochs=PARAMS.TRAIN.MAX_EPOCHS,
        log_every_n_steps=1,
    )
    # run the training
    trainer.fit(
        pl_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    return None
