""" A collection of data utility functions and classes plus additional functions for 
plotting and data transformations needed during training deep learning models."""

import random
# from collections import OrderedDict
# from copy import deepcopy
from typing import Any, Callable, Dict, List, Tuple

import cv2
import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from numpy import unravel_index
# from pytorch_lightning.loggers import TensorBoardLogger
# from segmentation_models_pytorch.base.modules import Activation
from torch.utils.data import Dataset


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