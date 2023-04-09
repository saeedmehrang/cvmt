import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import random
from typing import List, Tuple
from PIL import Image
import torch
from torchvision import transforms


class HDF5MultitaskDataset(Dataset):
    def __init__(
        self,
        file_paths,
        transforms=None,
        landmark_keys=['v_landmarks', 'f_landmarks'],
        landmark_shapes={'v_landmarks': (13,2), 'f_landmarks': (19,2)},
    ):
        self.file_paths = file_paths
        self.transforms = transforms
        self.landmark_keys = landmark_keys
        self.landmark_shapes = landmark_shapes

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        # read the hdf5 file
        f = h5py.File(self.file_paths[index], "r")
        # read image data
        image = f['image']['data'][:]
        # read landmarks
        data_dict = {'image': image, 'v_landmarks': None, 'f_landmarks': None}
        # read vertebral landmarks
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
        # read facial landmarks
        if 'f_landmarks' in f.keys():
            f_landmarks = f['f_landmarks']['points'][:]
            if self.landmark_shapes['f_landmarks'] == f_landmarks.shape:
                data_dict['f_landmarks'] = f_landmarks
        # apply transforms if there are any
        if self.transforms is not None:
            data_dict = self.transforms(data_dict)
        # close the hdf5 file
        f.close()
        return data_dict


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


class ResizeAndLandmarksTransform(object):
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
        image, v_landmarks, f_landmarks = sample['image'], sample['v_landmarks'], sample['f_landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        # create the new height and width
        new_h, new_w = int(new_h), int(new_w)
        # resize the image
        r_transform = transforms.Resize((new_h, new_w))
        image = convert_to_bw(image)
        image = image.transpose((2, 0, 1)) # transpose to (C, H, W)
        image = torch.from_numpy(image)
        resized_image = r_transform(image)
        # create the new output dict
        resized_sample = {'image': resized_image, 'v_landmarks': None, 'f_landmarks': None}
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        if sample['v_landmarks'] is not None:
            v_landmarks = sample['v_landmarks'] * [new_w / w, new_h / h]
            resized_sample['v_landmarks'] = torch.from_numpy(v_landmarks)
        if sample['f_landmarks'] is not None:
            f_landmarks = sample['f_landmarks'] * [new_w / w, new_h / h]
            resized_sample['f_landmarks'] = torch.from_numpy(f_landmarks)

        return resized_sample


def convert_to_bw(image):
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
