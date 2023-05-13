import random
from typing import Callable, Dict, List, Tuple, Any, Union

import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms


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
        r_transform = transforms.Resize((new_h, new_w), antialias=True)
        image = convert_to_bw(image)
        image = image.transpose((2, 0, 1)) # transpose to (C, H, W)
        image = torch.from_numpy(image)
        resized_image = r_transform(image)
        # create the new output dict
        resized_sample = {'image': resized_image,}
        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        if 'v_landmarks' in sample and sample['v_landmarks'] is not None:
            v_landmarks = sample['v_landmarks']
            v_landmarks = v_landmarks * [new_w / w, new_h / h]
            resized_sample['v_landmarks'] = torch.from_numpy(v_landmarks)
        if 'f_landmarks' in sample and sample['f_landmarks'] is not None:
            f_landmarks = sample['f_landmarks']
            f_landmarks = f_landmarks * [new_w / w, new_h / h]
            resized_sample['f_landmarks'] = torch.from_numpy(f_landmarks)
        if 'edges' in sample and sample['edges'] is not None:
            edges = sample['edges']
            edges = convert_to_bw(edges)
            edges = edges.transpose((2, 0, 1)) # transpose to (C, H, W)
            edges = torch.from_numpy(edges)
            resized_edges = r_transform(edges)
            resized_sample['edges'] = resized_edges
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

        
def plot_image_landmarks(
    image,
    v_landmarks=None,
    f_landmarks=None,
):
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


class MultiTaskLandmarkUNet1(nn.Module):
    """Multi-task U-Net architecture for landmark detection and image processing tasks"""

    def __init__(
        self, 
        in_channels: int=3, 
        out_channels1: int=1,
        out_channels2: int=1,
        out_channels3: int=13,
        out_channels4: int=19,
        enc_chan_multiplier: int=1,
        dec_chan_multiplier: int=1,
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
        """
        super().__init__()
        
        assert isinstance(enc_chan_multiplier, int)
        assert isinstance(dec_chan_multiplier, int)
        
        enc_out_chans = np.array([16, 16, 32, 32, 64])*enc_chan_multiplier
        dec_out_chans = np.array([64, 32, 32, 32, 32])*dec_chan_multiplier
        dec_in_chan_start = 128*dec_chan_multiplier
        
        # encoder
        self.dconv1 = DoubleConv(in_channels, enc_out_chans[0])
        self.dconv2 = DoubleConv(enc_out_chans[0], enc_out_chans[1])
        self.dconv3 = DoubleConv(enc_out_chans[1], enc_out_chans[2])
        self.dconv4 = DoubleConv(enc_out_chans[2], enc_out_chans[3])
        self.dconv5 = DoubleConv(enc_out_chans[3], enc_out_chans[4])
        
        # bridge
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # decoder
        self.upconv4 = DoubleConv(dec_in_chan_start + enc_out_chans[4], dec_out_chans[0])
        self.upconv3 = DoubleConv(dec_out_chans[0] + enc_out_chans[3], dec_out_chans[1])
        self.upconv2 = DoubleConv(dec_out_chans[1] + enc_out_chans[2], dec_out_chans[2])
        self.upconv1 = DoubleConv(dec_out_chans[2] + enc_out_chans[1], dec_out_chans[3])
        
        # outputs
        self.conv1_1 = DoubleConv(dec_out_chans[3], dec_out_chans[4],)
        self.conv1_2 = nn.Conv2d(dec_out_chans[4], out_channels1, kernel_size=1)
        
        self.conv2_1 = DoubleConv(dec_out_chans[3], dec_out_chans[4],)
        self.conv2_2 = nn.Conv2d(dec_out_chans[4], out_channels2, kernel_size=1)

        self.conv3_1 = DoubleConv(dec_out_chans[3], dec_out_chans[4],)
        self.conv3_2 = nn.Conv2d(dec_out_chans[4], out_channels3, kernel_size=1)
        
        self.conv4_1 = DoubleConv(dec_out_chans[3], dec_out_chans[4],)
        self.conv4_2 = nn.Conv2d(dec_out_chans[4], out_channels4, kernel_size=1)

    def forward(self, x: torch.Tensor, task_id: int):
        # Down sampling
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
        
        # Output for task 3: auxiliary supervised edge detection
        x_edge = self.conv3_1(x)
        x_edge = self.conv3_2(x_edge)

        # Output for task 3: auxiliary unsupervised image reconstruction
        x_recon = self.conv4_1(x)
        x_recon = self.conv4_2(x_recon)
    
        if task_id == 0:
            # Output for task 1: supervised vertebral landmark detection
            x = self.conv1_1(x)
            x = self.conv1_2(x)
        elif task_id == 1:
            # Output for task 2: supervised facial landmark detection
            x = self.conv2_1(x)
            x = self.conv2_2(x)
        else:
            raise ValueError('Bad Task ID passed')
        return (x_edge, x_recon, x)


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
