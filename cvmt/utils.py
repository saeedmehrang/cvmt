"""Shared utilities that are used by data and/or ml modules."""

from typing import *

from easydict import EasyDict
import yaml
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time
import matplotlib.pyplot as plt


def nested_dict_to_easydict(nested_dict: Dict) -> EasyDict:
    if isinstance(nested_dict, dict):
        for key, value in nested_dict.items():
            nested_dict[key] = nested_dict_to_easydict(value)
        return EasyDict(nested_dict)
    elif isinstance(nested_dict, list):
        return [nested_dict_to_easydict(item) for item in nested_dict]
    else:
        return nested_dict


def load_yaml_params(path) -> Dict[str, Any]:
    with open(path, 'r') as f:
        params = yaml.safe_load(f)
    return params


def remove_lightning_logs_dir():
    path = "lightning_logs"
    if os.path.exists(path):
        shutil.rmtree(path)


def rescale_landmarks(
    landmarks: List[Tuple[int]],
    original_size: Tuple[int],
    input_size: Tuple[int] = (256, 256),
) -> np.ndarray:
    """
    Rescale landmarks to original image size.

    Parameters:
    landmarks (list): List of tuples representing landmarks in the format (x, y).
    original_size (tuple): Original size of the image in the format (height, width).
    model_size (tuple): Size of the image used by the model for prediction in the format (height, width).

    Returns:
    list: List of tuples representing rescaled landmarks.
    """
    height_ratio = original_size[0] / input_size[0]
    width_ratio = original_size[1] / input_size[1]

    rescaled_landmarks = [( y*height_ratio, x*width_ratio,) for y, x in landmarks]
    rescaled_landmarks = np.around(rescaled_landmarks, 1)
    return rescaled_landmarks


def img_coord_2_cartesian_coord(landmarks: np.ndarray, swap_x_y: bool=True) -> np.ndarray:
    """Image coordinates are defined with a reverted y (height) axis. Here
    it is changed so y axis has proper direction. Also, x and y axes order are 
    swapped, so x comes first and then y comes.
    """
    landmarks_ = landmarks.copy()
    # swap height and width
    if swap_x_y:
        landmarks_ = np.flip(landmarks_, 1)
    # invert the y axis as the original y axis in an image is inverted
    landmarks_[:, 1] = -1 * landmarks_[:, 1]
    return landmarks_


def translate_landmarks(
    landmarks: Union[List, Tuple, np.ndarray],
    ref_index: int,
) -> np.ndarray:
    """Shifting all the landmarks coordinates such that the landmark at the
    ref_index is regarded as the origin of the cartesian plane.
    """
    # Ensure landmarks is a numpy array
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array(landmarks)
    # Get the reference point
    ref_point = landmarks[ref_index]
    # Translate all points such that the reference point is at the origin
    translated_landmarks = landmarks - ref_point
    return translated_landmarks


def plot_landmarks(landmarks: np.ndarray):
    """Debugging plots for simply visualizing the shape and the location
    of all landmarks.
    """
    # Ensure landmarks is a numpy array
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array(landmarks)
    
    # Separate the heights and widths into separate arrays for plotting
    heights = landmarks[:, 1]
    widths = landmarks[:, 0]
    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(5,5))
    colors = np.concatenate(
        (
            ['r' for i in range(3)],
            ['b' for i in range(5)],
            ['g' for i in range(5)],
        )
    )
    markers = np.concatenate(
        (
            ["P" for i in range(3)],
            ["o" for i in range(5)],
            ["*" for i in range(5)],
        )
    )
    for w, h, c, m in zip(widths, heights, colors, markers):
        plt.scatter(w, h , c=c, marker=m)
    # Add labels and title
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Landmarks Scatter Plot')
    plt.axis('equal')
    
    # Show the plot
    plt.show()

    
def rotate_landmarks(landmarks: np.ndarray, ref_index: int) -> np.ndarray:
    """Rotate the translated landmarks such that the two lowest points are both placed 
    at the x axis (y=0).
    """
    # Ensure landmarks is a numpy array
    if not isinstance(landmarks, np.ndarray):
        landmarks = np.array(landmarks)
    # Get the reference point
    ref_point = landmarks[ref_index]
    # Calculate the angle between the reference point and the x-axis
    angle = np.arctan2(ref_point[1], ref_point[0])
    angle = -1*angle
    # Create a rotation matrix
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    # Rotate all points
    rotated_landmarks = np.dot(landmarks, rotation_matrix.T)
    return rotated_landmarks


def plot_image_and_vertebral_landmarks(
    img_name: str,
    model_id: str,
    landmarks: np.ndarray,
    image: np.ndarray,
    save: bool = False,
):
    """Create a grid of two and plot the same image in both of the grids. Then,
    overlay the predicted landmarks on the second image. The landmarks must be
    scaled to the image size.
    """
    fig = plt.figure(tight_layout=True, figsize=(15,15))
    gs = gridspec.GridSpec(1, 2)

    # show the image
    ax = fig.add_subplot(gs[0])
    ax.imshow(image, cmap='gray')

    ax = fig.add_subplot(gs[1])
    ax.imshow(image, cmap='gray')

    # plot the landmarks on top of the image
    ax.scatter(landmarks[:, 1], landmarks[:, 0], s=10, c='cyan')
    for i, l in enumerate(landmarks):
        ax.text(l[1], l[0], str(i), color='greenyellow')  # Annotate the index

    plt.tight_layout()
    plt.show()
    if save:
        verify_dir = "artifacts/verification"
        fig.savefig(os.path.join(verify_dir, f"{img_name}_{model_id}.jpg"), dpi=300)
    time.sleep(1)


def normalize_coords(
    landmarks: np.ndarray,
    ref_index: int,
    height_wise: bool = False,
) -> np.ndarray:
    """Normalize all the landmarks based on the height of the point specified at
    index `ref_index`.
    """
    ref_point = landmarks[ref_index]
    if height_wise:
        ratio = ref_point[1]
    else:
        ratio = ref_point[0]
    normalized_landmarks = np.divide(landmarks, ratio)
    normalized_landmarks = np.around(normalized_landmarks, 2)
    return normalized_landmarks
