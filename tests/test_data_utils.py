"""Unit tests for data utils"""

import pytest
import unittest
from cvmt.data.utils import load_image
import numpy as np


IMAGE_DIR = "../docs/images"
IMAGE_FILENAME = "155.png"


# This is your test case
def test_load_image():
    image = load_image(image_dir=IMAGE_DIR, image_filename=IMAGE_FILENAME)
    assert isinstance(
        image, np.ndarray
    )  # check if the returned object is a numpy array
