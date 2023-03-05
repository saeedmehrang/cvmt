import json
import os
import yaml
import shutil
import glob
from typing import List, Dict, Tuple, Union, Any
from numbers import Number
import h5py
from PIL import Image
import numpy as np


with open("../../code_configs/params.yaml") as f:
    params = yaml.safe_load(f)


def load_and_preprocess_image_and_annotations(
    image_filename: str,
    image_dir: str,
    annot_dir: str,
    unwanted_fields: List[str],
) -> Tuple[np.ndarray, Union[Dict, None]]:
    image_path = os.path.join(
        image_dir,
        image_filename,
    )
    image = np.asarray(Image.open(
        image_path,
        mode='r',
    ))
    # create a placeholder for annotations
    annots = None
    # create annotation path
    filename = image_filename.split('.')[0]
    annot_filename = filename+'.json'
    annot_path = os.path.join(
        annot_dir,
        annot_filename,
    )
    # read annotations if exist
    if os.path.exists(annot_path):
        with open(annot_path, 'r') as f:
            annots = json.load(f)
        # remove the unwanted image data
        for field in unwanted_fields:
            if field in annots:
                annots[field] = None
    return image, annots


def save_image_and_annots_hdf5(
    save_dir: str,
    filename: str,
    image: np.ndarray,
    annots: Union[Dict, None],
):
    if not filename.endswith('.hdf5'):
        filename += '.hdf5'
    # create the filepath
    filepath = os.path.join(save_dir, filename)
    # check if it exists
    if os.path.exists(filepath):
        raise FileExistsError(
            "The file exsist! Either remove the existing file or change the save directory!"
        )
    # create the hdf5 file handle    
    f = h5py.File(filepath, 'w')
    # write image data
    image_data_grp = f.create_group("image")
    image_data_grp.create_dataset('data', data=image)
    # write annotation data
    if annots is not None:
        annots_data_grp = f.create_group("annots")
        # wite the annotation data
        annots_data_grp.attrs['version'] = annots['version']
        annots_data_grp.attrs['imageHeight'] = annots['imageHeight']
        annots_data_grp.attrs['imageWidth'] = annots['imageWidth']
        shapes = annots_data_grp.create_group('shapes')
        for shape in annots['shapes']:
            group_id = str(shape['group_id'])
            grp_shape = shapes.create_group(group_id)
            grp_shape.attrs['label'] = shape['label']
            grp_shape.attrs['shape_type'] = shape['shape_type']
            points = np.array(shape['points']).astype(int)
            grp_shape.create_dataset('points', data=points)        
    # close the h5py
    f.close()
    return True


def harmonize_hdf5(
    image_filename: str,
    image_dir: str,
    annot_dir: str,
):
    image, annots = load_and_preprocess_image_and_annotations(
        image_filename=image_filename,
        image_dir=image_dir,
        annot_dir=annot_dir,
        unwanted_fields=params['UNWANTED_JSON_FIELDS'],
    )
    h5py_filename = image_filename.split('.')[0]+'.hdf5'
    save_image_and_annots_hdf5(
        save_dir=params['PRIMARY_DATA_DIRECTORY'],
        filename=h5py_filename,
        image=image,
        annots=annots,
    )
    return None


def read_harmonized_hdf5(
    h5py_filename: str,
):
    h5py_file = os.path.join(params['PRIMARY_DATA_DIRECTORY'], h5py_filename)
    f = h5py.File(h5py_file, 'r')
    # read image
    image = f['image']['data'][:]
    # read annotations
    vertebrate_ids, landmarks, label, shape_type = None, None, None, None
    if 'annots' in f.keys():
        landmarks, labels, shape_types = [], [], []
        vertebrate_ids = list(f['annots']['shapes'].keys())
        for vertebrate_id in vertebrate_ids:
            landmarks = f['annots']['shapes'][vertebrate_id]['points'][:]
            label = f['annots']['shapes'][vertebrate_id].attrs['label']
            shape_type = f['annots']['shapes'][vertebrate_id].attrs['shape_type']
    return image, vertebrate_ids, landmarks, label, shape_type