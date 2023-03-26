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
import hashlib
from scipy.spatial.distance import pdist
from itertools import combinations


with open("../../code_configs/params.yaml") as f:
    params = yaml.safe_load(f)


CORRECT_V_LANDMARKS_SHAPE = {
    '2': 3, '3': 5, '4': 5,
}

    
def load_image(
    image_filename: str,
    image_dir: str,
) -> np.ndarray:
    image_path = os.path.join(
        image_dir,
        image_filename,
    )
    image = np.asarray(Image.open(
        image_path,
        mode='r',
    ))
    return image


def load_and_clean_vertebral_annots(
    image_filename:str,
    v_annot_dir: str,
    unwanted_fields_v_annot: List[str],
) -> Union[Dict, None]:
    # create a placeholder for vertebral annotations
    v_annots = None
    # create vertebral annotation path
    filename = image_filename.split('.')[0]
    v_annot_filename = filename+'.json'
    v_annot_path = os.path.join(
        v_annot_dir,
        v_annot_filename,
    )
    # read vertebral annotations if exist
    if os.path.exists(v_annot_path):
        with open(v_annot_path, 'r') as f:
            v_annots = json.load(f)
    # clean the data
    if v_annots is not None:
        try:
            # remove the unwanted image data
            for field in unwanted_fields_v_annot:
                if field in v_annots:
                    v_annots[field] = None
            # apply further corrections
            # correct the shape groupnames
            shape_grps = [shape['group_id'] for shape in v_annots['shapes']]
            shape_grps = correct_shape_group_names(shape_grps)
            for shape_grp, shape in zip(shape_grps, v_annots['shapes']):
                shape['group_id'] = shape_grp
            # drop the extra landmark that is added sometimes
            landmarks_list = [shape['points'] for shape in v_annots['shapes']]
            landmarks_list = [
                drop_extra_landmarks(landmarks, shape_grp) 
                for shape_grp, landmarks in zip(shape_grps, landmarks_list)]
            for landmarks, shape in zip(landmarks_list, v_annots['shapes']):
                shape['points'] = landmarks
        except Exception as e:
            print(e)
            print("Error above encountered! V landmarks set to None.")
    return v_annots


def load_facial_annots(
    image_filename:str,
    f_annot_dir: str,
) -> Union[np.ndarray, None]:
    # create a placeholder for facial annotations
    f_annots = None
    # create vertebral annotation path
    if f_annot_dir is not None:
        filename = image_filename.split('.')[0]
        f_annot_filename = filename+'.txt'
        f_annot_path = os.path.join(
            f_annot_dir,
            f_annot_filename,
        )
        # read vertebral annotations if exist
        if os.path.exists(f_annot_path):
            f_annots = np.loadtxt(f_annot_path, max_rows=19, delimiter=',')
    return f_annots


def load_and_clean_image_and_annotations(
    image_filename: str,
    image_dir: str,
    v_annot_dir: Union[str, None],
    f_annot_dir: Union[str, None],
    unwanted_fields_v_annot: List[str],
) -> Tuple[np.ndarray, Union[Dict, None], np.ndarray]:
    image = load_image(
        image_filename=image_filename,
        image_dir=image_dir,
    )
    v_annots = load_and_clean_vertebral_annots(
        image_filename=image_filename,
        v_annot_dir=v_annot_dir,
        unwanted_fields_v_annot=unwanted_fields_v_annot,
    )
    f_annots = load_facial_annots(
        image_filename=image_filename,
        f_annot_dir=f_annot_dir,
    )
    return image, v_annots, f_annots


def correct_shape_group_names(
    shape_grps: List[Union[str, int]],
) -> List[str]:
    shape_grps = [str(n) for n in shape_grps]
    if all(len(n) == 1 for n in shape_grps) is False:
        shape_grps = [n[0] for n in shape_grps]
    if len(set(shape_grps)) != 3:
        raise ValueError("The shape group names are malformed!")
    else:
        return shape_grps


def drop_extra_landmarks(
    landmarks: List[List[float]],
    shape_grp: str,
) -> np.ndarray:
    landmarks = np.array(landmarks)
    if landmarks.shape[0] > CORRECT_V_LANDMARKS_SHAPE[shape_grp]:
        indices = list(combinations(range(len(landmarks)), 2))
        distances = pdist(landmarks, 'euclidean')
        index = np.argmin(distances)
        duplicate_row = indices[index][-1]
        corrected_landmarks = np.delete(landmarks, duplicate_row, axis=0)
        return corrected_landmarks
    elif landmarks.shape[0] < CORRECT_V_LANDMARKS_SHAPE[shape_grp]:
        raise ValueError(
            "Number of annotated landmarks does not match the"
            " expected value!"
        )
    else:
        return landmarks
    


def write_v_annots(
    f: h5py._hl.files.File,
    v_annots: Union[Dict, None],
) -> bool:
    v_annots_data_grp = f.create_group("v_landmarks")
    # wite the annotation data
    v_annots_data_grp.attrs['version'] = v_annots['version']
    v_annots_data_grp.attrs['imageHeight'] = v_annots['imageHeight']
    v_annots_data_grp.attrs['imageWidth'] = v_annots['imageWidth']
    shapes = v_annots_data_grp.create_group('shapes')
    for shape in v_annots['shapes']:
        group_id = str(shape['group_id'])
        grp_shape = shapes.create_group(group_id)
        grp_shape.attrs['label'] = shape['label']
        grp_shape.attrs['shape_type'] = shape['shape_type']
        points = np.array(shape['points']).astype(int)
        grp_shape.create_dataset('points', data=points)        
    return True


def write_f_annots(
    f: h5py._hl.files.File,
    f_annots: Union[np.ndarray, None],
) -> bool:
    f_annots_data_grp = f.create_group("f_landmarks")
    points = np.array(f_annots).astype(int)
    f_annots_data_grp.create_dataset('points', data=points)
    return True


def write_image(
    f: h5py._hl.files.File,
    image: np.ndarray,
) -> bool:
    image_data_grp = f.create_group("image")
    image_data_grp.create_dataset('data', data=image)
    return True


def save_image_and_annots_hdf5(
    save_dir: str,
    harmonized_id: str,
    image: np.ndarray,
    v_annots: Union[Dict, None],
    f_annots: Union[np.ndarray, None],
) -> bool:
    filename = harmonized_id+'.hdf5'
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
    write_image(f, image)
    # write vertebral landmark annotation data
    if v_annots is not None:
        try:
            write_v_annots(f, v_annots,)
        except Exception as e:
            print(e)
            print("--- v annots was not None, but, the code encountered an error!")
    # write facial landmark annotation data
    if f_annots is not None:
        try:
            write_f_annots(f, f_annots,)
        except Exception as e:
            print(e)
            print("--- f annots was not None, but, the code encountered an error!")
    # close the h5py
    f.close()
    return True


def create_hash_code(
    image_dir: str,
    image_filename: str,
):
    fname = os.path.join(image_dir, image_filename)
    hashcode = hashlib.sha256(fname.encode('utf-8')).hexdigest()
    return hashcode

    
def harmonize_hdf5(
    image_filename: str,
    image_dir: str,
    v_annot_dir: str,
    f_annot_dir: str,
) -> Tuple[str, str, bool, bool]:
    image, v_annots, f_annots = load_and_clean_image_and_annotations(
        image_filename=image_filename,
        image_dir=image_dir,
        v_annot_dir=v_annot_dir,
        f_annot_dir=f_annot_dir,
        unwanted_fields_v_annot=params['UNWANTED_JSON_FIELDS'],
    )
    harmonized_id = create_hash_code(
        image_dir=image_dir,
        image_filename=image_filename,
    )
    save_image_and_annots_hdf5(
        save_dir=params['PRIMARY_DATA_DIRECTORY'],
        harmonized_id=harmonized_id,
        image=image,
        v_annots=v_annots,
        f_annots=f_annots,
    )
    v_annots_present = True if v_annots is not None else False
    f_annots_present = True if f_annots is not None else False
    return harmonized_id, v_annots_present, f_annots_present


def read_harmonized_hdf5(
    h5py_filename: str,
):
    h5py_file = os.path.join(params['PRIMARY_DATA_DIRECTORY'], h5py_filename)
    f = h5py.File(h5py_file, 'r')
    # read image
    image = f['image']['data'][:]
    # read vertebral annotations
    vertebrate_ids, v_landmarks, v_labels, v_shape_types = None, None, None, None
    if 'v_landmarks' in f.keys():
        v_landmarks, v_labels, v_shape_types = [], [], []
        vertebrate_ids = list(f['v_landmarks']['shapes'].keys())
        for vertebrate_id in vertebrate_ids:
            landmarks = f['v_landmarks']['shapes'][vertebrate_id]['points'][:]
            label = f['v_landmarks']['shapes'][vertebrate_id].attrs['label']
            shape_types = f['v_landmarks']['shapes'][vertebrate_id].attrs['shape_type']
            
            v_landmarks.append(landmarks)
            v_labels.append(label)
            v_shape_types.append(shape_types)
    v_landmarks = dict(
        vertebrate_ids=vertebrate_ids, 
        v_landmarks=v_landmarks, 
        v_labels=v_labels, 
        v_shape_types=v_shape_types,
    )    
    # read facial annotations
    f_landmarks = {'f_landmarks': None}
    if 'f_landmarks' in f.keys():
        f_landmarks = {
            'f_landmarks': f['f_landmarks']['points'][:]
        }
    return image, v_landmarks, f_landmarks