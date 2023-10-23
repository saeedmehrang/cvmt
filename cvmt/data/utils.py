import hashlib
import json
import os
from itertools import combinations
from numbers import Number
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_gradient_magnitude
from scipy.spatial.distance import pdist
from skimage.feature import canny

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
    image = convert_image_bw_no_channel(image)
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
            landmarks_list = [
                sort_landmarks(landmarks, shape_grp) 
                for shape_grp, landmarks in zip(shape_grps, landmarks_list)]
            for landmarks, shape in zip(landmarks_list, v_annots['shapes']):
                shape['points'] = landmarks
        except Exception as e:
            print(e)
            print("Error above encountered! V landmarks set to None.")
            v_annots = None
            return v_annots
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
    sigma: int = 1,
) -> Tuple[np.ndarray, np.ndarray, Union[Dict, None], np.ndarray]:
    image = load_image(
        image_filename=image_filename,
        image_dir=image_dir,
    )
    edges = compute_edges_ggm(
        image=image,
        sigma=sigma,
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
    return image, edges, v_annots, f_annots


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
    landmarks: List[float],
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


def sort_landmarks(
    landmarks: np.ndarray,
    shape_grp: str,
) -> np.ndarray:
    # vertebrae 2
    if shape_grp == '2':
        # sort by the width of the landmarks
        landmarks = landmarks[landmarks[:, 0].argsort()]
    elif shape_grp in ['3', '4']:
        # there are only 2 patterns in the order of 3rd and 4th vertebrae landmarks
        # if upper left corner is index 0 and we rotate counter clock-wise, patterns 
        # are A) 0 1 2 3 4 and B) 4 0 1 2 3
        # find the slope of the line connecting the first and one to the last landmarks
        # i.e. the line between 0 and 3
        diffs = landmarks[3] - landmarks[0]
        dx, dy = diffs[0], -diffs[1] # the negative sign is needed as y axis is reversed in images in python
        if dy/dx < 0:
            # pattern is A, no action needed
            pass
        elif dy/dx >= 0:
            # pattern is B, place first index at the last
            landmarks = np.roll(landmarks, 1, axis=0)
    else:
        raise ValueError(
            f"Unrecognized `shape_grp` value {shape_grp} in sort_landmarks!"
        )
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
        grp_shape.create_dataset('points', data=points, compression='gzip', compression_opts=9)        
    return True


def write_f_annots(
    f: h5py._hl.files.File,
    f_annots: Union[np.ndarray, None],
) -> bool:
    f_annots_data_grp = f.create_group("f_landmarks")
    points = np.array(f_annots).astype(int)
    f_annots_data_grp.create_dataset('points', data=points, compression='gzip', compression_opts=9)
    return True


def write_image(
    f: h5py._hl.files.File,
    image: np.ndarray,
) -> bool:
    image = image.astype(np.float16,)
    image_data_grp = f.create_group("image")
    image_data_grp.create_dataset('data', data=image, compression='gzip', compression_opts=9)
    return True


def save_image_and_annots_hdf5(
    save_dir: str,
    harmonized_id: str,
    image: np.ndarray,
    v_annots: Union[Dict, None],
    f_annots: Union[np.ndarray, None],
    edges: np.ndarray,
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
    f = h5py.File(filepath, 'w', )
    # write image data
    write_image(f, image)
    # write edges data
    write_edges(f, edges)
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
    hashcode = hashlib.blake2b(fname.encode('utf-8'), digest_size=10).hexdigest()
    return hashcode

    
def harmonize_hdf5(
    image_filename: str,
    image_dir: str,
    v_annot_dir: str,
    f_annot_dir: str,
    unwanted_json_fields: Any,
    primary_data_dir: Union[str, Path],
    dataset_name: str,
    dev_set: str,
    sigma: int = 1,
) -> Tuple[str, str, bool, bool]:
    image, edges, v_annots, f_annots = load_and_clean_image_and_annotations(
        image_filename=image_filename,
        image_dir=image_dir,
        v_annot_dir=v_annot_dir,
        f_annot_dir=f_annot_dir,
        unwanted_fields_v_annot=unwanted_json_fields,
        sigma=sigma,
    )
    harmonized_id = create_hash_code(
        image_dir=image_dir,
        image_filename=image_filename,
    )
    # check integrity of annotations
    invalid_v_annots, v_annots, v_annots_shapes = check_v_annots_integrity(v_annots)
    invalid_f_annots, f_annots, f_annots_shapes = check_f_annots_integrity(f_annots)
    # write the data
    save_image_and_annots_hdf5(
        save_dir=primary_data_dir,
        harmonized_id=harmonized_id,
        image=image,
        v_annots=v_annots,
        f_annots=f_annots,
        edges=edges,
    )
    # create metadata and store into a dict
    edges_present = True if edges is not None else False
    metadata = {
        'valid_v_annots': not(invalid_v_annots),
        'valid_f_annots': not(invalid_f_annots),
        'edges_present': edges_present,
        'harmonized_id': harmonized_id,
        'source_image_filename': image_filename,
        'dataset': dataset_name,
        'dev_set': dev_set,
    }
    # merge the two dicts
    metadata.update(v_annots_shapes)
    metadata.update(f_annots_shapes)
    return metadata


def check_v_annots_integrity(v_annots: Dict[str, Any]) -> Tuple[bool, Union[None, Dict[str, Any]], Dict[str, Any]]:
    invalid = False
    v_annots_present = True if v_annots is not None else False
    v_annots_shapes = {
        'v_annots_2_rows': None,
        'v_annots_2_cols': None,
        'v_annots_3_rows': None,
        'v_annots_3_cols': None,
        'v_annots_4_rows': None,
        'v_annots_4_cols': None,
    }
    # return early if not present
    if not v_annots_present:
        # return the dict with invaid True and v_annots None
        invalid = True
        return invalid, v_annots, v_annots_shapes
    # Next, check the shapes
    try:
        if len(v_annots['shapes']) != 3:
            raise ValueError(f"There are {len(v_annots['shapes'])} shapes in the v_annots!")
    except Exception as e:
        print(e)
        invalid = True
        v_annots = None
        # return the dict with invaid True and v_annots None
        return invalid, v_annots, v_annots_shapes
    # next, check each gorup shapes
    for shape in v_annots['shapes']:
        group_id = str(shape['group_id'])
        shape = np.array(shape['points']).shape
        try:
            if group_id not in ['2','3','4']:
                raise ValueError("The group_id of the v_annots is incorrect!")

            v_annots_shapes[f"v_annots_{group_id}_rows"] = shape[0]
            v_annots_shapes[f"v_annots_{group_id}_cols"] = shape[1]

        except Exception as e:
            print(e)
            invalid = True
            v_annots = None
            # return the dict with invaid True and v_annots None
            return invalid, v_annots, v_annots_shapes
    return invalid, v_annots, v_annots_shapes


def check_f_annots_integrity(f_annots: np.ndarray,) -> Tuple[bool, Union[None, Dict[str, Any]], Dict[str, Any]]:
    invalid = False
    f_annots_shapes = {
        'f_annots_rows': None,
        'f_annots_cols': None,
    }
    f_annots_present = True if f_annots is not None else False
    if f_annots_present:
        try:
            f_annots_rows, f_annots_cols = f_annots.shape
            f_annots_shapes['f_annots_rows'] = f_annots_rows
            f_annots_shapes['f_annots_cols'] = f_annots_cols
        except Exception as e:
            print(e)
            invalid = True
            f_annots = None
            return invalid, f_annots, f_annots_shapes
    else:
        invalid = True
    return invalid, f_annots, f_annots_shapes
        

def read_harmonized_hdf5(
    h5py_filename: str,
    primary_data_dir: Union[str, Path],
):
    h5py_file = os.path.join(primary_data_dir, h5py_filename)
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
    # read edges
    edges = f['edges']['data'][:]
    return image, edges, v_landmarks, f_landmarks


def compute_edges_canny(
    image: np.ndarray,
    sigma: int = 1,
) -> np.ndarray:
    edges = canny(image, sigma=sigma)
    return edges


def compute_edges_ggm(
    image: np.ndarray,
    sigma: int = 1,
) -> np.ndarray:
    # normalize the image
    image = image / 255.0
    # compute edges
    edges = gaussian_gradient_magnitude(image, sigma=sigma)
    return edges


def write_edges(
    f: h5py._hl.files.File,
    edges: np.ndarray,
) -> bool:
    edges = edges.astype(np.float16,)
    edges_data_grp = f.create_group("edges")
    edges_data_grp.create_dataset('data', data=edges, compression='gzip', compression_opts=9)
    return True


def convert_image_bw_no_channel(image):
    if len(image.shape) == 2:
        # image is black and white, leave it as it is
        return image
    elif len(image.shape) == 3 and image.shape[2] == 1:
        # image has a single channel, already black and white, remove channel axis
        return image[:, :, 0]
    elif len(image.shape) == 3 and image.shape[0] == 1:
        # image has a single channel, already black and white, remove channel axis
        return image[0, :, :]
    elif len(image.shape) == 3 and image.shape[2] > 1:
        # image has multiple channels, convert to black and white
        image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        return image
    else:
        raise ValueError("Invalid image shape")
