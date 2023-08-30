"""Python functions for data preparation. 

The first version of the data (raw zone) was in a format that needed to be changed manually. 

Then, after the manual corrections on raw zone, the intermediate zone data were created.

Finally, the primary zone data are created programmatically from the intermediate zone.
"""

import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import ray
from easydict import EasyDict
from ray.util.multiprocessing import Pool

from .utils import harmonize_hdf5


class PrepDataset:
    """ Prepare data functionality"""
    def __init__(
        self,
        interm_data_dir: Union[str, Path],
        img_dir_name_dset: str,
        primary_data_dir: Union[str, Path],
        interm_data_dir_name: str,
        interm_v_lmks_dir_name: str,
        interm_f_lmks_dir_name: str,
        unwanted_json_fields: Any,
        edge_sigma: float,
        dataset_name: str = "dataset_1",
        n_processes: int = -1,
    ) -> None:
        """
        
        Args:
            interm_data_dir: PARAMS.INTERMEDIATE_DATA_DIRECTORY
            img_dir_name_dset: PARAMS.DATASET_1_INTERM_IMG_DIR_NAME,
            interm_data_dir_name: PARAMS.DATASET_1_INTERM_DIR_NAME,
            interm_v_lmks_dir_name: PARAMS.DATASET_1_INTERM_V_LANDMARKS_DIR_NAME,
            interm_f_lmks_dir_name: PARAMS.DATASET_1_INTERM_F_LANDMARKS_DIR_NAME,
            edge_sigma: DATASET_1_EDGE_DETECT_SIGMA,
            dataset: dataset name,
        """
        # start the data processing
        self.primary_data_dir = primary_data_dir
        self.unwanted_json_fields = unwanted_json_fields
        self.interm_data_dir = interm_data_dir
        self.img_dir_name_dset = img_dir_name_dset
        self.interm_data_dir_name = interm_data_dir_name
        self.interm_v_lmks_dir_name = interm_v_lmks_dir_name
        self.interm_f_lmks_dir_name = interm_f_lmks_dir_name
        self.edge_sigma = edge_sigma
        self.dataset_name = dataset_name

        # initialize Ray
        ray.init()
        ray_cpus = int(ray._private.state.cluster_resources()["CPU"])
        if n_processes == -1:
            n_processes = ray_cpus
        if n_processes > ray_cpus:
            n_processes = ray_cpus
        self.pool = Pool(processes=n_processes)

    def __call__(self,) -> pd.DataFrame:
        metadata = []
        # construct the path variables
        for i in range(len(self.img_dir_name_dset)):
            # create the directory paths
            img_foldername = self.img_dir_name_dset[i]
            self.image_dir = os.path.join(
                self.interm_data_dir,
                self.interm_data_dir_name,
                img_foldername,
            )
            v_landmarks_foldername = self.interm_v_lmks_dir_name[i]
            self.v_landmarks_dir = os.path.join(
                self.interm_data_dir,
                self.interm_data_dir_name,
                v_landmarks_foldername,
            )
            self.f_landmarks_dir = None
            if self.interm_f_lmks_dir_name is not None:
                self.f_landmarks_dir = os.path.join(
                    self.interm_data_dir,
                    self.interm_data_dir_name,
                    self.interm_f_lmks_dir_name,
                )
            # parse the directory
            self.image_filenames = os.listdir(self.image_dir)
            
            # harmonize all the images and annotations
            if len(self.img_dir_name_dset) > 1:
                self.dev_set = img_foldername.split('/')[0]
            else:
                self.dev_set = None
            # initialize the parallel function
            harmonize_hdf5_ = partial(
                harmonize_hdf5,
                image_dir=self.image_dir,
                v_annot_dir=self.v_landmarks_dir,
                f_annot_dir=self.f_landmarks_dir,
                sigma=self.edge_sigma,
                primary_data_dir=self.primary_data_dir,
                unwanted_json_fields=self.unwanted_json_fields,
                dataset_name=self.dataset_name,
                dev_set=self.dev_set,
            )
            # Traverse the images and harmonize them one by one
            for record_metadata in self.pool.map(harmonize_hdf5_, self.image_filenames):
                metadata.append(record_metadata)

        # create a metadata pandas dataframe
        metadata = pd.DataFrame(metadata)
        metadata.to_hdf(
            os.path.join(self.primary_data_dir, f'metadata_{self.dataset_name}.hdf5'),
            key='df',
            index=False,
            mode='w',
            append=True,
            format='table',
        )
        # shutdown ray
        ray.shutdown()
        return metadata


def prep_all_datasets(params: EasyDict):
    """Run all datasets' preprocessing. The input data zone is 
    `Intermediate` and the output zone is `Primary`.
    """
    # dataset 1 initialize
    print("**** dataset_1_prep")
    dataset_1_prep = PrepDataset(
        interm_data_dir=params.INTERMEDIATE_DATA_DIRECTORY,
        primary_data_dir=params.PRIMARY_DATA_DIRECTORY,
        img_dir_name_dset=params.INTERM.DATASET_1.IMG_DIR_NAME,
        interm_data_dir_name=params.INTERM.DATASET_1.DIR_NAME,
        interm_v_lmks_dir_name=params.INTERM.DATASET_1.V_LANDMARKS_DIR_NAME,
        interm_f_lmks_dir_name=params.INTERM.DATASET_1.F_LANDMARKS_DIR_NAME,
        unwanted_json_fields=params.INTERM.UNWANTED_JSON_FIELDS,
        edge_sigma=params.PRIMARY.DATASET_1.EDGE_DETECT_SIGMA,
        dataset_name="dataset_1",
        n_processes=params.INTERM.N_PROCESSES,
    )
    # call
    dataset_1_prep()
    # dataset 2 initiaize
    print("**** dataset_2_prep")
    dataset_2_prep = PrepDataset(
        interm_data_dir=params.INTERMEDIATE_DATA_DIRECTORY,
        primary_data_dir=params.PRIMARY_DATA_DIRECTORY,
        img_dir_name_dset=params.INTERM.DATASET_2.IMG_DIR_NAME,
        interm_data_dir_name=params.INTERM.DATASET_2.DIR_NAME,
        interm_v_lmks_dir_name=params.INTERM.DATASET_2.V_LANDMARKS_DIR_NAME,
        interm_f_lmks_dir_name=params.INTERM.DATASET_2.F_LANDMARKS_DIR_NAME,
        unwanted_json_fields=params.INTERM.UNWANTED_JSON_FIELDS,
        edge_sigma=params.PRIMARY.DATASET_2.EDGE_DETECT_SIGMA,
        dataset_name="dataset_2",
        n_processes=params.INTERM.N_PROCESSES,
    )
    # call
    dataset_2_prep()
    # dataset 3 initiaize
    print("**** dataset_3_prep")
    dataset_3_prep = PrepDataset(
        interm_data_dir=params.INTERMEDIATE_DATA_DIRECTORY,
        primary_data_dir=params.PRIMARY_DATA_DIRECTORY,
        img_dir_name_dset=params.INTERM.DATASET_3.IMG_DIR_NAME,
        interm_data_dir_name=params.INTERM.DATASET_3.DIR_NAME,
        interm_v_lmks_dir_name=params.INTERM.DATASET_3.V_LANDMARKS_DIR_NAME,
        interm_f_lmks_dir_name=params.INTERM.DATASET_3.F_LANDMARKS_DIR_NAME,
        unwanted_json_fields=params.INTERM.UNWANTED_JSON_FIELDS,
        edge_sigma=params.PRIMARY.DATASET_3.EDGE_DETECT_SIGMA,
        dataset_name="dataset_3",
        n_processes=params.INTERM.N_PROCESSES,
    )
    # call
    dataset_3_prep()
    # dataset 4 initiaize
    print("**** dataset_4_prep")
    dataset_4_prep = PrepDataset(
        interm_data_dir=params.INTERMEDIATE_DATA_DIRECTORY,
        primary_data_dir=params.PRIMARY_DATA_DIRECTORY,
        img_dir_name_dset=params.INTERM.DATASET_4.IMG_DIR_NAME,
        interm_data_dir_name=params.INTERM.DATASET_4.DIR_NAME,
        interm_v_lmks_dir_name=params.INTERM.DATASET_4.V_LANDMARKS_DIR_NAME,
        interm_f_lmks_dir_name=params.INTERM.DATASET_4.F_LANDMARKS_DIR_NAME,
        unwanted_json_fields=params.INTERM.UNWANTED_JSON_FIELDS,
        edge_sigma=params.PRIMARY.DATASET_4.EDGE_DETECT_SIGMA,
        dataset_name="dataset_4",
        n_processes=params.INTERM.N_PROCESSES,
    )
    # call
    dataset_4_prep()
    # read the created metadata tables and concatenate them
    merge_metadata_tables(params=params)


def merge_metadata_tables(params: EasyDict):
    """Read the metadata tables of the 3 input datasets and merge them into one."""
    metadata_table_1 = pd.read_hdf(
        os.path.join(params.PRIMARY_DATA_DIRECTORY, 'metadata_dataset_1.hdf5'),
        key='df',
    )
    metadata_table_2 = pd.read_hdf(
        os.path.join(params.PRIMARY_DATA_DIRECTORY, 'metadata_dataset_2.hdf5'),
        key='df',
    )
    metadata_table_3 = pd.read_hdf(
        os.path.join(params.PRIMARY_DATA_DIRECTORY, 'metadata_dataset_3.hdf5'),
        key='df',
    )
    metadata_table_4 = pd.read_hdf(
        os.path.join(params.PRIMARY_DATA_DIRECTORY, 'metadata_dataset_4.hdf5'),
        key='df',
    )
    metadata_table_all = pd.concat(
        [metadata_table_1, metadata_table_2, metadata_table_3, metadata_table_4],
        axis=0,
    )
    metadata_table_all.reset_index(drop=True, inplace=True)
    # add validty column
    metadata_table_all = create_validity_column(metadata_table=metadata_table_all)
    # write to disk
    metadata_table_all.to_hdf(
        os.path.join(params.PRIMARY_DATA_DIRECTORY, 'metadata.hdf5'),
        key='df',
        index=False,
        mode='w',
    )


def create_validity_column(
    metadata_table: pd.DataFrame,
) -> pd.DataFrame:
    validty_arr = np.repeat(True, metadata_table.shape[0],)
    invalid_rows = metadata_table[
        (
            (metadata_table['v_annots_present'] == True) & (
            (metadata_table['v_annots_2_rows'] != 3) | 
            (metadata_table['v_annots_3_rows'] != 5) | 
            (metadata_table['v_annots_4_rows'] != 5))
        ) | (
            (metadata_table['f_annots_present'] == True) & (
            metadata_table['f_annots_rows'] != 19)
        )
    ]

    invalid_indices = invalid_rows.index.to_numpy()
    validty_arr[invalid_indices] = False
    metadata_table['valid'] = validty_arr
    return metadata_table
