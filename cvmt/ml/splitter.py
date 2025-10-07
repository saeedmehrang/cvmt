"""Train-val-test splitting functionality."""

from .utils import split_filenames, stratify_dataset_4
import numpy as np
import pandas as pd
import os
from easydict import EasyDict


def train_test_split(params: EasyDict) -> None:
    # read the HDF5 table using pandas HDFStore API that allows modifying the table
    # and writing back to the same file with less of boilerplate code.
    store = pd.HDFStore(
        os.path.join(params.PRIMARY_DATA_DIRECTORY, params.TRAIN.METADATA_TABLE_NAME),
        mode="a",
    )
    # read the table into a pandas DataFrame
    metadata_table = store.select("df")
    # add a grouping factor to the metadata table
    metadata_table["grouping_factor"] = metadata_table["dataset"]
    # stratify dataset 4 as it has age variable available
    if "dataset_4" in params.TRAIN.DATASETS_TO_INCLUDE:
        metadata_table = stratify_dataset_4(
            metadata_table=metadata_table,
            params=params,
        )
    # discard the samples that come with invalid or malformed data structure or
    # annotations as denoted by the column `valid_v_annots` in the metadata table.
    # for more information see the function `cvmt.data.prep.prep_all_datasets`.
    selected_samples = metadata_table.loc[
        (metadata_table["dataset"].isin(params.TRAIN.DATASETS_TO_INCLUDE))
        & (metadata_table["valid_v_annots"] == True),
        [
            "harmonized_id",
            "grouping_factor",
        ],
    ]
    # get the required inputs for the splitting function
    grouping_factor = selected_samples["grouping_factor"].tolist()
    filenames = selected_samples["harmonized_id"].tolist()
    # apply the splitting functionality to the filenames based on the grouping
    # defined by the dataset value
    train_files, val_files, test_files = split_filenames(
        filenames=filenames,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=100,
        grouping_factor=grouping_factor,
    )
    # find the training indices
    indices_train = np.where(
        np.isin(
            metadata_table["harmonized_id"].to_numpy(),
            train_files,
        )
    )[0]
    # find the validation indices
    indices_val = np.where(
        np.isin(
            metadata_table["harmonized_id"].to_numpy(),
            val_files,
        )
    )[0]
    # find the test indices
    indices_test = np.where(
        np.isin(
            metadata_table["harmonized_id"].to_numpy(),
            test_files,
        )
    )[0]
    # create an array that holds the split values
    split_arr = np.repeat(
        "undefined",
        metadata_table.shape[0],
    )
    split_arr[indices_train] = np.repeat("train", indices_train.shape)
    split_arr[indices_val] = np.repeat("val", indices_val.shape)
    split_arr[indices_test] = np.repeat("test", indices_test.shape)
    # place the split array into the dataframe
    metadata_table["split"] = split_arr

    # log the distribution of samples in each split based on dataset
    # TODO! uncomment and log the following line!
    # metadata_table.loc[:,['split', 'dataset']].value_counts()

    # write the new column to the metadata table
    metadata_table.reset_index(drop=True, inplace=True)
    # write the modified DataFrame back to the HDF5 file
    store.put("df", metadata_table, data_columns=True)
    store.close()
    return None
