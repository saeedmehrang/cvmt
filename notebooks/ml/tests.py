import h5py
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

from utils import create_dataloader, nested_dict_to_easydict




if __name__ == "__main__":

    with open("../../code_configs/params.yaml") as f:
        PARAMS = yaml.safe_load(f)
        PARAMS = nested_dict_to_easydict(PARAMS)


    task_config = PARAMS.TRAIN.SINGLE_TASK
    task_id = task_config.TASK_ID
    batch_size = task_config.BATCH_SIZE

    train_dataloader = create_dataloader(
        task_id=task_id,
        batch_size=1,
        split='train',
        shuffle=False,
    )

    for i_batch, sample_batched in enumerate(train_dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['v_landmarks'].size())
