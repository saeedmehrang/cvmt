""" The main function that is the interface to the package cvmt """
import pytorch_lightning as pl

pl.seed_everything(100, workers=True)

import os
import sys

from easydict import EasyDict

import wandb
from cvmt.data import prep_all_datasets
from cvmt.ml import train_test_split, trainer_v_landmarks_single_task
from cvmt.utils import (load_yaml_params, nested_dict_to_easydict,
                        remove_lightning_logs_dir)

FUNCTION_NAMES = ["data_prep", "train_test_split", "train"]
CONFIG_PARAMS_PATH = "configs/params.yaml"


def main(
    params: EasyDict,
) -> None:
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python main.py [function_name]")
        print(
            "Running all the steps end-to-end. The steps are as follows,"
            "- data_prep"
            "- train"
        )
        function_name = None
    else:
        # Get the function name from command-line arguments
        function_name = sys.argv[1]
    # setup wandb
    try:
        # make sure not to re-use an old run
        wandb.finish()
        # login to the wandb sever and initialize
        wandb.login()
        config = dict(params)
        run = wandb.init(**params.WANDB.INIT, config=config)
    except Exception as e:
        print(e)
        print(
            "Make sure to export your private `wandb_api_key` into your terminal."
            "Alternatively, follow the instructions in the README for the creation of a `.env` file in your configs directory."
        )
    # Execute the selected function
    if function_name == FUNCTION_NAMES[0]:
        print(f"** Running {function_name}")
        prep_all_datasets(params)
    elif function_name == FUNCTION_NAMES[1]:
        pass
        print(f"** Running {function_name}")
        train_test_split(params)
    elif function_name == FUNCTION_NAMES[2]:
        pass
        print(f"** Running {function_name}")
        trainer_v_landmarks_single_task(params)
    elif (function_name not in FUNCTION_NAMES) and (function_name is not None):
        print(f"Unknown function: {function_name}")
        sys.exit(1)
    else:
        print("****** Running prep_all_datasets ****** ")
        prep_all_datasets(params)
        print("****** Running train_test_split ****** ")
        train_test_split(params)
        print("****** Running trainer_v_landmarks_single_task ****** ")
        trainer_v_landmarks_single_task(params)
    # TODO! remove the unnecessary lightning_logs directory!
    # remove_lightning_logs_dir()
    return None


if __name__ == "__main__":
    # load params
    params: EasyDict = nested_dict_to_easydict(
        load_yaml_params(CONFIG_PARAMS_PATH)
    )
    # run the main
    main(params)
