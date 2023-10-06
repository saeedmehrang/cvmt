""" The main function that is the interface to the package cvmt """
import pytorch_lightning as pl

pl.seed_everything(100, workers=True)

import argparse
import os
import sys
from typing import *

import wandb
from cvmt.data import prep_all_datasets
from cvmt.ml import (train_test_split, trainer_edge_detection_single_task,
                     trainer_v_landmarks_single_task)
from cvmt.utils import (load_yaml_params, nested_dict_to_easydict,
                        remove_lightning_logs_dir)
from easydict import EasyDict

STEPS = ["data_prep", "train_test_split", "train"]
TRAINING_TASKS = ["v_landmarks", "edges"]
CONFIG_PARAMS_PATH = "configs/params.yaml"


def main(
    params: EasyDict,
    step: str,
    training_task: str = 'v_landmarks',
    checkpoint_path: Union[None, str] = None,
) -> None:
    """Main function to interact with cvmt library. 
    
    Args:
        params: An EasyDict of all the parameters needed to interact with the library. See `configs/params.yaml` for more info.
        step: An string for the name of the step to run. Options are ["data_prep", "train_test_split", "train"].
        training_task: An string for the name of the training task to run. Options are ["v_landmarks", "edges"].
        checkpoint_path: An string for tha path to a model checkpoint to be used as a pretrained model or continuing the training.

    Returns:
        None
    """

    # setup wandb
    try:
        # make sure not to re-use an old run
        wandb.finish()
        # login to the wandb sever and initialize
        wandb.login()
        config = dict(params)
        run = wandb.init(**params.WANDB.INIT, config=config)
        # log code
        wandb.run.log_code(".")
    except Exception as e:
        print(e)
        print(
            "Make sure to export your private `wandb_api_key` into your terminal."
            "Alternatively, follow the instructions in the README for the creation of a `.env` file in your configs directory."
        )
    # Execute the selected function
    if step == STEPS[0]:
        print(f"** Running {step}")
        prep_all_datasets(params)
    elif step == STEPS[1]:
        print(f"** Running {step}")
        train_test_split(params)
    elif step == STEPS[2]:
        print(f"** Running {step}")
        if training_task == TRAINING_TASKS[0]:
            trainer_v_landmarks_single_task(params, checkpoint_path=checkpoint_path)
        elif training_task == TRAINING_TASKS[1]:
            trainer_edge_detection_single_task(params,)
    elif (step not in STEPS) and (step is not None):
        print(f"Unknown function: {step}")
        sys.exit(1)
    else:
        print("****** Running prep_all_datasets ****** ")
        prep_all_datasets(params)
        print("****** Running train_test_split ****** ")
        train_test_split(params)
        print("****** Running trainer_v_landmarks_single_task without pretraining! ****** ")
        trainer_v_landmarks_single_task(params)
    # TODO! remove the unnecessary lightning_logs directory!
    # remove_lightning_logs_dir()
    return None


def parse_arguments():
    parser = argparse.ArgumentParser(description='Read command line arguments.')
    parser.add_argument('--step', type=str, help='pipeline step',)
    parser.add_argument('--training-task', type=str, help='training_task', default='v_landmarks')
    parser.add_argument('--checkpoint-path', type=str, help='checkpoint_path', default=None)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # load params
    default_params: EasyDict = nested_dict_to_easydict(
        load_yaml_params(CONFIG_PARAMS_PATH)
    )
    # Parse command-line arguments
    args = vars(parse_arguments())
    # run the main
    main(params=default_params, **args)
