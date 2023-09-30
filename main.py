""" The main function that is the interface to the package cvmt """
import pytorch_lightning as pl

pl.seed_everything(100, workers=True)

import argparse
import os
import sys

from easydict import EasyDict

import wandb
from cvmt.data import prep_all_datasets
from cvmt.ml import (train_test_split, trainer_edge_detection_single_task,
                     trainer_v_landmarks_single_task)
from cvmt.utils import (load_yaml_params, nested_dict_to_easydict,
                        remove_lightning_logs_dir)

STEPS = ["data_prep", "train_test_split", "train"]
TRAINING_TASKS = ["v_landmarks", "edges"]
CONFIG_PARAMS_PATH = "configs/params.yaml"


def main(
    params: EasyDict,
) -> None:
    # Parse command-line arguments
    args = parse_arguments()
    # Get the function name from command-line arguments
    step = args.step
    training_task = args.training_task
    checkpoint_path = args.checkpoint_path
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
    parser.add_argument('--checkpoint-path', type=str, help='checkpoint_path',)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # load params
    params: EasyDict = nested_dict_to_easydict(
        load_yaml_params(CONFIG_PARAMS_PATH)
    )
    # run the main
    main(params)
