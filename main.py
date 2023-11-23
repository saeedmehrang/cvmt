""" The main function that is the interface to the package cvmt """
import pytorch_lightning as pl

pl.seed_everything(100, workers=True)

import git
import argparse
import os
import sys
from typing import *

import wandb
from cvmt.data import prep_all_datasets
from cvmt.ml import (train_test_split, trainer_edge_detection_single_task,
                     trainer_v_landmarks_single_task, tester_v_landmarks_single_task)
from cvmt.verifier import  verify_model_perf
from cvmt.utils import (load_yaml_params, nested_dict_to_easydict,
                        remove_lightning_logs_dir)
from cvmt.inference.inference import predict_image_cmd_interface
from easydict import EasyDict

STEPS = ["data_prep", "train_test_split", "train", "verify", "test", "inference"]
TRAINING_TASKS = ["v_landmarks", "edges",]
VERIFICATION_SPLIT = ["val", "test"]
CONFIG_PARAMS_PATH = "configs/params.yaml"


def main(
    params: EasyDict,
    step: str,
    training_task: str = 'v_landmarks',
    verify_split: str = 'val',
    filepath: str = '',
    pix2cm: float = 10.0,
) -> None:
    """Main function to interact with cvmt library. 
    
    Args:
        params: An EasyDict of all the parameters needed to interact with the library. See `configs/params.yaml` for more info.
        step: An string for the name of the step to run. Options are ["data_prep", "train_test_split", "train"].
        training_task: An string for the name of the training task to run. Options are ["v_landmarks", "edges"].
        verify_split: An string for the name of the data split to use for the input of verification. Options are ["val", "test"].

    Returns:
        None
    """

    # setup wandb
    try:
        config_wandb(params,)
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
            print(f"** Running training for {training_task}")
            trainer_v_landmarks_single_task(params,)
        elif training_task == TRAINING_TASKS[1]:
            print(f"** Running training for {training_task}")
            trainer_edge_detection_single_task(params,)
    elif step == STEPS[3]:
        print(f"** Running {step}")
        if verify_split == VERIFICATION_SPLIT[0]:
            print(f"** Running verification for {verify_split}")
            verify_model_perf(params, split=verify_split)
        elif verify_split == VERIFICATION_SPLIT[1]:
            print(f"** Running verification for {verify_split}")
            verify_model_perf(params, split=verify_split)
    elif step == STEPS[4]:
        print(f"** Running {step}")
        tester_v_landmarks_single_task(params,)
    elif step == STEPS[5]:
        print(f"** Running {step}")
        stage = predict_image_cmd_interface(params, filepath=filepath, px2cm_ratio=pix2cm) 
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
    parser.add_argument('--verify-split', type=str, help='verification input data split', default='val')
    parser.add_argument('--filepath', type=str, help='path to the image for inference',)
    parser.add_argument('--pix2cm', type=float, help='pixel to centimeter ratio as depiced on the image ruler',)
    args = parser.parse_args()
    return args


def get_git_info():
    repo = git.Repo(search_parent_directories=True)
    commit_hash = repo.head.object.hexsha
    branch = repo.active_branch.name
    return commit_hash, branch


def config_wandb(params,):
    # make sure not to re-use an old run
    wandb.finish()
    # login to the wandb sever and initialize
    wandb.login()
    config = dict(params)
    run = wandb.init(
        **params.WANDB.INIT,
        config=config,
    )
    # log code
    wandb.run.log_code(".")
    # log code commit hash and branch
    commit_hash, branch = get_git_info()
    run.summary['git_commit_hash'] = commit_hash
    run.summary['git_branch'] = branch


if __name__ == "__main__":
    # load params
    default_params: EasyDict = nested_dict_to_easydict(
        load_yaml_params(CONFIG_PARAMS_PATH)
    )
    # Parse command-line arguments
    args = vars(parse_arguments())
    # run the main
    main(params=default_params, **args)
