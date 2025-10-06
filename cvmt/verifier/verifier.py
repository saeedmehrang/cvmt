"""Tools and functionalities for the verification of the model performance on either of the input
data splits, i.e. val or test.
"""

import os

import random
import wandb
from easydict import EasyDict
from cvmt.ml.trainer import create_dataloader, mean_radial_error, max_indices_4d_tensor

from cvmt.ml.utils import download_wandb_model_checkpoint
from cvmt.inference.inference import load_pretrained_model_eval_mode

import torch
from typing import *
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D


def verify_model_perf(params: Union[EasyDict, Dict], split: str = "val") -> None:
    """Run the `validation set` data through the trained model and gather the performance
    metrics, figures, and sample graphs for visualization. This step is done prior to finalizing
    the training and before going to `testing` with the `test set`. Be aware that testing is only
    done once we are happy with the training and the verfication. Do testing as infreqeuntly as
    possible to reduce the chance of introducing bias or unconsciously tuning on test set.
    """
    # download the model
    checkpoint_path, model_id = download_wandb_model_checkpoint(
        wandb_checkpoint_uri=params.VERIFY.WANDB_CHECKPOINT_REFERENCE_NAME
    )
    print("Model checkpoint downloaded here: ", checkpoint_path)
    # create data loader
    use_pretrain = True
    task_config = params.TRAIN.V_LANDMARK_TASK
    task_id = task_config.TASK_ID
    loss_name = params.TRAIN.LOSS_NAME
    model_params = params.MODEL.PARAMS
    n_images_to_plot = params.VERIFY.N_IMGAES_TO_PLOT
    # instantiate dataloader
    dataloader = create_dataloader(
        task_id=task_id,
        batch_size=1,
        split=split,
        shuffle=False,
        params=params,
        sampler_n_samples=None,
    )
    # load the model
    model, device = load_pretrained_model_eval_mode(
        model_params=model_params,
        use_pretrain=use_pretrain,
        checkpoint_path=checkpoint_path,
        task_id=task_id,
        loss_name=loss_name,
    )
    # collect predictions and sample images
    radial_errors, sample_images = collect_predictions_and_sample_graphs(
        data_loader=dataloader,
        device=device,
        task_id=task_id,
        model=model,
        n_images_to_plot=n_images_to_plot,
    )
    # compute statistics of mre
    med_mre, perc_25_mre, perc_75_mre = (
        np.median(radial_errors),
        np.percentile(radial_errors, 25),
        np.percentile(radial_errors, 75),
    )
    print("25th percentile of mre: ", perc_25_mre)
    print("50th percentile (median) of mre: ", med_mre)
    print("75th percentile of mre: ", perc_75_mre)
    mean_mre, std_mre = np.mean(radial_errors), np.std(radial_errors)
    print("Arithmetic mean of mre: ", mean_mre)
    print("Standard deviation of mre: ", std_mre)
    # plot the histogram
    plot_histogram(
        array=radial_errors,
        model_id=model_id,
        split=split,
        save_fig=True,
        log_fig=True,
    )
    # plot sample graphs
    plot_images_and_landmark_coords(
        sample_images,
        model_id=model_id,
        split=split,
        category="all",
        save_fig=True,
        log_fig=True,
        fig_name="random_samples",
    )
    return None


def plot_histogram(
    array: Union[np.ndarray, List[float]],
    model_id: str,
    split: str = "val",
    save_fig: bool = False,
    log_fig: bool = False,
) -> None:
    num_bins = 25
    fig, ax = plt.subplots()
    # the histogram of the data
    n, bins, _ = ax.hist(
        array,
        num_bins,
        density=False,
    )
    ax.set_xlabel("Mean radial error")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of mean radial error")
    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.show()
    file_name = f"mean_radial_error_hist_{split}_set_{model_id}"
    if save_fig:
        fig.savefig(
            os.path.join("artifacts/verification", file_name + ".jpg"),
            dpi=300,
        )
    if log_fig:
        wandb.log({file_name: wandb.Image(fig)})
    return None


def collect_predictions_and_sample_graphs(
    data_loader: torch.utils.data.DataLoader,
    device: str,
    task_id: int,
    model: torch.nn.Module,
    n_images_to_plot: int,
):
    radial_errors = []
    sample_images = []
    c = 0
    for i, batch in enumerate(data_loader):
        images, targets = batch["image"], batch["v_landmarks"]
        images = images.to(device)
        targets = targets.to(device)
        # Pass images through the model
        with torch.no_grad():
            predictions = model(images, task_id=task_id)
        mre = mean_radial_error(preds=predictions, targets=targets)
        mre = mre.item()
        radial_errors.append(mre)
        # create a list for plotting
        if random.random() > 0.5 and c < n_images_to_plot:
            c += 1

            preds_coords = max_indices_4d_tensor(predictions)
            preds_coords = preds_coords.cpu().numpy()
            preds_coords = np.squeeze(preds_coords)

            targs_coords = max_indices_4d_tensor(targets)
            targs_coords = targs_coords.cpu().numpy()
            targs_coords = np.squeeze(targs_coords)

            landmarks_coords = {"preds": preds_coords, "targets": targs_coords}
            image = images[0].cpu().numpy()
            sample_images.append((image, landmarks_coords, mre))
    return radial_errors, sample_images


def plot_images_and_landmark_coords(
    items: List[Any],
    model_id: str,
    split: str,
    category: str = "all",
    save_fig: bool = False,
    log_fig: bool = False,
    fig_name: str = "my_figure",
):
    # if user desires a specific category
    if category == "all":
        categories = ["preds", "targets"]
    else:
        categories = [category]

    # Calculate the number of rows for subplots
    if len(items) > 1:
        rows = len(items) // 2
        fig, axs = plt.subplots(rows, 2, figsize=(16, 8 * rows))
        axs = axs.flatten()
    else:
        fig, axs = plt.subplots(1, 1, figsize=(16, 16))
        axs = [axs]

    for ax, item in zip(axs, items):
        image, landmarks, mre = item
        image = image.squeeze()
        target_landmarks = landmarks["targets"]
        pred_landmarks = landmarks["preds"]
        ax.imshow(
            image,
            cmap="gray",
        )
        if "targets" in categories:
            for landmark in target_landmarks:
                # Assuming each landmark is a tuple of (x, y) coordinates
                ax.add_patch(
                    patches.Circle((landmark[1], landmark[0]), radius=1, color="yellow")
                )

        if "preds" in categories:
            for i, landmark in enumerate(pred_landmarks):
                # Assuming each landmark is a tuple of (x, y) coordinates
                ax.add_patch(
                    patches.Circle((landmark[1], landmark[0]), radius=1, color="cyan")
                )
                ax.text(
                    landmark[1], landmark[0], str(i), color="orange"
                )  # Annotate the index

        ax.set_title(f"MRE={mre}")  # Set your title here

    # Create a legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Target Landmarks",
            markerfacecolor="yellow",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Predicted Landmarks",
            markerfacecolor="cyan",
            markersize=10,
        ),
    ]
    # Place the legend on the axes
    for ax in axs:
        ax.legend(handles=legend_elements, loc="lower right")
    plt.tight_layout()
    plt.suptitle(f"Model: {model_id}")
    plt.show()
    # save fig
    verify_dir = "artifacts/verification"
    os.makedirs(verify_dir, exist_ok=True)
    file_name = f"{fig_name}_{split}_{model_id}"
    if save_fig:
        fig.savefig(os.path.join(verify_dir, file_name + ".jpg"), dpi=300)
    if log_fig:
        wandb.log({file_name: wandb.Image(fig)})
    return None
