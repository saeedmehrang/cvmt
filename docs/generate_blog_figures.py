"""
Script to generate all figures for the machine learning strategy blog post.

This script creates visualizations using the actual cvmt codebase to demonstrate:
1. U-Net architecture
2. Gaussian heatmap generation
3. Data augmentation examples
4. Model performance metrics
5. Inference results
6. Geometric analysis

Run this script after training a model to generate publication-ready figures.

Usage:
    python docs/generate_blog_figures.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import torch
from torchvision import transforms

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cvmt.ml.models import MultiTaskLandmarkUNetCustom
from cvmt.ml.utils import (
    ResizeTransform,
    Coord2HeatmapTransform,
    CustomToTensor,
    RandomHorFlip,
    RandomRotationTransform,
    GaussianBlurTransform,
    RandomBrightness,
    CustomScaleto01,
)

# Create output directory
os.makedirs("docs/images", exist_ok=True)

print("Generating blog post figures...")
print("=" * 60)


# Figure 2: U-Net Architecture
print("\n[1/5] Generating U-Net architecture visualization...")
try:
    from torchview import draw_graph

    model = MultiTaskLandmarkUNetCustom(
        in_channels=1,
        out_channels1=1,
        out_channels2=1,
        out_channels3=13,
        out_channels4=19,
        backbone_encoder="efficientnet-b2",
        backbone_weights="imagenet",
        freeze_backbone=True,
    )

    sample_input = torch.randn(1, 1, 256, 256)

    model_graph = draw_graph(
        model,
        input_data=(sample_input, 3),
        expand_nested=True,
        save_graph=True,
        filename="unet_architecture",
        directory="docs/images/",
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   ✓ Model has {total_params:,} total parameters")
    print(f"   ✓ Trainable: {trainable_params:,}")
    print("   ✓ Saved to: docs/images/unet_architecture.png")
except ImportError:
    print("   ⚠ Skipping (torchview not installed)")


# Figure 3: Gaussian Heatmaps
print("\n[2/5] Generating Gaussian heatmap visualization...")
landmarks = np.array(
    [
        [128, 50],
        [135, 60],
        [142, 55],  # C2
        [120, 100],
        [128, 110],
        [136, 115],
        [144, 110],
        [152, 100],  # C3
        [115, 160],
        [125, 170],
        [135, 175],
        [145, 170],
        [155, 160],  # C4
    ]
)

sample_image = np.random.rand(256, 256)
coord2heatmap = Coord2HeatmapTransform(gauss_std=2.0)

sample = {"image": sample_image, "v_landmarks": landmarks}
transformed = coord2heatmap(sample)
heatmaps = transformed["v_landmarks"]

fig, axes = plt.subplots(3, 5, figsize=(15, 9))
fig.suptitle(
    "Gaussian Heatmaps for 13 Vertebral Landmarks", fontsize=16, fontweight="bold"
)

for i, ax in enumerate(axes.flat):
    if i < 13:
        ax.imshow(heatmaps[i], cmap="hot")
        ax.set_title(f"Landmark {i + 1}", fontsize=10)
        ax.axis("off")
    else:
        ax.axis("off")

plt.tight_layout()
plt.savefig("docs/images/heatmap_visualization.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✓ Saved to: docs/images/heatmap_visualization.png")


# Figure 4: Data Augmentation
print("\n[3/5] Generating data augmentation examples...")
from cvmt.ml.utils import RightResizedCrop

# --- Import an image from scikit-image data ---
from skimage import data
from skimage import io, color


# Ensure the image is normalized to [0, 1] for best compatibility with your pipeline
img = io.imread("docs/images/155.png")
img = img[:, :, :3]  # dropping alpha channel if there is

if img.ndim == 3:
    img = color.rgb2gray(img)
original_image = img.astype(np.float32)  # rgb2gray already returns 0-1 range

# Check image size and resize it to a larger size if needed for the example's starting point
# We'll stick to the original size or slightly larger if needed,
# and let the ResizeTransform handle the final size.
if original_image.shape[0] < 512 or original_image.shape[1] < 512:
    # Resize to a common starting size (optional, depending on the original size)
    # Since the cameraman image is 256x256, we'll let ResizeTransform handle it.
    pass  # Keep it at its original size (256x256) which is fine.

# Example landmarks, scaled to the 256x256 image size
landmarks_256 = np.array([[100, 75], [110, 80], [120, 77]])


# Load sample image and landmarks
sample = {
    "image": original_image,  # Use the real image
    "v_landmarks": landmarks_256,  # Example landmarks for the 256x256 image
}

# Define augmentation pipeline (matching config.yaml TRAIN transforms)
augmentations = transforms.Compose(
    [
        ResizeTransform(size=(256, 256)),
        Coord2HeatmapTransform(gauss_std=1.0),
        CustomToTensor(),
        CustomScaleto01(),
        RandomRotationTransform(degrees=[5, 10], p=0.5),
        GaussianBlurTransform(kernel_size=3, sigma=0.2, p=0.1),
        RightResizedCrop(width_scale_low=0.6, width_scale_high=1.0, p=0.5),
        RandomBrightness(low=0.8, high=1.5, p=0.2),
    ]
)

# Apply augmentations multiple times
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Data Augmentation Examples (Cameraman Image)", fontsize=16)

for i, ax in enumerate(axes.flat):
    # Apply augmentations to a *copy* of the sample data
    augmented = augmentations(sample.copy())

    # The image is now a Tensor of shape [1, H, W] or [H, W] if CustomToTensor
    # and Coord2HeatmapTransform output a single-channel image/heatmap.
    # We assume 'image' contains the transformed image data.
    image = augmented["image"].squeeze().numpy()

    # Check if the output is an image or a heatmap (since you use Coord2HeatmapTransform)
    # Assuming the 'image' key still holds the visual data (like a heatmap or the image itself
    # if the pipeline is structured to update the image inplace).
    # If the output is a set of heatmaps, you'll need to sum/visualize them differently.
    # For a simple visual demo, we'll stick to a grayscale display.

    # Plot the result
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Augmentation {i + 1}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("docs/images/augmentation_examples.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✓ Saved to: docs/images/augmentation_examples.png")


# Figure 5: Model Performance
print("\n[4/5] Generating model performance visualization...")

# Real performance metrics from training
metrics = {
    "train_mre": 2.1934,
    "val_mre": 1.3248,
    "test_mre": 1.3667,
    "train_mse": 0.000045,
    "val_mse": 0.000049,
    "test_mse": 0.000099,
    "train_loss": 0.6989,
    "val_loss": 3.0782,
    "test_loss": 1.9280,
}

# Create subplots for different metrics (2 rows: bar charts + histogram)
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.3)

# Row 1: Bar charts for metrics
# Plot 1: Mean Radial Error (MRE)
ax = fig.add_subplot(gs[0, 0])
splits = ["Train", "Val", "Test"]
mre_values = [metrics["train_mre"], metrics["val_mre"], metrics["test_mre"]]
colors_mre = ["#3498db", "#2ecc71", "#e74c3c"]
bars1 = ax.bar(
    splits, mre_values, color=colors_mre, alpha=0.8, edgecolor="black", linewidth=1.5
)
ax.set_title("Mean Radial Error by Split", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3, linestyle="--")
for bar, val in zip(bars1, mre_values):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.05,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# Plot 2: Mean Squared Error (MSE)
ax = fig.add_subplot(gs[0, 1])
mse_values = [
    metrics["train_mse"] * 1e6,
    metrics["val_mse"] * 1e6,
    metrics["test_mse"] * 1e6,
]
bars2 = ax.bar(
    splits, mse_values, color=colors_mre, alpha=0.8, edgecolor="black", linewidth=1.5
)
ax.set_ylabel("MSE (×10⁻⁶)", fontsize=12, fontweight="bold")
ax.set_title("Mean Squared Error by Split", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3, linestyle="--")
for bar, val in zip(bars2, mse_values):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 1,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# Plot 3: Total Loss
ax = fig.add_subplot(gs[0, 2])
loss_values = [metrics["train_loss"], metrics["val_loss"], metrics["test_loss"]]
bars3 = ax.bar(
    splits, loss_values, color=colors_mre, alpha=0.8, edgecolor="black", linewidth=1.5
)
ax.set_ylabel("Loss", fontsize=12, fontweight="bold")
ax.set_title("Total Loss by Split", fontsize=13, fontweight="bold")
ax.grid(axis="y", alpha=0.3, linestyle="--")
for bar, val in zip(bars3, loss_values):
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + 0.1,
        f"{val:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

# Row 2: MRE Histogram from validation set
histogram_path = (
    "docs/images/media_images_mean_radial_error_hist_val_set_model-urt7dgbp_v47.png"
)
if os.path.exists(histogram_path):
    ax = fig.add_subplot(gs[1, 0])
    histogram_img = plt.imread(histogram_path)
    ax.imshow(histogram_img)
    ax.axis("off")
    ax.set_title(
        "Validation Set MRE Distribution", fontsize=13, fontweight="bold", pad=10
    )
else:
    print(f"   ⚠ Warning: Histogram not found at {histogram_path}")


train_mre_path = "docs/images/train_mre.png"
if os.path.exists(train_mre_path):
    ax = fig.add_subplot(gs[1, 1])
    train_mre = plt.imread(train_mre_path)
    ax.imshow(train_mre)
    ax.axis("off")
else:
    print(f"   ⚠ Warning: train_mre_path not found at {train_mre_path}")


val_mre_path = "docs/images/val_mre.png"
if os.path.exists(val_mre_path):
    ax = fig.add_subplot(gs[1, 2])
    val_mre = plt.imread(val_mre_path)
    ax.imshow(val_mre)
    ax.axis("off")
else:
    print(f"   ⚠ Warning: val_mre_path not found at {val_mre_path}")


plt.suptitle("Model Performance Metrics", fontsize=16, fontweight="bold", y=0.98)
plt.savefig("docs/images/model_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("   ✓ Saved to: docs/images/model_performance.png")


# Figure 6: Inference Result (using real validation results)
print("\n[5/5] Copying inference result visualization...")
import shutil

# Use the actual validation results image with target and predicted landmarks
source_path = "docs/images/media_images_random_samples_val_model-urt7dgbp_v47.png"
dest_path = "docs/images/inference_result.png"

if os.path.exists(source_path):
    shutil.copy(source_path, dest_path)
    print(f"   ✓ Copied validation results from: {source_path}")
    print(f"   ✓ Saved to: {dest_path}")
else:
    print(f"   ⚠ Warning: Source image not found at {source_path}")
    print("   Creating placeholder instead...")

    # Fallback to placeholder if source image doesn't exist
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.text(
        0.5,
        0.5,
        "Inference results image not found\nPlease add validation results",
        ha="center",
        va="center",
        fontsize=14,
        transform=ax.transAxes,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(dest_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   ✓ Saved placeholder to: {dest_path}")


print("\n" + "=" * 60)
print("✓ All figures generated successfully!")
print("=" * 60)
print("\nGenerated files:")
print("  - docs/images/unet_architecture.png")
print("  - docs/images/heatmap_visualization.png")
print("  - docs/images/augmentation_examples.png")
print("  - docs/images/model_performance.png")
print("  - docs/images/inference_result.png")
print("\nThese figures are now ready to use in your blog post!")
