"""The functions and classes for defining deep neural network model architectures."""

from typing import Union

import numpy as np
import torch
from segmentation_models_pytorch.encoders import get_encoder
from torch import nn
import pytorch_lightning as pl
from typing import *


class MultiTaskLandmarkUNetCustom(pl.LightningModule):
    """Multi-task U-Net architecture for landmark detection and image processing tasks"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels1: int = 1,
        out_channels2: int = 1,
        out_channels3: int = 13,
        out_channels4: int = 19,
        enc_chan_multiplier: int = 1,
        dec_chan_multiplier: int = 1,
        backbone_encoder: Union[None, str] = None,
        backbone_weights: Union[str, None] = "imagenet",
        freeze_backbone: bool = True,
        enc_out_chans: np.ndarray = np.array([4, 8, 16, 32, 64]),
        dec_out_chans: np.ndarray = np.array([64, 32, 16, 8, 4]),
    ) -> None:
        """
        Initialize the MultiTaskLandmarkUNet1 model

        Parameters:
            in_channels (int): Number of input channels (default: 3)
            out_channels1 (int): Number of output channels for image reconstruction (default: 1)
            out_channels2 (int): Number of output channels for edge detection (default: 1)
            out_channels3 (int): Number of output channels for v landmark detection (default: 13)
            out_channels4 (int): Number of output channels for f landmark detection (default: 19)
            enc_chan_multiplier (int): Multiplication factor that increases the hidden channels of the encoder (default: 1)
            dec_chan_multiplier (int): Multiplication factor that increases the hidden channels of the decoder (default: 1)
            backbone_encoder: Union[None, str] = None,
            backbone_weights: Union[str, None] = "imagenet",
        """
        super().__init__()

        assert isinstance(enc_chan_multiplier, int)
        assert isinstance(dec_chan_multiplier, int)

        # encoder
        self.backbone_encoder = backbone_encoder
        if backbone_encoder is not None:
            self.encoder = get_encoder(
                backbone_encoder,
                in_channels=1,
                depth=5,
                weights=backbone_weights,
            )
            if freeze_backbone:
                for param in self.encoder.parameters():
                    param.requires_grad = False
            # infer the encoder output channels from the model itself
            out = self.encoder(torch.randn(1, 1, 256, 256))
            if not isinstance(out, list) and len(out) < 5:
                raise ValueError(
                    "The selected backbone for the encoder does not "
                    "have sufficient depth! The encoder has to output "
                    "at least 5 feature maps!"
                )
            out.reverse()
            enc_out_chans = [out[i].shape[1] for i in range(len(out))]
            enc_out_chans = enc_out_chans[:5]
            enc_out_chans.reverse()

        else:
            # set the number of output channels in the encoder
            self.dconv1 = DoubleConv(in_channels, enc_out_chans[0])
            self.dconv2 = DoubleConv(enc_out_chans[0], enc_out_chans[1])
            self.dconv3 = DoubleConv(enc_out_chans[1], enc_out_chans[2])
            self.dconv4 = DoubleConv(enc_out_chans[2], enc_out_chans[3])
            self.dconv5 = DoubleConv(enc_out_chans[3], enc_out_chans[4])

        # define the number of channels of encoder and the decoder
        enc_out_chans = enc_out_chans * enc_chan_multiplier
        dec_out_chans = dec_out_chans * dec_chan_multiplier

        # bridge
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # decoder
        self.upconv4 = DoubleConv(enc_out_chans[4] + enc_out_chans[3], dec_out_chans[0])
        self.upconv3 = DoubleConv(dec_out_chans[0] + enc_out_chans[2], dec_out_chans[1])
        self.upconv2 = DoubleConv(dec_out_chans[1] + enc_out_chans[1], dec_out_chans[2])
        self.upconv1 = DoubleConv(dec_out_chans[2] + enc_out_chans[0], dec_out_chans[3])

        # outputs
        # task 1
        self.conv1_1 = DoubleConv(
            dec_out_chans[3],
            dec_out_chans[4],
        )
        self.conv1_2 = OutputTransition(
            dec_out_chans[4],
            out_channels1,
            kernel_size=1,
        )

        # task 2
        self.conv2_1 = DoubleConv(
            dec_out_chans[3],
            dec_out_chans[4],
        )
        self.conv2_2 = OutputTransition(
            dec_out_chans[4],
            out_channels2,
            kernel_size=1,
        )

        # task 3
        self.conv3_1 = DoubleConv(
            dec_out_chans[3],
            dec_out_chans[4],
        )
        self.conv3_2 = OutputTransition(
            dec_out_chans[4],
            out_channels3,
            kernel_size=1,
        )

        # task 4
        self.conv4_1 = DoubleConv(
            dec_out_chans[3],
            dec_out_chans[4],
        )
        self.conv4_2 = OutputTransition(
            dec_out_chans[4],
            out_channels4,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor, task_id: int):
        # store the input shape
        # inp_shape = x.shape
        # Down sampling
        if self.backbone_encoder is not None:
            _, x1, x3, x5, x7, x9 = self.encoder(x)
        else:
            x1 = self.dconv1(x)
            x2 = self.maxpool(x1)
            x3 = self.dconv2(x2)
            x4 = self.maxpool(x3)
            x5 = self.dconv3(x4)
            x6 = self.maxpool(x5)
            x7 = self.dconv4(x6)
            x8 = self.maxpool(x7)
            x9 = self.dconv5(x8)

        # Up sampling
        x = self.upsample(x9)
        x = torch.cat([x, x7], dim=1)
        x = self.upconv4(x)
        x = self.upsample(x)
        x = torch.cat([x, x5], dim=1)
        x = self.upconv3(x)
        x = self.upsample(x)
        x = torch.cat([x, x3], dim=1)
        x = self.upconv2(x)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.upconv1(x)

        # upsample once more if we have a backbone encoder
        if self.backbone_encoder is not None:
            x = self.upsample(x)

        # return the outputs based on the task
        if task_id == 1:
            # Output for task 1: unsupervised image reconstruction
            x = self.conv1_1(x)
            x = self.conv1_2(x)
        elif task_id == 2:
            # Output for task 2: supervised edge detection
            x = self.conv2_1(x)
            x = self.conv2_2(x)
        elif task_id == 3:
            # Output for task 3: supervised vertebral landmark detection
            x = self.conv3_1(x)
            x = self.conv3_2(x)
        elif task_id == 4:
            # Output for task 4: supervised facial landmark detection
            x = self.conv4_1(x)
            x = self.conv4_2(x)
        else:
            raise ValueError("Bad Task ID passed")
        return x


class DoubleConv(nn.Module):
    """Double Convolution Layer"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class LUConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(out_channels)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def make_n_conv_layer(in_channels, depth, double_channel=False):
    if double_channel:
        layer1 = LUConv(in_channels, 32 * (2 ** (depth + 1)))
        layer2 = LUConv(32 * (2 ** (depth + 1)), 32 * (2 ** (depth + 1)))
    else:
        layer1 = LUConv(in_channels, 32 * (2**depth))
        layer2 = LUConv(32 * (2**depth), 32 * (2**depth) * 2)

    return nn.Sequential(layer1, layer2)


class DownTransition(nn.Module):
    def __init__(self, in_channels, depth):
        super(DownTransition, self).__init__()
        self.ops = make_n_conv_layer(in_channels, depth)
        self.pool = nn.MaxPool2d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.pool(out_before_pool)
        return out, out_before_pool


class UpTransition(nn.Module):
    def __init__(self, in_channels, out_channels, depth):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.ops = make_n_conv_layer(
            in_channels + out_channels // 2, depth, double_channel=True
        )

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv, skip_x), 1)
        out = self.ops(concat)
        return out


class OutputTransition(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
    ):
        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size)
        self.act = nn.Sigmoid()

    def forward(self, x):
        out = self.act(self.final_conv(x))
        return out


class UNetCL2023(pl.LightningModule):
    def __init__(self, in_channels=3, out_channels=38):
        super(UNetCL2023, self).__init__()

        self.down_tr64 = DownTransition(in_channels, 0)
        self.down_tr128 = DownTransition(64, 1)
        self.down_tr256 = DownTransition(128, 2)
        self.down_tr512 = DownTransition(256, 3)

        self.up_tr256 = UpTransition(512, 512, 2)
        self.up_tr128 = UpTransition(256, 256, 1)
        self.up_tr64 = UpTransition(128, 128, 0)
        self.out_tr = OutputTransition(64, out_channels)

    def forward(self, x: torch.Tensor, **kwargs):
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128, self.skip_out128 = self.down_tr128(self.out64)
        self.out256, self.skip_out256 = self.down_tr256(self.out128)
        self.out512, self.skip_out512 = self.down_tr512(self.out256)

        self.out_up_256 = self.up_tr256(self.out512, self.skip_out256)
        self.out_up_128 = self.up_tr128(self.out_up_256, self.skip_out128)
        self.out_up_64 = self.up_tr64(self.out_up_128, self.skip_out64)
        self.out = self.out_tr(self.out_up_64)

        return self.out


class PretrainedUNetCL2023(pl.LightningModule):
    def __init__(self, model_path, out_channels):
        super(PretrainedUNetCL2023, self).__init__()

        # Load the pre-trained model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(model_path, map_location=device)

        # Define your model architecture here
        self.model = UNetCL2023(
            in_channels=3, out_channels=38
        )  # default input and output are 3 and 38

        # Load the state dictionary into the model
        self.model.load_state_dict(state_dict)

        # replace first and the last blocks
        self.model.down_tr64 = DownTransition(1, 0)
        self.model.out_tr = OutputTransition(in_channels=64, out_channels=13)

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.model(x)
        return x


def load_model(model_name: str, **kwargs):
    if model_name == "unet_cl2023":
        model = UNetCL2023(**kwargs)  # in_channels=3, n_class=38
    if model_name == "unet_cl2023_pretrained":
        model = PretrainedUNetCL2023(**kwargs)  # model_path, n_class=38
    elif model_name == "custom_unet":
        model = MultiTaskLandmarkUNetCustom(**kwargs)
    else:
        raise ValueError(
            "Please input valid model name, {} not in model zones.".format(model_name)
        )
    return model


if __name__ == "__main__":
    model = load_model(model_name="UNetCL2023")
    print(model)
