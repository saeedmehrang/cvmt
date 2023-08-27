""" The functions and classes for defining deep neural network model architectures. """

from typing import Union

import numpy as np
import torch
from segmentation_models_pytorch.encoders import get_encoder
from torch import nn
import pytorch_lightning as pl


class MultiTaskLandmarkUNetCustom(pl.LightningModule):
    """Multi-task U-Net architecture for landmark detection and image processing tasks"""

    def __init__(
        self, 
        in_channels: int=1, 
        out_channels1: int=1,
        out_channels2: int=1,
        out_channels3: int=13,
        out_channels4: int=19,
        enc_chan_multiplier: int=1,
        dec_chan_multiplier: int=1,
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
            if not isinstance(out, list) and len(out)<5:
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
        enc_out_chans = enc_out_chans*enc_chan_multiplier    
        dec_out_chans = dec_out_chans*dec_chan_multiplier
        
        # bridge
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # decoder
        self.upconv4 = DoubleConv(enc_out_chans[4] + enc_out_chans[3], dec_out_chans[0])
        self.upconv3 = DoubleConv(dec_out_chans[0] + enc_out_chans[2], dec_out_chans[1])
        self.upconv2 = DoubleConv(dec_out_chans[1] + enc_out_chans[1], dec_out_chans[2])
        self.upconv1 = DoubleConv(dec_out_chans[2] + enc_out_chans[0], dec_out_chans[3])
        
        # outputs
        # task 1
        self.conv1_1 = DoubleConv(dec_out_chans[3], dec_out_chans[4],)
        self.conv1_2 = nn.Conv2d(dec_out_chans[4], out_channels1, kernel_size=1)
        
        # task 2
        self.conv2_1 = DoubleConv(dec_out_chans[3], dec_out_chans[4],)
        self.conv2_2 = nn.Conv2d(dec_out_chans[4], out_channels2, kernel_size=1)

        # task 3
        self.conv3_1 = DoubleConv(dec_out_chans[3], dec_out_chans[4],)
        self.conv3_2 = nn.Conv2d(dec_out_chans[4], out_channels3, kernel_size=1)
        
        # task 4
        self.conv4_1 = DoubleConv(dec_out_chans[3], dec_out_chans[4],)
        self.conv4_2 = nn.Conv2d(dec_out_chans[4], out_channels4, kernel_size=1)

    def forward(self, x: torch.Tensor, task_id: int):
        # store the input shape
        inp_shape = x.shape
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
            raise ValueError('Bad Task ID passed')
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
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
