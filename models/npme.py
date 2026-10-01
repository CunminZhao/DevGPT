import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.geometry_conditioner import GeometryConditioner
from models.geometry_readout import GeometryReadout


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm3d(out_ch, affine=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm3d(out_ch, affine=True)

        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, bias=False),
                nn.InstanceNorm3d(out_ch, affine=True),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class CNNEncoder(nn.Module):
    def __init__(self, num_classes, base_ch, latent_channels):
        super().__init__()
        if latent_channels % 2 != 0:
            raise ValueError(f"latent_channels must be even, got {latent_channels}")
        self.latent_channels = latent_channels
        self.geo_channels = latent_channels // 2

        self.init_conv = nn.Sequential(
            nn.Conv3d(num_classes, base_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(base_ch, affine=True),
            nn.LeakyReLU(0.1),
        )
        self.layer1 = ResBlock3D(base_ch, base_ch * 2)
        self.layer2 = ResBlock3D(base_ch * 2, base_ch * 4)
        self.layer3 = ResBlock3D(base_ch * 4, base_ch * 8)
        self.layer4 = ResBlock3D(base_ch * 8, base_ch * 16)
        self.layer5 = ResBlock3D(base_ch * 16, base_ch * 16)

        self.adaptive_pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.bottleneck = nn.Conv3d(base_ch * 16, latent_channels, kernel_size=1)

    def forward(self, x):
        x = F.max_pool3d(self.layer1(self.init_conv(x)), 2)
        x = F.max_pool3d(self.layer2(x), 2)
        x = F.max_pool3d(self.layer3(x), 2)
        x = F.max_pool3d(self.layer4(x), 2)
        x = F.max_pool3d(self.layer5(x), 2)
        x = self.adaptive_pool(x)
        return self.bottleneck(x)


class CNNDecoder(nn.Module):
    def __init__(self, num_classes, base_ch, latent_channels, target_shape):
        super().__init__()
        self.target_shape = target_shape
        self.ch_bnb = base_ch * 16

        self.unbottleneck = nn.Conv3d(latent_channels, self.ch_bnb, kernel_size=1)
        self.up1 = nn.ConvTranspose3d(self.ch_bnb, base_ch * 8, 2, stride=2)
        self.res1 = ResBlock3D(base_ch * 8, base_ch * 8)
        self.up2 = nn.ConvTranspose3d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.res2 = ResBlock3D(base_ch * 4, base_ch * 4)
        self.up3 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.res3 = ResBlock3D(base_ch * 2, base_ch * 2)
        self.up4 = nn.ConvTranspose3d(base_ch * 2, base_ch, 2, stride=2)
        self.res4 = ResBlock3D(base_ch, base_ch)
        self.up5 = nn.ConvTranspose3d(base_ch, base_ch, 2, stride=2)
        self.res5 = ResBlock3D(base_ch, base_ch)
        self.final = nn.Conv3d(base_ch, num_classes, 1)

    def forward(self, z):
        x = self.unbottleneck(z)
        x = self.res1(self.up1(x))
        x = self.res2(self.up2(x))
        x = self.res3(self.up3(x))
        x = self.res4(self.up4(x))
        x = self.res5(self.up5(x))

        logits = self.final(x)
        if logits.shape[2:] != self.target_shape:
            logits = F.interpolate(logits, size=self.target_shape, mode="trilinear", align_corners=True)
        return logits


class ResNetAutoEncoder(nn.Module):
    def __init__(
        self,
        num_classes,
        base_ch,
        latent_channels,
        input_shape,
        geometry_dim,
        latent_spatial_shape,
        geometry_conditioner_hidden_dims,
        geometry_conditioner_channels,
        geometry_conditioner_dropout,
        geometry_readout_output_dims,
        geometry_readout_dropout,
    ):
        super().__init__()
        if latent_channels % 2 != 0:
            raise ValueError(f"latent_channels must be even, got {latent_channels}")

        self.geo_channels = latent_channels // 2
        if self.geo_channels != 16:
            raise ValueError(
                f"geometry readout expects 16 geometry groups; got {self.geo_channels}. "
                "Set latent_channels=32."
            )

        self.encoder = CNNEncoder(num_classes, base_ch, latent_channels)
        self.conditioner = GeometryConditioner(
            geometry_dim=geometry_dim,
            hidden_dims=geometry_conditioner_hidden_dims,
            condition_channels=geometry_conditioner_channels,
            film_channels=self.geo_channels,
            spatial_shape=latent_spatial_shape,
            dropout=geometry_conditioner_dropout,
        )
        self.geometry_readout = GeometryReadout(
            output_dims=geometry_readout_output_dims,
            dropout=geometry_readout_dropout,
        )
        self.decoder = CNNDecoder(num_classes, base_ch, latent_channels, input_shape)

    def forward(self, x, geometry):
        h = self.encoder(x)
        h_geo, h_high = torch.split(h, [self.geo_channels, self.geo_channels], dim=1)
        gamma, beta = self.conditioner(geometry)
        h_geo = (1.0 + gamma) * h_geo + beta
        z = torch.cat([h_geo, h_high], dim=1)
        z = math.pi * torch.tanh(z)
        geo_pred = self.geometry_readout(z[:, : self.geo_channels])
        recon = self.decoder(z)
        return recon, z, geo_pred
