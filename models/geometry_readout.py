import math

import torch
import torch.nn as nn


class GeometryGroupHead(nn.Module):
    def __init__(self, out_dim, dropout):
        super().__init__()
        hidden_dim = 8 if out_dim == 1 else 16
        self.net = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(8, affine=True),
            nn.GELU(),
            nn.Conv3d(8, 1, kernel_size=1),
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class GeometryReadout(nn.Module):
    def __init__(self, output_dims, dropout):
        super().__init__()
        self.output_dims = tuple(output_dims)
        self.output_dim = sum(self.output_dims)
        self.heads = nn.ModuleList(
            [GeometryGroupHead(out_dim=out_dim, dropout=dropout) for out_dim in self.output_dims]
        )

    def forward(self, z_geo):
        if z_geo.dim() != 5:
            raise ValueError(f"z_geo must have shape [B, 16, 4, 4, 4], got {tuple(z_geo.shape)}")
        if z_geo.size(1) != len(self.heads):
            raise ValueError(f"expected {len(self.heads)} geometry groups, got {z_geo.size(1)}")

        outputs = []
        for idx, head in enumerate(self.heads):
            outputs.append(head(z_geo[:, idx : idx + 1]))

        geometry = torch.cat(outputs, dim=1)
        return math.pi * torch.tanh(geometry)
