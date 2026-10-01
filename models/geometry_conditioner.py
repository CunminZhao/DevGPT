import torch
import torch.nn as nn


class GeometryConditioner(nn.Module):
    def __init__(
        self,
        geometry_dim,
        hidden_dims,
        condition_channels,
        film_channels,
        spatial_shape,
        dropout,
    ):
        super().__init__()
        self.geometry_dim = geometry_dim
        self.condition_channels = condition_channels
        self.film_channels = film_channels
        self.spatial_shape = spatial_shape

        d, h, w = spatial_shape
        condition_size = condition_channels * d * h * w

        layers = []
        in_dim = geometry_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, condition_size))
        self.mlp = nn.Sequential(*layers)

        self.spatial_projector = nn.Sequential(
            nn.Conv3d(condition_channels, condition_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(condition_channels, affine=True),
            nn.GELU(),
            nn.Conv3d(condition_channels, 2 * film_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(2 * film_channels, affine=True),
            nn.GELU(),
            nn.Conv3d(2 * film_channels, 2 * film_channels, kernel_size=1),
        )

    def forward(self, geometry):
        if geometry.dim() != 2:
            raise ValueError(f"geometry must have shape [B, G], got {tuple(geometry.shape)}")
        if geometry.size(1) != self.geometry_dim:
            raise ValueError(f"expected geometry_dim={self.geometry_dim}, got {geometry.size(1)}")

        batch_size = geometry.size(0)
        d, h, w = self.spatial_shape

        condition = self.mlp(geometry)
        condition = condition.view(batch_size, self.condition_channels, d, h, w)
        condition = self.spatial_projector(condition)

        gamma, beta = torch.chunk(condition, chunks=2, dim=1)
        return gamma, beta
