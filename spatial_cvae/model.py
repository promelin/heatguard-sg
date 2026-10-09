"""Conditional spatial VAE for multi-layer HDB neighbourhood layouts."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.block = ConvBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.down(x))


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels: int, base_channels: int) -> None:
        super().__init__()
        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 8, base_channels * 8]
        self.stem = ConvBlock(in_channels, widths[0])
        self.downs = nn.ModuleList(
            [DownBlock(widths[index], widths[index + 1]) for index in range(len(widths) - 1)]
        )
        self.widths = widths

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = [self.stem(x)]
        for block in self.downs:
            features.append(block(features[-1]))
        return features


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
        self.block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        modulation: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x = self.up(x)
        if modulation is not None:
            scale, shift = modulation
            x = x * (1.0 + 0.35 * torch.tanh(scale)) + 0.35 * shift
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class ConditionalSpatialVAE(nn.Module):
    def __init__(
        self,
        condition_channels: int = 14,
        output_channels: int = 13,
        base_channels: int = 32,
        latent_dim: int = 128,
        tile_size: int = 256,
        latent_injection: bool = False,
        condition_skip_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if tile_size % 32:
            raise ValueError("tile_size must be divisible by 32")
        self.condition_channels = condition_channels
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.tile_size = tile_size
        self.latent_injection = latent_injection
        self.condition_skip_dropout = condition_skip_dropout
        self.bottleneck_size = tile_size // 32

        self.condition_encoder = FeatureEncoder(condition_channels, base_channels)
        self.posterior_encoder = FeatureEncoder(condition_channels + output_channels, base_channels)
        bottleneck_channels = self.posterior_encoder.widths[-1]
        self.posterior_pool = nn.AdaptiveAvgPool2d(1)
        self.to_mu = nn.Linear(bottleneck_channels, latent_dim)
        self.to_logvar = nn.Linear(bottleneck_channels, latent_dim)
        self.from_latent = nn.Linear(
            latent_dim,
            bottleneck_channels * self.bottleneck_size * self.bottleneck_size,
        )
        self.bottleneck = ConvBlock(bottleneck_channels * 2, bottleneck_channels)

        widths = self.condition_encoder.widths
        self.ups = nn.ModuleList(
            [
                UpBlock(widths[5], widths[4], widths[4]),
                UpBlock(widths[4], widths[3], widths[3]),
                UpBlock(widths[3], widths[2], widths[2]),
                UpBlock(widths[2], widths[1], widths[1]),
                UpBlock(widths[1], widths[0], widths[0]),
            ]
        )
        self.latent_affines = nn.ModuleList(
            [nn.Linear(latent_dim, block.up.out_channels * 2) for block in self.ups]
            if latent_injection
            else []
        )
        self.head = nn.Conv2d(widths[0], output_channels, 1)

    def encode(self, condition: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.posterior_encoder(torch.cat([condition, target], dim=1))
        pooled = self.posterior_pool(features[-1]).flatten(1)
        return self.to_mu(pooled), self.to_logvar(pooled).clamp(-12.0, 8.0)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, condition: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        skips = self.condition_encoder(condition)
        channels = skips[-1].shape[1]
        x = self.from_latent(z).view(-1, channels, self.bottleneck_size, self.bottleneck_size)
        x = self.bottleneck(torch.cat([x, skips[-1]], dim=1))
        for index, (block, skip) in enumerate(zip(self.ups, reversed(skips[:-1]))):
            if self.condition_skip_dropout:
                skip = F.dropout2d(skip, p=self.condition_skip_dropout, training=self.training)
            modulation = None
            if self.latent_injection:
                affine = self.latent_affines[index](z)
                scale, shift = affine.chunk(2, dim=1)
                modulation = (scale[:, :, None, None], shift[:, :, None, None])
            x = block(x, skip, modulation)
        return self.head(x)

    def forward(
        self, condition: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(condition, target)
        logits = self.decode(condition, self.reparameterize(mu, logvar))
        return logits, mu, logvar

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, count: int = 1) -> torch.Tensor:
        if condition.shape[0] == 1 and count > 1:
            condition = condition.expand(count, -1, -1, -1)
        elif condition.shape[0] != count:
            raise ValueError("condition batch must be one item or match count")
        z = torch.randn(count, self.latent_dim, device=condition.device)
        return torch.sigmoid(self.decode(condition, z))


def model_from_checkpoint(checkpoint: dict, device: torch.device | str = "cpu") -> ConditionalSpatialVAE:
    config = checkpoint["model_config"]
    model = ConditionalSpatialVAE(**config)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device)
