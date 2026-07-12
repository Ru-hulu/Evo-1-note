import math
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn


Tuple3 = Tuple[int, int, int]


def _to_3tuple(value: Union[int, Tuple3], name: str) -> Tuple3:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
        return (value, value, value)

    if len(value) != 3:
        raise ValueError(f"{name} must be an int or a 3-tuple, got {value}.")

    out = tuple(int(v) for v in value)
    if any(v <= 0 for v in out):
        raise ValueError(f"{name} values must be positive, got {value}.")
    return out


def _check_divisible(grid_size: Tuple3, patch_size: Tuple3) -> None:
    bad = [g % p for g, p in zip(grid_size, patch_size)]
    if any(bad):
        raise ValueError(
            "Voxel grid size must be divisible by patch size. "
            f"Got grid_size={grid_size}, patch_size={patch_size}."
        )


def _axis_sincos(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    if dim <= 0:
        return torch.empty(length, 0, device=device, dtype=dtype)

    pos = torch.arange(length, device=device, dtype=torch.float32)
    half_dim = math.ceil(dim / 2)
    omega = torch.arange(half_dim, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(half_dim, 1)))
    angles = pos[:, None] * omega[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    return emb[:, :dim].to(dtype=dtype)


def build_3d_sincos_position_embedding(
    patch_grid_size: Tuple3,
    embed_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Build a 3D sin-cos positional embedding with shape (1, N, embed_dim)."""

    d, h, w = patch_grid_size
    d_dim = embed_dim // 3
    h_dim = embed_dim // 3
    w_dim = embed_dim - d_dim - h_dim

    d_emb = _axis_sincos(d, d_dim, device, dtype)
    h_emb = _axis_sincos(h, h_dim, device, dtype)
    w_emb = _axis_sincos(w, w_dim, device, dtype)

    d_grid = d_emb[:, None, None, :].expand(d, h, w, d_dim)
    h_grid = h_emb[None, :, None, :].expand(d, h, w, h_dim)
    w_grid = w_emb[None, None, :, :].expand(d, h, w, w_dim)
    pos = torch.cat([d_grid, h_grid, w_grid], dim=-1)
    return pos.reshape(1, d * h * w, embed_dim)


class VoxelPatchEmbedding(nn.Module):
    """ViT-style patch embedding for dense 3D voxel grids.

    Input shape:
        voxel: (batch, channels, depth, height, width)

    Output shape:
        tokens: (batch, num_voxel_tokens, output_dim)

    A Conv3d with kernel_size=stride=patch_size is equivalent to flattening
    each non-overlapping 3D patch and applying one shared Linear projection.
    """

    def __init__(
        self,
        in_channels: int,  # voxel 输入通道数，例如 occupancy + EE heatmap + SDF ，我们默认就是1
        embed_dim: int,  # Conv3d patch embedding 输出的 token 维度
        patch_size: Union[int, Tuple3] = 8,  # 3D patch 大小（这一小块有多大），int 表示 D/H/W 使用同一个值
        grid_size: Optional[Union[int, Tuple3]] = None,  # 固定 voxel 网格大小，例如 32 或 (32, 32, 32)
        output_dim: Optional[int] = None,  # 最终输出 token 维度，通常设为 VLM hidden dim
        add_modality_embedding: bool = True,  # 是否加入 voxel modality embedding，帮助 VLM 区分图像/文本/voxel token，形式上很简单，就是给每个token加上这个embedding
        bias: bool = True,  # Conv3d patch embedding 是否使用 bias
        norm_layer: Optional[type[nn.Module]] = nn.LayerNorm,  # patch token 上的归一化层；None 表示不使用
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}.")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}.")

        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        self.output_dim = int(output_dim or embed_dim)
        self.patch_size = _to_3tuple(patch_size, "patch_size")
        self.grid_size = _to_3tuple(grid_size, "grid_size") if grid_size is not None else None

        self.patch_embed = nn.Conv3d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )

        self.patch_grid_size: Optional[Tuple3] = None
        self.num_patches: Optional[int] = None
        _check_divisible(self.grid_size, self.patch_size)
        self.patch_grid_size = tuple(g // p for g, p in zip(self.grid_size, self.patch_size))
        self.num_patches = math.prod(self.patch_grid_size)

        if add_modality_embedding:
            self.modality_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            nn.init.trunc_normal_(self.modality_embed, std=0.02)
        else:
            self.register_parameter("modality_embed", None)

        self.norm = norm_layer(self.embed_dim) if norm_layer is not None else nn.Identity()
        self.projector = (
            nn.Identity()
            if self.output_dim == self.embed_dim
            else nn.Linear(self.embed_dim, self.output_dim)
        )

    def _validate_input(self, voxel: Tensor) -> Tuple3:
        if voxel.ndim != 5:
            raise ValueError(
                "voxel must have shape (batch, channels, depth, height, width), "
                f"got {tuple(voxel.shape)}."
            )
        if voxel.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} voxel channels, got {voxel.shape[1]}."
            )

        grid_size = tuple(int(v) for v in voxel.shape[-3:])
        _check_divisible(grid_size, self.patch_size)

        return grid_size

    def _position_embedding(self, patch_grid_size: Tuple3, tokens: Tensor) -> Tensor:
        return build_3d_sincos_position_embedding(
            patch_grid_size=patch_grid_size,
            embed_dim=self.embed_dim,
            device=tokens.device,
            dtype=tokens.dtype,
        )

    def forward(self, voxel: Tensor, return_patch_grid: bool = False):
        """Embed a dense voxel grid into VLM-ready spatial tokens."""

        self._validate_input(voxel)

        x = self.patch_embed(voxel)
        patch_grid_size = tuple(int(v) for v in x.shape[-3:])

        x = x.flatten(2).transpose(1, 2).contiguous()
        x = x + self._position_embedding(patch_grid_size, x)
        if self.modality_embed is not None:
            x = x + self.modality_embed.to(device=x.device, dtype=x.dtype)

        x = self.norm(x)
        x = self.projector(x)

        if return_patch_grid:
            return x, patch_grid_size
        return x