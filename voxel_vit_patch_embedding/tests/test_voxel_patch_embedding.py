import pytest
import torch

from voxel_vit_patch_embedding import VoxelPatchEmbedding


def test_voxel_patch_embedding_shape() -> None:
    encoder = VoxelPatchEmbedding(
        in_channels=3,
        grid_size=(32, 32, 32),
        patch_size=8,
        embed_dim=128,
        output_dim=896,
    )

    voxel = torch.randn(2, 3, 32, 32, 32)
    tokens, patch_grid = encoder(voxel, return_patch_grid=True)

    assert patch_grid == (4, 4, 4)
    assert tokens.shape == (2, 64, 896)


def test_supports_dynamic_grid() -> None:
    encoder = VoxelPatchEmbedding(
        in_channels=1,
        patch_size=(4, 8, 8),
        embed_dim=96,
        output_dim=96,
    )

    voxel = torch.randn(1, 1, 16, 32, 32)
    tokens = encoder(voxel)

    assert tokens.shape == (1, 64, 96)


def test_rejects_non_divisible_grid() -> None:
    encoder = VoxelPatchEmbedding(
        in_channels=1,
        grid_size=(32, 32, 32),
        patch_size=8,
        embed_dim=64,
    )

    with pytest.raises(ValueError, match="divisible"):
        encoder(torch.randn(1, 1, 31, 32, 32))
