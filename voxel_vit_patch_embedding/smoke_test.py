import torch

from voxel_vit_patch_embedding import VoxelPatchEmbedding


def check_shape_and_backward() -> None:
    batch_size = 2
    voxel_channels = 4
    vlm_hidden_dim = 896

    encoder = VoxelPatchEmbedding(
        in_channels=voxel_channels,
        grid_size=(32, 32, 32),
        patch_size=8,
        embed_dim=256,
        output_dim=vlm_hidden_dim,
    )

    voxel = torch.randn(batch_size, voxel_channels, 32, 32, 32)
    tokens, patch_grid = encoder(voxel, return_patch_grid=True)

    assert patch_grid == (4, 4, 4), patch_grid
    assert tokens.shape == (batch_size, 64, vlm_hidden_dim), tokens.shape

    loss = tokens.square().mean()
    loss.backward()

    assert encoder.patch_embed.weight.grad is not None
    assert encoder.projector.weight.grad is not None

    print("OK: shape + backward")
    print(f"patch_grid={patch_grid}")
    print(f"tokens={tuple(tokens.shape)}")


def check_dynamic_grid() -> None:
    encoder = VoxelPatchEmbedding(
        in_channels=1,
        patch_size=(4, 8, 8),
        embed_dim=96,
        output_dim=96,
    )

    voxel = torch.randn(1, 1, 16, 32, 32)
    tokens = encoder(voxel)

    assert tokens.shape == (1, 64, 96), tokens.shape
    print("OK: dynamic grid")
    print(f"tokens={tuple(tokens.shape)}")


if __name__ == "__main__":
    check_shape_and_backward()
    check_dynamic_grid()
