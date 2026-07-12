import torch

from voxel_vit_patch_embedding import VoxelPatchEmbedding


def main() -> None:
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
    voxel_tokens, patch_grid = encoder(voxel, return_patch_grid=True)

    print(f"patch_grid: {patch_grid}")
    print(f"voxel_tokens: {tuple(voxel_tokens.shape)}")

    image_tokens = torch.randn(batch_size, 256, vlm_hidden_dim)
    prompt_tokens = torch.randn(batch_size, 64, vlm_hidden_dim)
    fused_tokens = torch.cat([image_tokens, voxel_tokens, prompt_tokens], dim=1)

    print(f"fused_tokens: {tuple(fused_tokens.shape)}")


if __name__ == "__main__":
    main()
