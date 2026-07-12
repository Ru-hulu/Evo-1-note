# Voxel ViT Patch Embedding

This folder contains a small PyTorch module for turning a dense robot voxel grid
into spatial tokens that can be concatenated with image tokens and prompt tokens
before a VLM/VLA backbone.

## Shape

```text
voxel:       B x C x D x H x W
tokens:      B x N_voxel x output_dim
N_voxel:     (D / patch_D) * (H / patch_H) * (W / patch_W)
```

For example:

```text
32 x 32 x 32 voxel, patch_size=8 -> 4 x 4 x 4 = 64 tokens
32 x 32 x 32 voxel, patch_size=4 -> 8 x 8 x 8 = 512 tokens
```

## Why Conv3d

The implementation uses:

```python
nn.Conv3d(kernel_size=patch_size, stride=patch_size)
```

This is equivalent to flattening every non-overlapping 3D patch and applying the
same linear projection to each patch. It is the dense 3D analogue of the
standard ViT patch embedding used for 2D images.

## Usage

```python
import torch
from voxel_vit_patch_embedding import VoxelPatchEmbedding

voxel_encoder = VoxelPatchEmbedding(
    in_channels=4,
    grid_size=(32, 32, 32),
    patch_size=8,
    embed_dim=256,
    output_dim=896,
)

voxel = torch.randn(2, 4, 32, 32, 32)
voxel_tokens = voxel_encoder(voxel)

image_tokens = torch.randn(2, 256, 896)
prompt_tokens = torch.randn(2, 64, 896)
vlm_tokens = torch.cat([image_tokens, voxel_tokens, prompt_tokens], dim=1)
```

## Remote smoke test

Local development does not require running PyTorch. To validate on a remote 4090
machine:

```bash
chmod +x verify_remote_4090.sh
./verify_remote_4090.sh user@host /tmp/voxel_vit_patch_embedding python3
```

The remote test checks:

```text
32^3 voxel + patch_size=8 -> 64 tokens
token dim -> 896
forward + backward pass
sin-cos 3D position embedding
```

## Recommended first setting

```text
voxel grid:     32^3
patch size:     8
voxel tokens:   64
embed dim:      256
output dim:     VLM hidden dim, e.g. 896 for the current InternVL3 wrapper
position:       fixed 3D sin-cos
```

The module always adds fixed 3D sin-cos position embedding. Use `patch_size=4`
later if 64 voxel tokens are too coarse.
