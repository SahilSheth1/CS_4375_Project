import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size:  int = 224,
        patch_size:  int = 16,
        in_channels: int = 3,
        embed_dim:   int = 256,
    ):
        super().__init__()

        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )

        self.image_size  = image_size
        self.patch_size  = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Size of the raw flattened patch vector: P*P*C
        patch_dim = patch_size * patch_size * in_channels

        self.projection_weight = nn.Parameter(
            torch.empty(embed_dim, patch_dim)
        )
        self.projection_bias = nn.Parameter(
            torch.zeros(embed_dim)
        )

        nn.init.kaiming_uniform_(self.projection_weight, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        assert H == self.image_size and W == self.image_size, (
            f"Expected ({self.image_size}, {self.image_size}), got ({H}, {W}). "
            "Make sure your DataLoader resizes images before passing them here."
        )

        P = self.patch_size
        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, self.num_patches, -1)

        patch_embeddings = x @ self.projection_weight.T + self.projection_bias

        return patch_embeddings  # (B, num_patches, embed_dim)

    def extra_repr(self) -> str:
        return (
            f"image_size={self.image_size}, patch_size={self.patch_size}, "
            f"num_patches={self.num_patches}, "
            f"embed_dim={self.projection_bias.shape[0]}"
        )

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, num_patches: int, embed_dim: int):
        super().__init__()

        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )

        # Normal initialisation with small std — standard practice for pos embeds
        nn.init.normal_(self.position_embeddings, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.position_embeddings

    def extra_repr(self) -> str:
        _, N, D = self.position_embeddings.shape
        return f"num_patches={N}, embed_dim={D}"

class PatchAndPositionEmbedding(nn.Module):
    def __init__(
        self,
        image_size:  int = 224,
        patch_size:  int = 16,
        in_channels: int = 3,
        embed_dim:   int = 256,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            image_size  = image_size,
            patch_size  = patch_size,
            in_channels = in_channels,
            embed_dim   = embed_dim,
        )

        self.positional_embedding = LearnablePositionalEmbedding(
            num_patches = self.patch_embedding.num_patches,
            embed_dim   = embed_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(x)       # (B, N, D)
        x = self.positional_embedding(x)  # (B, N, D)
        return x

    @property
    def num_patches(self):
        return self.patch_embedding.num_patches