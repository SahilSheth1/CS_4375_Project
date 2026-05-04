import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()

        assert (
            image_size % patch_size == 0
        ), f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Size of the raw flattened patch vector: P*P*C
        patch_dim = patch_size * patch_size * in_channels

        self.projection_weight = nn.Parameter(torch.empty(embed_dim, patch_dim))
        self.projection_bias = nn.Parameter(torch.zeros(embed_dim))

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

        # (B, num_patches, embed_dim)
        return patch_embeddings

    def extra_repr(self) -> str:
        return (
            f"image_size={self.image_size}, patch_size={self.patch_size}, "
            f"num_patches={self.num_patches}, "
            f"embed_dim={self.projection_bias.shape[0]}"
        )


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, num_patches: int, embed_dim: int):
        super().__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

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
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        self.positional_embedding = LearnablePositionalEmbedding(
            num_patches=self.patch_embedding.num_patches,
            embed_dim=embed_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, N, D)
        x = self.patch_embedding(x)
        # (B, N, D)
        x = self.positional_embedding(x)
        return x

    @property
    def num_patches(self):
        return self.patch_embedding.num_patches


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, D)
        x = self.proj_dropout(self.out_proj(x))
        return x

    def extra_repr(self) -> str:
        embed_dim = self.num_heads * self.head_dim
        return f"num_heads={self.num_heads}, head_dim={self.head_dim}, embed_dim={embed_dim}"


class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = embed_dim * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.net[-2].out_features}, "
            f"hidden_dim={self.net[0].out_features}"
        )


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardBlock(embed_dim, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding = PatchAndPositionEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )
        self.layers = nn.Sequential(
            *[
                TransformerEncoderLayer(embed_dim, num_heads, mlp_ratio, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # store for extra_repr
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.layers(x)
        x = self.norm(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"num_layers={self._num_layers}, num_heads={self._num_heads}, "
            f"embed_dim={self._embed_dim}, num_patches={self.embedding.num_patches}"
        )


class CharFieldHead(nn.Module):
    def __init__(self, embed_dim: int, field: str, dropout: float = 0.1):
        super().__init__()
        from field_vocab import MAX_LEN, CHAR_VOCAB_SIZE

        self.max_len = MAX_LEN[field]
        self.vocab_size = CHAR_VOCAB_SIZE

        # Project each patch token, then attend over them per output position
        self.query = nn.Parameter(torch.zeros(self.max_len, embed_dim))
        nn.init.normal_(self.query, std=0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads=4, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, self.vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D) — patch tokens from encoder
        B = x.size(0)
        # (B, max_len, D)
        q = self.query.unsqueeze(0).expand(B, -1, -1)
        # (B, max_len, D)
        out, _ = self.cross_attn(q, x, x)
        out = self.norm(out)
        out = self.dropout(out)
        # (B, max_len, vocab_size)
        return self.proj(out)


class ReceiptViT(nn.Module):
    FIELDS = ["vendor", "date", "total", "address"]

    def __init__(
        self,
        vocab_sizes: dict,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = ViTEncoder(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )
        self.heads = nn.ModuleDict(
            {field: CharFieldHead(embed_dim, field, dropout) for field in self.FIELDS}
        )

    def forward(self, x: torch.Tensor) -> dict:
        features = self.encoder(x)
        return {field: self.heads[field](features) for field in self.FIELDS}
