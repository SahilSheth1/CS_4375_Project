import sys
import torch
from vit_model import (
    MultiHeadSelfAttention,
    FeedForwardBlock,
    TransformerEncoderLayer,
    ViTEncoder,
)


def run_tests() -> None:
    passed = 0
    total = 0

    def check(label: str, condition: bool) -> None:
        nonlocal passed, total
        total += 1
        status = "PASS" if condition else "FAIL"
        if condition:
            passed += 1
        print(f"  [{status}] {label}")

    #
    print("\n \tMHSA shape tests \t")
    x_196 = torch.randn(2, 196, 256)

    mhsa4 = MultiHeadSelfAttention(embed_dim=256, num_heads=4)
    out4 = mhsa4(x_196)
    check("MHSA output shape", out4.shape == (2, 196, 256))

    mhsa8 = MultiHeadSelfAttention(embed_dim=256, num_heads=8)
    out8 = mhsa8(x_196)
    check("MHSA 8-head shape", out8.shape == (2, 196, 256))

    raised = False
    try:
        MultiHeadSelfAttention(256, 7)
    except AssertionError:
        raised = True
    check("MHSA invalid heads raises", raised)

    print("\n \t FFN shape tests \t")
    ffn = FeedForwardBlock(embed_dim=256, mlp_ratio=4)
    out = ffn(x_196)
    check("FFN output shape", out.shape == (2, 196, 256))


    print("\n\t TransformerEncoderLayer shape tests \t")
    layer = TransformerEncoderLayer(embed_dim=256, num_heads=4)
    out = layer(x_196)
    check("EncoderLayer output shape", out.shape == (2, 196, 256))
    print("\n\t ViTEncoder end-to-end tests \t")
    img = torch.randn(2, 3, 224, 224)

    # Expperiment 2 config
    enc2 = ViTEncoder(
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=256,
        num_layers=4,
        num_heads=4,
    )
    out2 = enc2(img)
    check("ViTEncoder Exp2 shape", out2.shape == (2, 196, 256))

    # Experiment 3 config 
    enc3 = ViTEncoder(
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=256,
        num_layers=6,
        num_heads=8,
    )
    out3 = enc3(img)
    check("ViTEncoder Exp3 shape", out3.shape == (2, 196, 256))

    # Where the evaluation method needs to be declared outisde of the model.ipynb file
    enc2.eval()
    with torch.no_grad():
        a = enc2(img)
        b = enc2(img)
    check("ViTEncoder deterministic eval", torch.equal(a, b))

    # Dropout active in train mode
    enc_drop = ViTEncoder(
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=256,
        num_layers=4,
        num_heads=4,
        dropout=0.5,
    )
    enc_drop.train()
    with torch.no_grad():
        c = enc_drop(img)
        d = enc_drop(img)
    check("ViTEncoder dropout active in train", not torch.equal(c, d))

    # Taking redundnat measures 
    print("\n \tParameter count (for redundancy) \t")
    enc2_fresh = ViTEncoder(
        image_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=256,
        num_layers=4,
        num_heads=4,
    )
    param_count = sum(p.numel() for p in enc2_fresh.parameters() if p.requires_grad)
    print(f"  Exp2 trainable parameters: {param_count:,}")
    check("ViTEncoder param count in range", 1_000_000 <= param_count <= 20_000_000)

    # Summary
    print(f"\n{passed}/{total} tests passed.")
    if passed < total:
        sys.exit(1)


if __name__ == "__main__":
    run_tests()
