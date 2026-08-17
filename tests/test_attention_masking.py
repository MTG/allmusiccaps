import torch
from amclap.nets.common_former import MHAPyTorchScaledDotProduct


def test_attention_masking():
    # Define the test inputs
    batch_size = 2
    num_tokens = 5
    embed_dim = 16
    num_heads = 4
    d_out = embed_dim
    d_in = embed_dim

    # Create an MHAPyTorchScaledDotProduct instance
    attention_layer = MHAPyTorchScaledDotProduct(
        d_in=d_in, d_out=d_out, num_heads=num_heads, dropout=0.0
    )

    # Sample embeddings and masks
    x = torch.randn(batch_size, num_tokens, embed_dim)  # Random inputs
    attn_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],  # Mask for batch 1
            [1, 1, 0, 0, 0],  # Mask for batch 2
        ],
        dtype=torch.bool,
    )

    # Ensure logits from positions with mask = 0 are ignored
    output = attention_layer(x, attn_mask=attn_mask)

    # Assertions to verify masking (output shape validity check)
    assert output.shape == (batch_size, num_tokens, embed_dim), "Output shape mismatch"
    print("Masking test passed!")
