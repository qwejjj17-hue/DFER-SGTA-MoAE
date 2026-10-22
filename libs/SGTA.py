import torch
import torch.nn as nn


class SGTA(nn.Module):
    """Spatial-Group Token Attention: grouped spatial-token self-attention."""

    def __init__(self, in_channels, num_heads, reduction):
        super().__init__()
        self.qkv_conv = nn.Conv3d(in_channels, ..., kernel_size=1, bias=False)
        self.attn = nn.MultiheadAttention(embed_dim=..., num_heads=num_heads)

    def forward(self, x):
        # Q/K/V projection and grouped self-attention (see Sec. 3.3 of the paper).
        qkv = self.qkv_conv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        attn_out, _ = self.attn(q, k, v)
        # Projection, normalization and residual connection are omitted here.
        # The complete implementation can be requested from the authors by email.
        return attn_out
