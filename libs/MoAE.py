import torch
import torch.nn as nn
import torch.nn.functional as F


class AdapterExpert(nn.Module):
    """Lightweight bottleneck adapter expert: down-project, activate, up-project."""

    def __init__(self, in_channels, reduction_ratio):
        super().__init__()
        self.down_conv = nn.Conv3d(in_channels, ..., kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.up_conv = nn.Conv3d(..., in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.relu(self.down_conv(x))
        return self.up_conv(out)


class MoAE(nn.Module):
    """Mixture of Adapter Experts with a softmax gating router."""

    def __init__(self, in_channels, num_experts, reduction_ratio):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, num_experts, kernel_size=1, bias=False)
        )
        self.experts = nn.ModuleList(
            [AdapterExpert(in_channels, reduction_ratio) for _ in range(num_experts)]
        )

    def forward(self, x):
        gate_weights = F.softmax(self.gate(x), dim=1)
        out = 0
        for i, expert in enumerate(self.experts):
            out += expert(x) * gate_weights[:, i:i + 1, :, :, :]
        # Residual connection omitted here; full code available on request.
        return out
