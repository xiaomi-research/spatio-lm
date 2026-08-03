from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class CausalLM3dOutputWithPast(CausalLMOutputWithPast):
    """
    Base class for outputs of causal language models with condition
    """

    condition_3d: Optional[Tuple[torch.FloatTensor, ...]] = None
    dpt_3d: Optional[Dict[str, torch.FloatTensor]] = None


class FeatureDimPool(nn.Module):
    """
    对 feature_dim 做池化降维（无可学习参数）。
    支持 avg / max 两种方式。

    输入: [B, T, D]
    输出: [B, T, out_dim]
    """

    def __init__(self, out_dim: int, mode: Literal["avg", "max"] = "max"):
        super().__init__()
        assert mode in ["avg", "max"], f"Unsupported mode: {mode}"
        self.out_dim = out_dim
        self.mode = mode

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T, D]
        Returns:
            [B, T, out_dim]
        """
        # 使用自适应池化到指定维度
        if self.mode == "avg":
            x_pooled = F.adaptive_avg_pool1d(x, self.out_dim)
        else:  # max pooling
            x_pooled = F.adaptive_max_pool1d(x, self.out_dim)
        return x_pooled
