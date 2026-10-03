import math
import torch.nn as nn


def initialize_transformer(module: nn.Module, d_model: int) -> None:
    """按 Transformer 常见做法初始化线性层和词向量。"""
    for child in module.modules():
        if isinstance(child, nn.Linear):
            # 线性层用 Xavier 初始化，通常能让前期训练更稳。
            nn.init.xavier_uniform_(child.weight)
            if child.bias is not None:
                nn.init.zeros_(child.bias)
        elif isinstance(child, nn.Embedding):
            # 词向量按 d_model 缩放初始化；padding 行固定为 0，避免 PAD 带入信息。
            nn.init.normal_(child.weight, mean=0.0, std=d_model ** -0.5)
            if child.padding_idx is not None:
                nn.init.zeros_(child.weight[child.padding_idx])
