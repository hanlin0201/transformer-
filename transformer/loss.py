import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingLoss(nn.Module):
    """带标签平滑的交叉熵，只统计非 PAD 的目标 token。"""

    def __init__(self, vocab_size: int, pad_id: int, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        # 先把 batch 和序列长度展平成一维，方便统一计算每个 token 的损失。
        logits = logits.reshape(-1, logits.size(-1))
        target = target.reshape(-1)
        non_pad = target != self.pad_id
        token_count = non_pad.sum()
        if token_count.item() == 0:
            return logits.sum() * 0.0, token_count

        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            # 标签平滑：正确答案保留大部分概率，其余概率分给其他非 PAD 词。
            true_dist = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
            true_dist[:, self.pad_id] = 0.0
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist.masked_fill_(~non_pad.unsqueeze(1), 0.0)
        loss = torch.sum(-true_dist * log_probs)
        return loss, token_count

    def feature_loss_chunks(self, features: torch.Tensor, target: torch.Tensor, projection: nn.Module, chunk_tokens: int = 256):
        """把 Decoder 特征分块计算损失，降低一次性投影到词表的显存压力。

        训练时需要对每个分块立刻 backward；如果先把所有分块加起来再反传，
        计算图仍然会留在显存里，就达不到省显存的效果。
        """
        flat_features = features.reshape(-1, features.size(-1))
        flat_target = target.reshape(-1)
        total_tokens = (flat_target != self.pad_id).sum()
        if total_tokens.item() == 0:
            yield flat_features.sum() * 0.0, total_tokens, True
            return
        starts = list(range(0, flat_features.size(0), chunk_tokens))
        for i, start in enumerate(starts):
            end = min(start + chunk_tokens, flat_features.size(0))
            # 只把当前小块投影到词表，避免一次生成很大的 logits。
            logits = projection(flat_features[start:end])
            loss, _ = self.forward(logits, flat_target[start:end])
            yield loss, total_tokens, i == len(starts) - 1

    def forward_from_features(self, features: torch.Tensor, target: torch.Tensor, projection: nn.Module, chunk_tokens: int = 256):
        # 验证阶段不需要逐块 backward，可以直接把每块损失累加起来。
        total_loss = features.sum() * 0.0
        total_tokens = None
        for loss, token_count, _ in self.feature_loss_chunks(features, target, projection, chunk_tokens):
            total_loss = total_loss + loss
            total_tokens = token_count
        return total_loss, total_tokens
