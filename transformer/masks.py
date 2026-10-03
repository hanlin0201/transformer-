import torch


def make_src_mask(src_tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """构造源句 mask，True 表示 attention 可以读取的位置。"""
    # 形状扩展成 [batch, 1, 1, src_len]，方便和多头注意力分数广播相乘。
    return (src_tokens != pad_id).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt_tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """构造目标句 mask，同时挡住 padding 和未来 token。"""
    # pad_mask 负责挡住 PAD；causal 负责保证当前位置只能看见自己和左边。
    pad_mask = (tgt_tokens != pad_id).unsqueeze(1).unsqueeze(2)
    length = tgt_tokens.size(1)
    causal = torch.tril(torch.ones((length, length), device=tgt_tokens.device, dtype=torch.bool))
    return pad_mask & causal.unsqueeze(0).unsqueeze(0)
