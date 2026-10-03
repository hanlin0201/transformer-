import torch


def _pad(sequences, pad_id):
    """把同一个 batch 里的不同长度序列补齐到一样长。"""
    max_len = max(len(seq) for seq in sequences)
    out = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        out[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return out


def collate_translation_batch(samples, pad_id: int):
    """把若干条样本拼成训练用 batch。"""
    src = _pad([s["src_ids"] for s in samples], pad_id)
    tgt = _pad([s["tgt_ids"] for s in samples], pad_id)
    # 目标句右移一位：输入给 Decoder 的是不含最后 EOS 的序列，
    # 需要预测的是不含开头 BOS 的序列。
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]
    # 记录真实目标 token 数，后面用它做 loss 平均，避免 PAD 影响统计。
    num_tgt_tokens = int((tgt_output != pad_id).sum().item())
    return {
        "src": src,
        "tgt_input": tgt_input,
        "tgt_output": tgt_output,
        "num_tgt_tokens": num_tgt_tokens,
    }
