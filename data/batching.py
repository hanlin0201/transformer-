import random
from torch.utils.data import Sampler


class TokenBatchSampler(Sampler):
    """按 padding 后的 token 开销动态组 batch，思路类似 fairseq。

    当下面这个估计开销不超过 max_tokens 时，就继续往当前 micro-batch 里放样本：
        max(max_src_len, max_tgt_train_len) * batch_size <= max_tokens

    这样比只看目标句长度更稳，因为 Transformer 的显存主要受 batch 内补齐后的
    序列长度和注意力矩阵影响。
    """

    def __init__(self, src_lengths, tgt_lengths, max_tokens: int, shuffle: bool = True, sortish_chunk_size: int = 4096, seed: int = 42):
        self.src_lengths = list(src_lengths)
        self.tgt_lengths = list(tgt_lengths)
        if len(self.src_lengths) != len(self.tgt_lengths):
            raise ValueError("src_lengths and tgt_lengths must have the same length")
        self.lengths = [max(int(s), max(1, int(t) - 1)) for s, t in zip(self.src_lengths, self.tgt_lengths)]
        self.max_tokens = int(max_tokens)
        self.shuffle = shuffle
        self.sortish_chunk_size = int(sortish_chunk_size)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        indices = list(range(len(self.lengths)))
        if self.shuffle:
            # 先随机打乱，再在局部 chunk 内按长度排序，兼顾随机性和 padding 效率。
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(indices)
            chunks = [indices[i : i + self.sortish_chunk_size] for i in range(0, len(indices), self.sortish_chunk_size)]
            indices = []
            for chunk in chunks:
                chunk.sort(key=lambda i: self.lengths[i], reverse=True)
                indices.extend(chunk)
        else:
            indices.sort(key=lambda i: self.lengths[i], reverse=True)

        batch = []
        max_len = 0
        for idx in indices:
            # 用当前 batch 的最大长度估算加入新样本后的 token 开销。
            item_len = max(1, self.lengths[idx])
            candidate_max_len = max(max_len, item_len)
            candidate_cost = candidate_max_len * (len(batch) + 1)
            if batch and candidate_cost > self.max_tokens:
                yield batch
                batch = []
                max_len = 0
            batch.append(idx)
            max_len = max(max_len, item_len)
        if batch:
            yield batch

    def __len__(self):
        total = sum(self.lengths)
        return max(1, (total + self.max_tokens - 1) // self.max_tokens)
