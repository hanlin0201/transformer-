import torch
import torch.nn.functional as F


class BeamHypothesis:
    """
    表示 beam search 中的一条候选翻译路径。

    tokens: 当前已经生成出来的 token id 序列
            例如 [BOS, 123, 456]
    score:  这条路径目前的累计 log 概率分数
            分数越高，说明模型越认为这条翻译可能性大
    ended:  这条路径是否已经生成了 EOS
            如果生成了 EOS，说明这条翻译已经结束
    """

    def __init__(self, tokens, score, ended=False):
        self.tokens = tokens
        self.score = score
        self.ended = ended


def length_penalty(length: int, alpha: float) -> float:
    """
    长度惩罚函数。

    beam search 会倾向于短句，因为短句累加的 log 概率项少，
    所以这里用 length penalty 修正分数，避免模型总是生成过短翻译。

    length: 当前候选句子的长度
    alpha:  长度惩罚强度，常见值是 0.6

    返回值越大，表示对长句的归一化修正越明显。
    """
    return ((5.0 + length) / 6.0) ** alpha


@torch.no_grad()
def beam_search(
    model,
    src_tokens,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    beam_size: int = 4,
    alpha: float = 0.6,
    max_len: int = 100,
):
    """
    使用 beam search 进行翻译生成。

    model: 已训练好的 Transformer 模型
    src_tokens: 源语言句子的 token id，例如英文句子
    bos_id: 目标句开始符号 BOS 的 id
    eos_id: 目标句结束符号 EOS 的 id
    pad_id: padding 的 id
    beam_size: 每一步保留多少条候选翻译路径
    alpha: length penalty 的参数
    max_len: 最多生成多少个 token，防止无限生成

    返回:
    最终选择出来的最佳翻译 token id 序列。
    """

    # 切换到评估模式。
    # 这样 dropout 等训练专用行为会关闭。
    model.eval()

    # 如果 src_tokens 是一维，例如 [12, 35, 98, 2]，
    # 就加一个 batch 维度，变成 [[12, 35, 98, 2]]。
    if src_tokens.dim() == 1:
        src_tokens = src_tokens.unsqueeze(0)

    # 这个 beam_search 实现一次只处理一句话。
    # 如果 batch size 不是 1，就报错。
    if src_tokens.size(0) != 1:
        raise ValueError("beam_search expects one source sentence at a time")

    # 先把源句送进 Encoder。
    # memory 是 Encoder 对源句的表示，后续 Decoder 每一步都会参考它。
    # src_mask 用来防止 attention 看 PAD。
    memory, src_mask = model.encode(src_tokens)

    # 初始 beam 只有一条路径：只包含 BOS。
    # score = 0.0，因为还没有生成任何真实词。
    # ended = False，表示还没结束。
    beams = [BeamHypothesis([bos_id], 0.0, False)]

    # finished 用来存已经生成 EOS 的完整候选翻译。
    finished = []

    # 最多生成 max_len - 1 步。
    # 因为初始已经有一个 BOS 了。
    for _ in range(max_len - 1):
        # candidates 保存这一轮扩展出来的所有新候选。
        candidates = []

        # active 表示还没结束的候选路径。
        # 已经遇到 EOS 的路径不需要继续生成。
        active = [b for b in beams if not b.ended]

        # 把已经结束的候选放进 finished。
        finished.extend([b for b in beams if b.ended])

        # 如果所有路径都结束了，就停止生成。
        if not active:
            break

        # 把当前所有 active beam 的 token 序列组成一个 batch。
        # 例如 active 里有 4 条候选：
        # [BOS, A]
        # [BOS, B]
        # [BOS, C]
        # [BOS, D]
        # tgt 就是形状 [4, 当前长度] 的张量。
        tgt = torch.tensor(
            [b.tokens for b in active],
            dtype=torch.long,
            device=src_tokens.device,
        )

        # Encoder 的 memory 原本只有一句源句，对应 batch size = 1。
        # 但现在有 len(active) 条候选翻译，所以要复制 memory。
        repeated_memory = memory.expand(len(active), -1, -1).contiguous()

        # src_mask 也同样复制，让每条候选都能 attend 到同一个源句。
        repeated_src_mask = src_mask.expand(len(active), -1, -1, -1).contiguous()

        # 用 Decoder 根据当前候选序列预测下一个词。
        # model.decode 输出形状大概是：
        # [active_beam数量, 当前目标长度, vocab_size]
        #
        # [:, -1, :] 表示只取最后一个位置的预测结果，
        # 因为我们现在只关心“下一个词”是什么。
        logits = model.decode(tgt, repeated_memory, repeated_src_mask)[:, -1, :]

        # 把 logits 转成 log 概率。
        # 用 log_softmax 是因为 beam search 要累加概率，
        # 累加 log 概率比直接乘概率更稳定。
        log_probs = F.log_softmax(logits, dim=-1)

        # 对每条 active beam，选出概率最高的 beam_size 个下一个词。
        #
        # top_scores: 这些词的 log 概率
        # top_ids:    这些词在词表里的 token id
        top_scores, top_ids = torch.topk(log_probs, beam_size, dim=-1)

        # 遍历每一条 active beam。
        for i, beam in enumerate(active):
            # 对当前 beam 的 top-k 个下一个词逐个扩展。
            for score, idx in zip(top_scores[i].tolist(), top_ids[i].tolist()):
                # 新候选路径 = 原来的 tokens + 新预测出来的 idx
                tokens = beam.tokens + [idx]

                # 如果新生成的词是 EOS，就说明这条路径结束了。
                ended = idx == eos_id

                # 新路径的总分 = 原路径累计分数 + 新词的 log 概率
                candidates.append(
                    BeamHypothesis(tokens, beam.score + score, ended)
                )

        # 把之前已经完成的翻译也加入 candidates，
        # 避免它们在后续排序中丢失。
        candidates.extend(finished)

        # 对所有候选路径排序。
        # 注意这里不是直接按 score 排，而是按：
        # score / length_penalty
        #
        # 这样可以缓解短句天然更占优势的问题。
        candidates.sort(
            key=lambda b: b.score / length_penalty(len(b.tokens), alpha),
            reverse=True,
        )

        # 只保留分数最高的 beam_size 条路径。
        beams = candidates[:beam_size]

        # 更新 finished：当前 beams 里面已经结束的路径。
        finished = [b for b in beams if b.ended]

        # 如果保留下来的 beam 全部都已经结束，
        # 就可以提前停止，不用继续生成。
        if len(finished) >= beam_size:
            break

    # 如果有已经完成的翻译，就从 finished 里选；
    # 如果没有任何路径生成 EOS，就从当前 beams 里选。
    final = finished if finished else beams

    # 最后再按长度惩罚后的分数排序。
    final.sort(
        key=lambda b: b.score / length_penalty(len(b.tokens), alpha),
        reverse=True,
    )

    # 返回最佳候选翻译的 token id 序列。
    return final[0].tokens