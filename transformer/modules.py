import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """最基础的注意力计算：用 q 和 k 算相似度，再按权重汇总 v。"""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # q 和 k 相乘得到每个位置之间的相关程度，除以根号维度是为了让数值更稳定。
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        if mask is not None:
            # mask=False 的位置代表不能看，例如 padding 或目标句未来位置。
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        # 用注意力权重对 v 加权求和，得到每个位置的新表示。
        context = torch.matmul(attn, v)
        return context, attn


class MultiHeadAttention(nn.Module):
    """多头注意力：把同一份表示拆成多个头，各自关注不同关系后再合并。"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:#多头要能被除尽
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(dropout)
        #在 MultiHeadAttention 内部，创建一个 ScaledDotProductAttention 实例，并把 dropout 参数传给它。
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x):
        # 从 [batch, length, d_model] 变成[batch, length, heads, d_head]再变成 [batch, heads, length, d_head]。
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.d_head).transpose(1, 2)
    #让length, self.num_heads两个维度转置，这样每个head都能看到完整句子
    #关键在于后面做 attention 时，我们会这样用：
    # q, k, v 形状都是 (B, H, L, D)，scores = q @ k.transpose(-2, -1)   # (B, H, L, L)
    #如果没 transpose，形状是 (B, L, H, D)，最后两维是 (H, D)，没法直接对 L 做注意力。所以 transpose 的目的是把 L 放到倒数第二维，让矩阵乘法作用在 (L, D) 上。
    #PyTorch 的 @（或 torch.matmul）对高维张量的规则是：只对最后两维做矩阵乘法，前面的维度全部当作 batch 维，逐元素对应。
    
    def forward(self, query, key, value, mask=None):
        # 先分别投影成 q/k/v，再调用shape切割为多个 head 并行计算注意力。(q,k,v都调用query函数进行同样操作)
        q = self._shape(self.q_proj(query))
        k = self._shape(self.k_proj(key))
        v = self._shape(self.v_proj(value))

        #计算attention，得到每个位置的新表示和注意力权重
        context, attn = self.attention(q, k, v, mask)
        #context = attn @ v   # (B, H, L, D)
        #每个 head 里，每个 token 都根据注意力权重，融合了整句话的信息。


        # 把多个 head 的结果拼回 d_model 维度，再做一次输出投影。
        context = context.transpose(1, 2).contiguous().view(query.size(0), query.size(1), self.d_model)
        
        #transpose：(B, H, L, D)  →  (B, L, H, D)因为我们要把 H 和 D 合并 成 d_model = H × D。
        #而 view/reshape 合并的是相邻维度，所以必须先把 H 挪到 D 旁边。
        #contiguous() 把数据在内存里真正按新顺序排好。view 要求内存连续，否则会报错

        #view(query.size(0), query.size(1), self.d_model)query.size(0) = B，query.size(1) = L，self.d_model = H × D
        #(B, L, H, D)  →  (B, L, d_model)

        return self.out_proj(context), attn
        #attn是注意力权重，context是注意力加权后的输出，attn 对前向传播和训练 loss 没有直接作用，但它是可解释性、可视化、调试和某些高级训练技巧的必需品。


class PositionwiseFeedForward(nn.Module):
    """逐位置前馈网络：每个 token 单独经过两层线性变换。"""
    #FFN(x)=max(0,xW1+b1)W2+b2
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)#W1 and b1
        self.linear2 = nn.Linear(d_ff, d_model)#W2 and b2
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码：给模型补充 token 在句子中的位置信息。"""

    def __init__(self, d_model: int, max_positions: int = 512):
                        #向量维度      #句子长度
        super().__init__()
        # 预先生成固定的位置编码，训练时不会更新。
        #位置编码的公式实现，具体还存疑
        positions = torch.arange(max_positions, dtype=torch.float).unsqueeze(1)
        div_terms = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_positions, d_model)
        pe[:, 0::2] = torch.sin(positions * div_terms)
        pe[:, 1::2] = torch.cos(positions * div_terms)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):#给向量加上位置编码
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"sequence length {x.size(1)} exceeds max_positions {self.pe.size(1)}")
        return x + self.pe[:, : x.size(1)].to(dtype=x.dtype)
