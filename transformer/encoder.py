import math
import torch.nn as nn
from .modules import MultiHeadAttention, PositionwiseFeedForward, SinusoidalPositionalEncoding


class EncoderLayer(nn.Module):
    """Encoder 的一层：先做自注意力，再做前馈网络。"""

    def __init__(self, d_model, d_ff, num_heads, dropout, attention_dropout, norm_first=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, attention_dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x, src_mask):
        # norm_first=True 时，把 LayerNorm 放在每个子模块前面，适合一些更深的模型。
        if self.norm_first:
            x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), src_mask)[0])
            x = x + self.dropout(self.ffn(self.norm2(x)))
            return x
        # 标准写法：子模块输出先和原输入相加（残差连接），再做 LayerNorm。
        attn_out, _ = self.self_attn(x, x, x, src_mask)#（query，key，value，mask）
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    """完整 Encoder：词向量加位置编码，然后堆叠多层 EncoderLayer。"""
#和教学视频区别：教学视频在外部就建好了各个模块，直接传入，这里传入参数，在内部建模块
    def __init__(self, embedding, num_layers, d_model, d_ff, num_heads, dropout, attention_dropout, max_positions, scale_embedding=True, norm_first=False):
        super().__init__()
        self.embedding = embedding
        self.position = SinusoidalPositionalEncoding(d_model, max_positions)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, num_heads, dropout, attention_dropout, norm_first)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model) if scale_embedding else 1.0

    def forward(self, src_tokens, src_mask):
        # 把 token id 转成向量，并加入位置信息。
        x = self.embedding(src_tokens) * self.scale
        x = self.dropout(self.position(x))
        # 逐层过每个 EncoderLayer，每层输出再喂给下一层，src_mask 会挡住 padding。
        for layer in self.layers:
            x = layer(x, src_mask)

        #返回最终的源句表示，形状 (B, L, d_model)
        return x
