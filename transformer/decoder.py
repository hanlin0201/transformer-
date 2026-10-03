import math
import torch.nn as nn
from .modules import MultiHeadAttention, PositionwiseFeedForward, SinusoidalPositionalEncoding


class DecoderLayer(nn.Module):
    """Decoder 的一层：目标句自注意力、看 Encoder 的注意力、前馈网络。"""

    def __init__(self, d_model, d_ff, num_heads, dropout, attention_dropout, norm_first=False):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, attention_dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, attention_dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x, memory, tgt_mask, src_mask):
        #memory是encoder输出的源句表示，tgt_mask是目标句mask，src_mask是源句mask
        # norm_first=True 时，每个子模块先归一化再计算。
        if self.norm_first:
            y = self.norm1(x)
            x = x + self.dropout(self.self_attn(y, y, y, tgt_mask)[0])
            x = x + self.dropout(self.cross_attn(self.norm2(x), memory, memory, src_mask)[0])
            x = x + self.dropout(self.ffn(self.norm3(x)))
            return x
        # 第一段看目标句已经生成的位置，tgt_mask 会挡住未来 token。
        attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 第二段看 Encoder 输出的源句表示，src_mask 会挡住源句 padding。
        attn_out, _ = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(attn_out))
        
        # 最后用前馈网络进一步加工每个位置的表示。
        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class Decoder(nn.Module):
    """完整 Decoder：目标词向量加位置编码，然后堆叠多层 DecoderLayer。"""

    def __init__(self, embedding, num_layers, d_model, d_ff, num_heads, dropout, attention_dropout, max_positions, scale_embedding=True, norm_first=False):
        super().__init__()
        self.embedding = embedding
        self.position = SinusoidalPositionalEncoding(d_model, max_positions)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_ff, num_heads, dropout, attention_dropout, norm_first)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model) if scale_embedding else 1.0

    def forward(self, tgt_tokens, memory, tgt_mask, src_mask):
        # 目标 token 先变成向量，并加上位置信息。
        x = self.embedding(tgt_tokens) * self.scale
        x = self.dropout(self.position(x))
        # 每一层都同时利用目标句上下文和 Encoder 的源句表示。
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return x
