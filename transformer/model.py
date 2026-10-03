import torch
import torch.nn as nn
from .decoder import Decoder
from .encoder import Encoder
from .initialization import initialize_transformer
from .masks import make_src_mask, make_tgt_mask


class Transformer(nn.Module):
    """把 Encoder、Decoder 和词表投影组合成完整的翻译模型。"""

    def __init__(
        self,
        vocab_size: int,#词表大小
        pad_id: int,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        d_model: int = 512,#向量维度
        d_ff: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        max_positions: int = 512,
        share_src_tgt_embedding: bool = True,
        share_tgt_embedding_projection: bool = True,
        scale_embedding: bool = True,
        norm_first: bool = False,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.d_model = d_model
        # 源语言和目标语言可以共用同一个词向量表，适合共享 SentencePiece 词表。
        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embedding = self.src_embedding if share_src_tgt_embedding else nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.encoder = Encoder(self.src_embedding, encoder_layers, d_model, d_ff, num_heads, dropout, attention_dropout, max_positions, scale_embedding, norm_first)
        self.decoder = Decoder(self.tgt_embedding, decoder_layers, d_model, d_ff, num_heads, dropout, attention_dropout, max_positions, scale_embedding, norm_first)
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        initialize_transformer(self, d_model)
        if share_tgt_embedding_projection:
            # 输出层权重和目标词向量共用，减少参数量，也常用于翻译模型。
            self.output_projection.weight = self.tgt_embedding.weight

    def encode(self, src_tokens):
        # Encoder 阶段只处理源句，并生成源句 mask 供 Decoder 复用。
        src_mask = make_src_mask(src_tokens, self.pad_id)
        memory = self.encoder(src_tokens, src_mask)
        #memory是encoder输出的源句表示，src_mask是源句mask
        return memory, src_mask

    def decode_features(self, tgt_tokens, memory, src_mask):
        # Decoder 阶段会额外构造目标句 mask，防止看到未来位置。
        tgt_mask = make_tgt_mask(tgt_tokens, self.pad_id)
        return self.decoder(tgt_tokens, memory, tgt_mask, src_mask)

    def decode(self, tgt_tokens, memory, src_mask):
        # 生成时常用：把 Decoder 特征直接转成词表 logits。
        dec = self.decode_features(tgt_tokens, memory, src_mask)
        return self.output_projection(dec)

    def forward_features(self, src_tokens, tgt_input_tokens):
        # 训练时先编码源句，再用右移后的目标句输入 Decoder。
        memory, src_mask = self.encode(src_tokens)
        return self.decode_features(tgt_input_tokens, memory, src_mask)

    def forward(self, src_tokens, tgt_input_tokens):
        # 返回每个目标位置上的词表预测分数。
        dec = self.forward_features(src_tokens, tgt_input_tokens)
        return self.output_projection(dec)

    @classmethod
    def from_config(cls, cfg, vocab_size: int, pad_id: int):
        # 从配置文件的 model 字段创建模型，训练和推理脚本都会用到。
        model_cfg = cfg["model"]
        return cls(vocab_size=vocab_size, pad_id=pad_id, **model_cfg)
