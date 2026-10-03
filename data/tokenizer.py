from pathlib import Path
import sentencepiece as spm


class SharedSentencePieceTokenizer:
    """源语言和目标语言共用的 SentencePiece 分词器包装。"""

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"SentencePiece model not found: {self.model_path}")
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(self.model_path))
        # 有些公开 SentencePiece 模型没有 PAD，这里就把词表末尾额外留给 PAD。
        self.original_vocab_size = self.sp.GetPieceSize()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id() if self.sp.bos_id() >= 0 else 1
        self.eos_id = self.sp.eos_id() if self.sp.eos_id() >= 0 else 2
        self.pad_id = self.sp.pad_id()
        if self.pad_id < 0:
            self.pad_id = self.original_vocab_size
            self.vocab_size = self.original_vocab_size + 1
        else:
            self.vocab_size = self.original_vocab_size

    def encode_src(self, text: str, max_len: int):
        """编码源句：截断后在末尾加 EOS。"""
        ids = self.sp.EncodeAsIds(text.strip())
        if max_len < 1:
            raise ValueError("max_len must be >= 1")
        ids = ids[: max_len - 1]
        ids.append(self.eos_id)
        return ids

    def encode_tgt(self, text: str, max_len: int):
        """编码目标句：开头加 BOS，末尾加 EOS。"""
        if max_len < 2:
            raise ValueError("target max_len must be >= 2 to fit BOS/EOS")
        ids = self.sp.EncodeAsIds(text.strip())[: max_len - 2]
        return [self.bos_id] + ids + [self.eos_id]

    def decode(self, ids):
        """把模型生成的 token id 转回文本，并跳过特殊符号。"""
        clean = []
        for idx in ids:
            idx = int(idx)
            if idx in (self.pad_id, self.bos_id, self.eos_id):
                continue
            if 0 <= idx < self.original_vocab_size:
                clean.append(idx)
        return self.sp.DecodeIds(clean)
