from pathlib import Path
from datasets import load_from_disk


class TranslationDataset:
    """从磁盘加载已经分词好的翻译数据集。"""

    def __init__(self, path):
        self.path = Path(path)
        self.dataset = load_from_disk(str(self.path))
        # 这两个长度会交给 sampler，用来按 token 数动态组 batch。
        self.src_lengths = list(self.dataset["src_len"])
        self.tgt_lengths = list(self.dataset["tgt_len"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 训练时只需要 token id，原始文本主要留给验证/测试输出使用。
        item = self.dataset[int(idx)]
        return {"src_ids": item["src_ids"], "tgt_ids": item["tgt_ids"]}
