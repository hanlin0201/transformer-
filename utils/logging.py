import json
from pathlib import Path


class JsonlLogger:
    """把训练日志一行一条写成 JSONL，方便后续分析。"""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record):
        # ensure_ascii=False 可以让中文等字符直接写入文件，阅读更方便。
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
