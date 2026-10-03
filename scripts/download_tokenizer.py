import argparse
from pathlib import Path
from urllib.request import urlretrieve

DEFAULT_URL = "https://huggingface.co/malloc/OpenNMT-py-English-German-Transformer/resolve/main/sentencepiece.model"


def main():
    parser = argparse.ArgumentParser(description="下载公开的 OpenNMT WMT14 英德共享 SentencePiece 模型。")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", default="assets/tokenizer/wmt14_ende_sp.model")
    args = parser.parse_args()
    # 下载到指定位置，后续 prepare_wmt14.py 和训练脚本会读取这个模型。
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"正在从 {args.url} 下载分词器")
    urlretrieve(args.url, out)
    print(f"已保存到 {out}")


if __name__ == "__main__":
    main()
