import argparse
from pathlib import Path
from datasets import load_dataset
from data.tokenizer import SharedSentencePieceTokenizer
from utils.config import load_config


def parse_direction(direction):
    """把 en-de 这样的方向字符串拆成源语言和目标语言。"""
    src, tgt = direction.split("-")
    return src, tgt


def encode_batch(batch, tokenizer, src_lang, tgt_lang, max_src_len, max_tgt_len):
    """把一批原始双语文本编码成 token id，并保留长度信息。"""
    src_ids = []
    tgt_ids = []
    src_texts = []
    tgt_texts = []
    for item in batch["translation"]:
        src_text = item[src_lang].strip()
        tgt_text = item[tgt_lang].strip()
        src = tokenizer.encode_src(src_text, max_src_len)
        tgt = tokenizer.encode_tgt(tgt_text, max_tgt_len)
        src_ids.append(src)
        tgt_ids.append(tgt)
        src_texts.append(src_text)
        tgt_texts.append(tgt_text)
    return {
        "src_ids": src_ids,
        "tgt_ids": tgt_ids,
        "src_len": [len(x) for x in src_ids],
        "tgt_len": [len(x) for x in tgt_ids],
        "src_text": src_texts,
        "tgt_text": tgt_texts,
    }


def keep_example(example, ratio):
    """过滤掉太短或源/目标长度比例过于悬殊的样本。"""
    if example["src_len"] <= 1 or example["tgt_len"] <= 2:
        return False
    if ratio is None:
        return True
    src = max(1, example["src_len"])
    tgt = max(1, example["tgt_len"])
    return (src / tgt <= ratio) and (tgt / src <= ratio)


def write_text_columns(dataset, out_dir: Path, name: str):
    """把验证/测试集的原文和参考译文写成普通文本，供翻译和 BLEU 使用。"""
    src_path = out_dir / f"{name}.src.txt"
    ref_path = out_dir / f"{name}.ref.txt"
    with src_path.open("w", encoding="utf-8") as src_f, ref_path.open("w", encoding="utf-8") as ref_f:
        for item in dataset.select_columns(["src_text", "tgt_text"]):
            src_f.write(item["src_text"].replace("\n", " ") + "\n")
            ref_f.write(item["tgt_text"].replace("\n", " ") + "\n")


def main():
    parser = argparse.ArgumentParser(description="用固定共享 SentencePiece 分词器准备 WMT14 英德数据。")
    parser.add_argument("--config", default="configs/base_wmt14_ende.yaml")
    parser.add_argument("--num-proc", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["dataset"]
    paths = cfg["paths"]
    out_dir = Path(paths["data_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = SharedSentencePieceTokenizer(paths["tokenizer_model"])
    src_lang, tgt_lang = parse_direction(data_cfg["direction"])

    # 配置里指定每个 split 对应 HuggingFace 数据集的哪一部分。
    split_map = {
        "train": data_cfg["train_split"],
        "valid": data_cfg["valid_split"],
        "test": data_cfg["test_split"],
    }
    for name, split in split_map.items():
        split_dir = out_dir / name
        if split_dir.exists() and not args.overwrite:
            print(f"跳过已存在的数据目录：{split_dir}")
            continue
        print(f"正在加载 {data_cfg['name']} {data_cfg['config']} split={split}")
        # 先下载/读取原始数据，再批量分词成 token id。
        raw = load_dataset(data_cfg["name"], data_cfg["config"], split=split, trust_remote_code=args.trust_remote_code)
        encoded = raw.map(
            lambda b: encode_batch(b, tokenizer, src_lang, tgt_lang, data_cfg["max_src_len"], data_cfg["max_tgt_len"]),
            batched=True,
            num_proc=args.num_proc,
            remove_columns=raw.column_names,
            desc=f"Encoding {name}",
        )
        ratio = data_cfg.get("length_ratio_filter")
        # 过滤异常样本；验证和测试集额外导出文本文件用于后续评估。
        encoded = encoded.filter(lambda ex: keep_example(ex, ratio), num_proc=args.num_proc, desc=f"Filtering {name}")
        if name in {"valid", "test"}:
            write_text_columns(encoded, out_dir, name)
        encoded = encoded.remove_columns(["src_text", "tgt_text"])
        encoded.save_to_disk(str(split_dir))
        print(f"已保存 {len(encoded)} 条样本到 {split_dir}")


if __name__ == "__main__":
    main()
