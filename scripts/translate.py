import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from data.tokenizer import SharedSentencePieceTokenizer
from transformer import Transformer
from transformer.beam_search import beam_search
from utils.checkpoint import load_checkpoint
from utils.config import load_config


def load_model(cfg, ckpt_path, tokenizer, device):
    """根据配置创建模型，并加载指定 checkpoint。"""
    model = Transformer.from_config(cfg, vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id).to(device)
    load_checkpoint(ckpt_path, model, map_location=device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="使用 beam search 翻译文本。")
    parser.add_argument("--config", default="configs/base_wmt14_ende.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/averaged.pt")
    parser.add_argument("--input", default=None, help="源语言纯文本文件；默认使用准备好的 test.src.txt")
    parser.add_argument("--output", default="outputs/translations/newstest2014.hyp.txt")
    parser.add_argument("--beam-size", type=int, default=None)
    parser.add_argument("--length-penalty", type=float, default=None)
    args = parser.parse_args()

    # 读取配置、分词器和模型权重。
    cfg = load_config(args.config)
    tokenizer = SharedSentencePieceTokenizer(cfg["paths"]["tokenizer_model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.checkpoint, tokenizer, device)
    data_dir = Path(cfg["paths"]["data_dir"])
    input_path = Path(args.input) if args.input else data_dir / "test.src.txt"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    beam_size = args.beam_size or cfg["decode"]["beam_size"]
    alpha = args.length_penalty if args.length_penalty is not None else cfg["decode"]["length_penalty"]
    max_positions = cfg["model"]["max_positions"]
    max_extra = cfg["decode"].get("max_len_extra", 50)

    # 逐行读取源句，逐行写出模型翻译结果。
    with input_path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    with output_path.open("w", encoding="utf-8") as out:
        for line in tqdm(lines, desc="translate"):
            # 源句编码后送入 beam search，最大生成长度受模型位置编码上限约束。
            src_ids = tokenizer.encode_src(line, cfg["dataset"]["max_src_len"])
            src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
            max_len = min(max_positions, len(src_ids) + max_extra)
            hyp_ids = beam_search(
                model,
                src,
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                pad_id=tokenizer.pad_id,
                beam_size=beam_size,
                alpha=alpha,
                max_len=max_len,
            )
            out.write(tokenizer.decode(hyp_ids) + "\n")
    print(f"已保存翻译结果到 {output_path}")


if __name__ == "__main__":
    main()
