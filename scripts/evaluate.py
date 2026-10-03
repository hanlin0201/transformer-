import argparse
from pathlib import Path
import sacrebleu
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="计算 WMT14 英德翻译结果的 sacreBLEU。")
    parser.add_argument("--config", default="configs/base_wmt14_ende.yaml")
    parser.add_argument("--hyp", default="outputs/translations/newstest2014.hyp.txt")
    parser.add_argument("--ref", default=None, help="默认使用准备好的 test.ref.txt")
    parser.add_argument("--output", default="outputs/bleu/newstest2014.sacrebleu.txt")
    args = parser.parse_args()

    # 读取模型输出和参考译文，然后交给 sacreBLEU 统一打分。
    cfg = load_config(args.config)
    ref_path = Path(args.ref) if args.ref else Path(cfg["paths"]["data_dir"]) / "test.ref.txt"
    hyp_path = Path(args.hyp)
    hyps = hyp_path.read_text(encoding="utf-8").splitlines()
    refs = ref_path.read_text(encoding="utf-8").splitlines()
    bleu = sacrebleu.corpus_bleu(
        hyps,
        [refs],
        tokenize=cfg["eval"].get("sacrebleu_tokenize", "13a"),
        lowercase=cfg["eval"].get("lowercase", False),
    )
    # 同时保存 BLEU 分数和 signature，方便之后复现实验设置。
    text = f"{bleu}\nsignature = {bleu.signature}\n"
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
