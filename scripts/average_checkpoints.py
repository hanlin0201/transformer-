import argparse
from pathlib import Path
import torch


def checkpoint_step(path: Path):
    """从 checkpoint_step_123.pt 这样的文件名里取出步数。"""
    return int(path.stem.split("_")[-1])


def main():
    parser = argparse.ArgumentParser(description="平均最后 N 个 Transformer 检查点。")
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--last", type=int, default=5)
    parser.add_argument("--output", default="outputs/checkpoints/averaged.pt")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    paths = sorted(ckpt_dir.glob("checkpoint_step_*.pt"), key=checkpoint_step)[-args.last :]
    if not paths:
        raise FileNotFoundError(f"no checkpoint_step_*.pt found in {ckpt_dir}")
    print("正在平均以下检查点：")
    for path in paths:
        print(f"  {path}")

    # 逐个读取模型权重并累加，最后除以 checkpoint 数量。
    avg_state = None
    first = None
    for path in paths:
        ckpt = torch.load(path, map_location="cpu")
        first = first or ckpt
        state = ckpt["model"]
        if avg_state is None:
            avg_state = {k: v.clone().float() for k, v in state.items()}
        else:
            for k, v in state.items():
                avg_state[k] += v.float()
    for k in avg_state:
        avg_state[k] /= len(paths)
    # 保留第一个 checkpoint 的其它字段，只替换 model 权重。
    first["model"] = avg_state
    first["averaged_from"] = [str(p) for p in paths]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(first, out)
    print(f"已保存平均后的检查点到 {out}")


if __name__ == "__main__":
    main()
