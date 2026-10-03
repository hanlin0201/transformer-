from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer, scheduler, update_step, cfg):
    """保存模型、优化器、学习率调度器和配置，便于之后恢复训练。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "update_step": update_step,
        "config": cfg,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    """加载 checkpoint；如果传入优化器/调度器，也一起恢复它们的状态。"""
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


def prune_checkpoints(directory, keep_last: int):
    """只保留最近 keep_last 个按 step 命名的 checkpoint。"""
    if keep_last is None or keep_last <= 0:
        return
    paths = sorted(Path(directory).glob("checkpoint_step_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    for path in paths[:-keep_last]:
        path.unlink(missing_ok=True)
