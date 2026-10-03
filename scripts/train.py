import argparse
import math
from contextlib import nullcontext
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.batching import TokenBatchSampler
from data.collate import collate_translation_batch
from data.dataset import TranslationDataset
from data.tokenizer import SharedSentencePieceTokenizer
from transformer import Transformer
from transformer.loss import LabelSmoothingLoss
from transformer.scheduler import NoamScheduler
from utils.checkpoint import save_checkpoint, prune_checkpoints, load_checkpoint
from utils.config import load_config
from utils.logging import JsonlLogger
from utils.seed import set_seed


def amp_context(device, precision):
    """根据设备和精度选择自动混合精度上下文。"""
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    raise ValueError(f"unknown precision: {precision}")


def make_grad_scaler(device, precision):
    """fp16 训练时启用梯度缩放，降低数值下溢风险。"""
    enabled = device.type == "cuda" and precision == "fp16"
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def move_batch(batch, device):
    """把一个 batch 移到训练设备上。"""
    return {
        "src": batch["src"].to(device, non_blocking=True),
        "tgt_input": batch["tgt_input"].to(device, non_blocking=True),
        "tgt_output": batch["tgt_output"].to(device, non_blocking=True),
        "num_tgt_tokens": batch["num_tgt_tokens"],
    }


@torch.no_grad()
def evaluate_loss(model, criterion, loader, device, precision, loss_chunk_tokens):
    """在验证集上只计算 loss，不更新参数。"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with amp_context(device, precision):
            features = model.forward_features(batch["src"], batch["tgt_input"])
            loss_sum, token_count = criterion.forward_from_features(features, batch["tgt_output"], model.output_projection, loss_chunk_tokens)
        total_loss += float(loss_sum.item())
        total_tokens += int(token_count.item())
    model.train()
    if total_tokens == 0:
        return math.inf
    return total_loss / total_tokens


def run_update(model, criterion, optimizer, scheduler, scaler, micro_batches, device, precision, update_step, grad_clip, loss_chunk_tokens):
    """完成一次优化器更新，内部可以累积多个 micro-batch 的梯度。"""
    total_tokens = sum(int(b["num_tgt_tokens"]) for b in micro_batches)
    if total_tokens <= 0:
        return update_step, True, 0.0, 0

    optimizer.zero_grad(set_to_none=True)
    loss_sum_for_log = 0.0
    for batch in micro_batches:
        batch = move_batch(batch, device)
        with amp_context(device, precision):
            features = model.forward_features(batch["src"], batch["tgt_input"])
            chunk_iter = criterion.feature_loss_chunks(features, batch["tgt_output"], model.output_projection, loss_chunk_tokens)
            for chunk_loss_sum, _token_count, is_last_chunk in chunk_iter:
                # 用本次更新中的真实 token 总数做平均，保证梯度累积时尺度一致。
                loss = chunk_loss_sum / total_tokens
                if not torch.isfinite(loss.detach()):
                    optimizer.zero_grad(set_to_none=True)
                    raise RuntimeError("non-finite loss detected; stop instead of advancing Noam step")
                loss_sum_for_log += float(chunk_loss_sum.detach().item())
                retain_graph = not is_last_chunk
                if scaler.is_enabled():
                    scaler.scale(loss).backward(retain_graph=retain_graph)
                else:
                    loss.backward(retain_graph=retain_graph)
        del features

    proposed_step = update_step + 1
    # Noam 学习率和优化器更新次数绑定；如果 fp16 溢出跳步，下面会回滚 step。
    lr = scheduler.set_step(proposed_step)
    skipped = False
    if scaler.is_enabled():
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            skipped = True
            scheduler.set_step(update_step)
        else:
            update_step = proposed_step
    else:
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        update_step = proposed_step

    optimizer.zero_grad(set_to_none=True)
    return update_step, skipped, loss_sum_for_log / total_tokens, total_tokens


def make_loader(dataset_path, pad_id, max_tokens, shuffle, seed, num_workers):
    """创建带动态 token batch 的 DataLoader。"""
    dataset = TranslationDataset(dataset_path)
    sampler = TokenBatchSampler(dataset.src_lengths, dataset.tgt_lengths, max_tokens=max_tokens, shuffle=shuffle, seed=seed)
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=partial(collate_translation_batch, pad_id=pad_id),
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, sampler


def main():
    parser = argparse.ArgumentParser(description="从零训练一个 WMT14 英德 Transformer。")
    parser.add_argument("--config", default="configs/base_wmt14_ende.yaml")
    parser.add_argument("--resume", default=None, help="用于恢复模型/优化器/调度器状态的检查点路径")
    args = parser.parse_args()

    # 读取配置并确定训练设备、随机种子和数值精度。
    cfg = load_config(args.config)
    set_seed(cfg["seed"], deterministic=cfg["train"].get("deterministic", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = cfg["train"].get("precision", "bf16")
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        print("当前 GPU 不支持 bf16，自动改用 fp16")
        precision = "fp16"

    # 初始化分词器、模型、损失函数、优化器和学习率调度器。
    tokenizer = SharedSentencePieceTokenizer(cfg["paths"]["tokenizer_model"])
    model = Transformer.from_config(cfg, vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id).to(device)
    criterion = LabelSmoothingLoss(tokenizer.vocab_size, tokenizer.pad_id, cfg["train"]["label_smoothing"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0,
        betas=(cfg["train"]["beta1"], cfg["train"]["beta2"]),
        eps=float(cfg["train"]["eps"]),
    )
    scheduler = NoamScheduler(optimizer, cfg["model"]["d_model"], cfg["train"]["warmup_steps"])
    scaler = make_grad_scaler(device, precision)

    # 训练集和验证集都用 token 数控制 batch 大小，避免长句 batch 占用过多显存。
    data_dir = Path(cfg["paths"]["data_dir"])
    train_loader, train_sampler = make_loader(
        data_dir / "train",
        tokenizer.pad_id,
        cfg["train"]["tokens_per_micro_batch_target"],
        True,
        cfg["seed"],
        cfg["train"].get("num_workers", 0),
    )
    valid_loader, _ = make_loader(
        data_dir / "valid",
        tokenizer.pad_id,
        cfg["train"]["tokens_per_micro_batch_target"] * 2,
        False,
        cfg["seed"],
        cfg["train"].get("num_workers", 0),
    )

    # 输出目录、日志器和训练过程中的常用超参数。
    output_dir = Path(cfg["paths"]["output_dir"])
    ckpt_dir = output_dir / "checkpoints"
    logger = JsonlLogger(output_dir / "logs" / "train.jsonl")
    max_updates = cfg["train"]["max_update_steps"]
    grad_accum_steps = cfg["train"]["grad_accum_steps"]
    grad_clip = cfg["train"].get("grad_clip")
    loss_chunk_tokens = cfg["train"].get("loss_chunk_tokens", 256)
    update_step = 0
    if args.resume:
        # 断点恢复会同时加载模型、优化器和调度器状态。
        ckpt = load_checkpoint(args.resume, model, optimizer=optimizer, scheduler=scheduler, map_location=device)
        update_step = int(ckpt.get("update_step", scheduler.current_step))
        scheduler.set_step(update_step)
        print(f"已从 {args.resume} 恢复训练，update_step={update_step}")
    epoch = 0
    model.train()

    pbar = tqdm(total=max_updates, initial=update_step, desc="updates")
    while update_step < max_updates:
        train_sampler.set_epoch(epoch)
        pending = []
        for batch in train_loader:
            # pending 收集多个 micro-batch，凑够 grad_accum_steps 后再更新一次参数。
            pending.append(batch)
            if len(pending) < grad_accum_steps:
                continue
            update_step, skipped, loss, tokens = run_update(
                model, criterion, optimizer, scheduler, scaler, pending, device, precision, update_step, grad_clip, loss_chunk_tokens
            )
            pending = []
            if skipped:
                logger.write({"event": "amp_skip", "step": update_step})
                continue
            pbar.update(1)
            # 定期写训练日志、跑验证集、保存 checkpoint。
            if update_step % cfg["train"]["log_every_steps"] == 0:
                lr = optimizer.param_groups[0]["lr"]
                record = {"step": update_step, "loss": loss, "ppl": math.exp(min(loss, 20)), "tokens": tokens, "lr": lr}
                logger.write(record)
                pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{lr:.2e}")
            if update_step % cfg["train"]["validate_every_steps"] == 0:
                val_loss = evaluate_loss(model, criterion, valid_loader, device, precision, loss_chunk_tokens)
                logger.write({"step": update_step, "valid_loss": val_loss, "valid_ppl": math.exp(min(val_loss, 20))})
            if update_step % cfg["train"]["save_every_steps"] == 0:
                save_checkpoint(ckpt_dir / f"checkpoint_step_{update_step}.pt", model, optimizer, scheduler, update_step, cfg)
                save_checkpoint(ckpt_dir / "checkpoint_last.pt", model, optimizer, scheduler, update_step, cfg)
                prune_checkpoints(ckpt_dir, cfg["train"].get("keep_last_checkpoints", 10))
            if update_step >= max_updates:
                break
        if pending and update_step < max_updates:
            # 一个 epoch 末尾剩下的 micro-batch 也要更新掉。
            update_step, skipped, loss, tokens = run_update(
                model, criterion, optimizer, scheduler, scaler, pending, device, precision, update_step, grad_clip, loss_chunk_tokens
            )
            if not skipped:
                pbar.update(1)
        epoch += 1

    save_checkpoint(ckpt_dir / "checkpoint_final.pt", model, optimizer, scheduler, update_step, cfg)
    pbar.close()
    print(f"训练结束，update_step={update_step}")


if __name__ == "__main__":
    main()






