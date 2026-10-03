class NoamScheduler:
    """原始 Transformer 论文里常用的学习率变化方式。

    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    这里的 step 指优化器真正更新的次数，不是 epoch，也不是单个 micro-batch。
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.set_step(0)

    def lr_at(self, step: int) -> float:
        # warmup 前逐步升高学习率，warmup 后按步数逐渐衰减。
        if step <= 0:
            return 0.0
        return (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))

    def set_step(self, step: int) -> float:
        # 根据指定 step 重新设置优化器里所有参数组的学习率。
        self.current_step = step
        lr = self.lr_at(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self):
        return {"current_step": self.current_step, "d_model": self.d_model, "warmup_steps": self.warmup_steps}

    def load_state_dict(self, state):
        self.current_step = int(state.get("current_step", 0))
        self.set_step(self.current_step)
