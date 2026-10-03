import os
import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False):
    """统一设置常见随机种子，尽量让实验结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic=True 更利于复现；False 通常速度更快。
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
