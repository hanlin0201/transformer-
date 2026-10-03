from pathlib import Path
import yaml


def load_config(path):
    """读取 YAML 配置，并记录配置文件的绝对路径。"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg
