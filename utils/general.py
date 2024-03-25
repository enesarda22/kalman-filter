from pathlib import Path

import torch


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_checkpoint(path, **kwargs):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    torch.save({**kwargs}, p)
