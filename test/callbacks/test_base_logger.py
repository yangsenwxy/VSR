import torch
import pytest
from box import Box
from pathlib import Path

from src.callbacks.base_logger import BaseLogger
from test.model.test_net import MyNet

def test_base_logger():
    cfg = Box.from_yaml(filename=Path("test/configs/test_config.yaml"))
    net = MyNet(**cfg.net.kwargs)
    logger = BaseLogger(log_dir='./checkpoints/', 
                        net=net,
                        dummy_input=torch.randn((32, 3, 512, 512)))