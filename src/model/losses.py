import torch
import torch.nn as nn


class MISRL1Loss(nn.L1Loss):
    """The mean of the multple L1Loss for MISR task.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, outputs, targets):
        losses = [super().forward(output, target) for output, target in zip(outputs, targets)]
        loss = torch.stack(losses).mean()
        return loss
