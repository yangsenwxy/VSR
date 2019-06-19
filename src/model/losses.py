import torch
import torch.nn as nn


class MISRL1Loss(nn.Module):
    """The mean of the multple L1Loss for MISR task.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.l1_loss = nn.L1Loss(**kwargs)

    def forward(self, outputs, targets):
        losses = [self.l1_loss(output, target) for output, target in zip(outputs, targets)]
        loss = torch.stack(losses).mean()
        return loss
