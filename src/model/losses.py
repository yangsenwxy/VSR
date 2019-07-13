import torch
import torch.nn as nn
from torchvision import models


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


class HuberLoss(nn.Module):
    """The implementation of the HuberLoss.
    Args:
        delta (float)
    """
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def forward(self, outputs, targets):
        abs_error = torch.abs(outputs - targets)
        delta = torch.ones_like(outputs) * self.delta
        quadratic = torch.min(abs_error, delta)
        linear = (abs_error - quadratic)
        losses = 0.5 * quadratic ** 2 + delta * linear
        return torch.mean(losses)


class PerceptualLoss(nn.Module):
    """Use VGG16 as the feature extrator.
    Args:
        requires_grad (boolean)
    """
    def __init__(self, requires_grad=False):
        super().__init__()
        self.vgg16 = VGG16(requires_grad)
        self.vgg16.eval()

    def forward(self, outputs, targets):
        if outputs.shape[1] != 3:
            # Need to extend the channel number to 3
            pred, gt = torch.repeat_interleave(outputs, 3, dim=1), torch.repeat_interleave(targets, 3, dim=1)
        else:
            pred, gt = outputs, targets

        pred_feat = self.vgg16(pred)
        gt_feat = self.vgg16(gt)
        return gt_feat - pred_feat


class VGG_OUTPUT(object):
    def __init__(self, relu1_2, relu2_2, relu3_3, relu4_3):
        self.__dict__ = locals()

    def __sub__(self, output):
        assert type(output) == type(self)

        diff_relu1_2 = self.relu1_2 - output.relu1_2
        diff_relu2_2 = self.relu2_2 - output.relu2_2
        diff_relu3_3 = self.relu3_3 - output.relu3_3
        diff_relu4_3 = self.relu4_3 - output.relu4_3
        diff = torch.mean(diff_relu1_2) + torch.mean(diff_relu2_2) + torch.mean(diff_relu3_3) + torch.mean(diff_relu4_3)
        return torch.abs(diff)


class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return VGG_OUTPUT(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
