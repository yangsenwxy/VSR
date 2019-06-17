import torch
import torch.nn as nn

from src.model.nets.base_net import BaseNet


class SRFBNet(BaseNet):
    """The implementation of super-resolution feedback network (SRFBN) with some modifications.

    First, the global residual skip connection do not perform upsampling and the feature maps are concatenated before the reconstruction block.
    Second, the model inputs are multiple different LR images (ref: https://arxiv.org/abs/1903.09814, https://github.com/Paper99/SRFBN_CVPR19/blob/master/networks/srfbn_arch.py).

    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
        num_features (int): The number of the internel feature maps.
        num_groups (int): The number of the projection groups in the feedback block.
        upscale_factor (int): The upscale factor (2, 3 ,4 or 8).
    """
    def __init__(self, in_channels, out_channels, num_features, num_groups, upscale_factor):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.num_groups = num_groups
        self.upscale_factor = upscale_factor
        self.lrf_block = _LRFBlock(in_channels, num_features) # The LR feature extraction block.
        self.f_block = _FBlock(num_features, num_groups, upscale_factor) # The feedback block.
        self.r_block = _RBlock(num_features, out_channels, upscale_factor) # The reconstruction block.

    def forward(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
            features = self.lrf_block(input)
            if i == 0:
                self.f_block.hidden_state = torch.zeros_like(features) # Reset the hidden state of the feedback block.
            features = self.f_block(features)
            self.f_block.hidden_state = features # Set the hidden state of the feedback block to the current output.
            features = input + features # The global residual skip connection.
            output = self.r_block(features)
            outputs.append(output)
        return outputs


class _LRFBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, 4 * out_channels, kernel_size=3, padding=1))
        self.add_module('prelu1', nn.PReLU(num_parameters=1, init=0.2))
        self.add_module('conv2', nn.Conv2d(4 * out_channels, out_channels, kernel_size=1))
        self.add_module('prelu2', nn.PReLU(num_parameters=1, init=0.2))


class _FBlock(nn.Module):
    def __init__(self, num_features, num_groups, upscale_factor):
        super().__init__()
        self.in_block = nn.Sequential()
        self.in_block.add_module('conv1', nn.Conv2d(num_features * 2, num_features, kernel_size=1))
        self.in_block.add_module('prelu1', nn.PReLU(num_parameters=1, init=0.2))

        self.up_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        if upscale_factor == 2:
            kernel_size, stride, padding = 6, 2, 2
        elif upscale_factor == 3:
            kernel_size, stride, padding = 7, 3, 2
        elif upscale_factor == 4:
            kernel_size, stride, padding = 8, 4, 2
        elif upscale_factor == 8:
            kernel_size, stride, padding = 12, 8, 2
        for i in range(num_groups):
            if i == 0:
                up_block = nn.Sequential()
                up_block.add_module('deconv1', nn.ConvTranspose2d(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding))
                up_block.add_module('prelu1', nn.PReLU(num_parameters=1, init=0.2))
                self.up_blocks.append(up_block)

                down_block = nn.Sequential()
                down_block.add_module('conv1', nn.Conv2d(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding))
                down_block.add_module('prelu1', nn.PReLU(num_parameters=1, init=0.2))
                self.down_blocks.append(down_block)
            else:
                up_block = nn.Sequential()
                up_block.add_module('conv1', nn.Conv2d(num_features * (i + 1), num_features, kernel_size=1))
                up_block.add_module('prelu1', nn.PReLU(num_parameters=1, init=0.2))
                up_block.add_module('deconv2', nn.ConvTranspose2d(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding))
                up_block.add_module('prelu2', nn.PReLU(num_parameters=1, init=0.2))
                self.up_blocks.append(up_block)

                down_block = nn.Sequential()
                down_block.add_module('conv1', nn.Conv2d(num_features * (i + 1), num_features, kernel_size=1))
                down_block.add_module('prelu1', nn.PReLU(num_parameters=1, init=0.2))
                down_block.add_module('conv2', nn.Conv2d(num_features, num_features, kernel_size=kernel_size, stride=stride, padding=padding))
                down_block.add_module('prelu2', nn.PReLU(num_parameters=1, init=0.2))
                self.down_blocks.append(down_block)

        self.out_block = nn.Sequential()
        self.out_block.add_module('conv1', nn.Conv2d(num_features * num_groups, num_features, kernel_size=1))
        self.out_block.add_module('prelu1', nn.PReLU(num_parameters=1, init=0.2))

        self._hidden_state = None

    @property
    def hidden_state(self):
        return self._hidden_state

    @hidden_state.setter
    def hidden_state(self, state):
        self._hidden_state = torch.empty_like(state).copy_(state)

    def forward(self, input):
        input = torch.cat([input, self.hidden_state], dim=1)
        features = self.in_block(input)

        lr_features_list, hr_features_list = [features], []
        for up_block, down_block in zip(self.up_blocks, self.down_blocks):
            concat_lr_features = torch.cat(lr_features_list, dim=1)
            hr_features = up_block(concat_lr_features)
            hr_features_list.append(hr_features)
            concat_hr_features = torch.cat(hr_features_list, dim=1)
            lr_features = down_block(concat_hr_features)
            lr_features_list.append(lr_features)

        features = torch.cat(lr_features_list[1:], dim=1)
        output = self.out_block(features)
        return output


class _RBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super().__init__()
        if upscale_factor == 2:
            kernel_size, stride, padding = 6, 2, 2
        elif upscale_factor == 3:
            kernel_size, stride, padding = 7, 3, 2
        elif upscale_factor == 4:
            kernel_size, stride, padding = 8, 4, 2
        elif upscale_factor == 8:
            kernel_size, stride, padding = 12, 8, 2
        self.add_module('deconv1', nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding))

        self.add_module('prelu1', nn.PReLU(num_parameters=1, init=0.2))
        self.add_module('conv2', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))