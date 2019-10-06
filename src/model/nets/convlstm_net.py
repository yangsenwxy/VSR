import torch
import torch.nn as nn
import math

from src.model.nets.base_net import BaseNet


class ConvLSTMNet(BaseNet):
    """
    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
        num_features (list of int): The number of the internel feature maps.
        bidirectional (bool):
        upscale_factor (int): The upscale factor (2, 3, 4 or 8).
    """
    def __init__(self, in_channels, out_channels, num_features, bidirectional, memory, fuse_hidden_states, upscale_factor):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.upscale_factor = upscale_factor

        if upscale_factor not in [2, 3, 4, 8]:
            raise ValueError(f'The upscale factor should be 2, 3, 4 or 8. Got {upscale_factor}.')

        num_feature = num_features[0]
        self.in_block = _InBlock(in_channels, num_feature)
        self.convlstm_block = _ConvLSTM(input_size=num_feature,
                                        hidden_sizes=num_features,
                                        kernel_size=3,
                                        num_layers=len(num_features),
                                        bidirectional=bidirectional,
                                        memory=memory,
                                        fuse_hidden_states=fuse_hidden_states)
        self.conv = nn.Conv2d(num_features[-1], num_feature, kernel_size=1)
        self.out_block = _OutBlock(num_feature, out_channels, upscale_factor)
        self.forward_twice = False

    def forward(self, inputs):
        in_features = torch.stack([self.in_block(input) for input in inputs], dim=0) # (T, B, C, H, W)
        if self.forward_twice:
            T = len(inputs)            
            #convlstm_features = self.convlstm_block(torch.cat([in_features[-2:], in_features], dim=0))[2:] 
            #30.6945, 0.8891, 0.0911
            #convlstm_features = self.convlstm_block(torch.cat([in_features[-2:], in_features, in_features[:2]], dim=0))[2:-2] 
            #30.6846, 0.8889, 0.0912
            convlstm_features = self.convlstm_block(torch.cat([in_features, in_features], dim=0))[T:]
            #30.6309, 0.8879, 0.0919
            #convlstm_features = self.convlstm_block(torch.cat([in_features, in_features, in_features], dim=0))[T:-T]
            #30.6159, 0.8876, 0.0921
            #30.5297, 0.8853, 0.0929 (no-memory)

        else:
            convlstm_features = self.convlstm_block(in_features)
        convlstm_features = [self.conv(convlstm_feature) for convlstm_feature in convlstm_features]
        outputs = [self.out_block(in_feature + convlstm_feature)
                   for in_feature, convlstm_feature in zip(in_features, convlstm_features)]
        return outputs


class _InBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('conv1', nn.Conv2d(in_channels, 4 * out_channels, kernel_size=3, padding=1))
        self.add_module('prelu1', nn.PReLU(num_parameters=1, init=0.2))
        self.add_module('conv2', nn.Conv2d(4 * out_channels, out_channels, kernel_size=1))
        self.add_module('prelu2', nn.PReLU(num_parameters=1, init=0.2))


class _OutBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, upscale_factor):
        super().__init__()
        if (math.log(upscale_factor, 2) % 1) == 0:
            for i in range(int(math.log(upscale_factor, 2))):
                self.add_module(f'conv{i+1}', nn.Conv2d(in_channels, 4 * in_channels, kernel_size=3, padding=1))
                self.add_module(f'pixelshuffle{i+1}', nn.PixelShuffle(2))
            self.add_module(f'conv{i+2}', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        elif upscale_factor == 3:
            self.add_module('conv1', nn.Conv2d(in_channels, 9 * in_channels, kernel_size=3, padding=1))
            self.add_module('pixelshuffle1', nn.PixelShuffle(3))
            self.add_module('conv2', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))


class _ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_sizes, kernel_size, num_layers,
                 bias=True, batch_first=False, bidirectional=False, memory=True, fuse_hidden_states=False):
        super().__init__()
        assert batch_first is False

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.memory = memory
        self.fuse_hidden_states = fuse_hidden_states

        self.cell_list = nn.ModuleList([nn.ModuleList() for _ in range(self.num_directions)])
        for i in range(self.num_directions):
            for layer in range(num_layers):
                if layer == 0:                    
                    in_channels = input_size + hidden_sizes[layer] if memory else 2 * input_size
                else:
                    in_channels = hidden_sizes[layer - 1] + hidden_sizes[layer] if memory else 2 * hidden_sizes[layer - 1]
                self.cell_list[i].append(_ConvLSTMCell(in_channels=in_channels,
                                                       hidden_size=hidden_sizes[layer],
                                                       kernel_size=kernel_size,
                                                       bias=bias,
                                                       memory=memory))
        
        if fuse_hidden_states:
            # use C3D to fuse the hidden states
            self.conv_list = nn.ModuleList()
            for layer in range(num_layers):
                self.conv_list.append(nn.Sequential(nn.Conv3d(hidden_sizes[layer], 
                                                              hidden_sizes[layer], 
                                                              kernel_size=kernel_size,
                                                              padding=kernel_size//2),
                                                    nn.PReLU(num_parameters=1, init=0.2),
                                                    nn.Conv3d(hidden_sizes[layer], 
                                                              hidden_sizes[layer], 
                                                              kernel_size=kernel_size,
                                                              padding=kernel_size//2),
                                                    nn.PReLU(num_parameters=1, init=0.2),
                                                    nn.Conv3d(hidden_sizes[layer], 
                                                              hidden_sizes[layer], 
                                                              kernel_size=kernel_size,
                                                              padding=kernel_size//2)))

        
    def forward(self, input, hidden_states=None):
        if hidden_states is None:
            hidden_states = self._init_hidden_states(input)

        seq_len = input.size(0)
        cur_layer_input = input
        for layer in range(self.num_layers):
            h_t, c_t = hidden_states[0][layer]
            output_inner = []
            for t in range(seq_len):
                h_t, c_t = self.cell_list[0][layer](cur_layer_input[t], h_t, c_t)
                output_inner.append(h_t)

            if self.bidirectional:
                h_t, c_t = hidden_states[1][layer]
                reversed_output_inner = []
                for t in reversed(range(seq_len)):
                    h_t, c_t = self.cell_list[1][layer](cur_layer_input[t], h_t, c_t)
                    reversed_output_inner.insert(0, h_t)
                output_inner = [(torch.cat([h, rh], dim=1))
                                for h, rh in zip(output_inner, reversed_output_inner)]
                layer_output = torch.stack(output_inner, dim=0) # (T, B, C, H, W)
            else:
                if self.fuse_hidden_states:
                    features = torch.stack(output_inner, dim=2) # (B, C, T, H, W)
                    layer_output = self.conv_list[layer](features).permute(2, 0, 1, 3, 4).contiguous() # (T, B, C, H, W)
                else:
                    layer_output = torch.stack(output_inner, dim=0)
                                      
            cur_layer_input = layer_output
        return layer_output

    def _init_hidden_states(self, input):
        _, B, _, H, W = input.size()
        _states = []
        for layer in range(self.num_layers):
            C = self.cell_list[0][layer].hidden_size
            zeros = torch.zeros(B, C, H, W, dtype=input.dtype, device=input.device)
            _states.append((zeros, zeros))
        states = [_states for _ in range(self.num_directions)]
        return states

class _ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_size, kernel_size, bias=True, memory=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.memory = memory
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=4 * hidden_size,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2,
                              bias=bias)

    def forward(self, input, h_0, c_0):
        if self.memory:
            features = self.conv(torch.cat([input, h_0], dim=1))
        else:
            features = self.conv(torch.cat([input, input], dim=1))
        cc_i, cc_f, cc_o, cc_g = torch.split(features, self.hidden_size, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        g = torch.tanh(cc_g)
        o = torch.sigmoid(cc_o)
        c_1 = f * c_0 + i * g
        h_1 = o * torch.tanh(c_1)
        return h_1, c_1
