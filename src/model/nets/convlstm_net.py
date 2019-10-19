import torch
import torch.nn as nn
import math
import copy

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
    def __init__(self, in_channels, out_channels, num_features, num_stages, upscale_factor,
                 bidirectional=False, memory=True, updated_memory=False, positional_encoding=False):
        super().__init__()
        assert num_features[0] == num_features[-1]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.num_stages = num_stages
        self.bidirectional = bidirectional
        self.upscale_factor = upscale_factor
        self.memory = memory
        self.updated_memory = updated_memory

        if upscale_factor not in [2, 3, 4, 8]:
            raise ValueError(f'The upscale factor should be 2, 3, 4 or 8. Got {upscale_factor}.')

        self.in_block = _InBlock(in_channels, num_features[0])
        self.convlstm_block = _ConvLSTM(input_size=num_features[0],
                                        hidden_sizes=num_features,
                                        kernel_size=3,
                                        num_layers=len(num_features),
                                        bidirectional=bidirectional,
                                        memory=memory)
        self.out_block = _OutBlock(num_features[0], out_channels, upscale_factor)
        self.positional_encoding = _PositionalEncoding() if positional_encoding else None

    def forward(self, inputs, pos_code, forward_input=None, backward_input=None):
        if self.updated_memory:
            if not self.bidirectional:
                backward_input = None
            with torch.no_grad():
                states = self.update_memory(forward_input, backward_input)
        else:
            states = None

        outputs = []
        in_features = torch.stack([self.in_block(input) for input in inputs], dim=0)
        for i in range(self.num_stages):
            if self.positional_encoding:
                encoded_features = self.positional_encoding(in_features, pos_code)
                convlstm_features = self.convlstm_block(encoded_features, states)
            else:
                convlstm_features = self.convlstm_block(in_features, states)
            features = [in_feature + convlstm_feature
                        for in_feature, convlstm_feature in zip(in_features, convlstm_features)]
            outputs.append([self.out_block(feature) for feature in features])
            in_features = torch.stack(features, dim=0)
        return outputs


class _InBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        self.add_module('prelu', nn.PReLU(num_parameters=1, init=0.2))


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
    def __init__(self, input_size, hidden_sizes, kernel_size, num_layers, bidirectional=False, memory=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.memory = memory

        self.forward_cells = nn.ModuleList()
        for layer in range(num_layers):
            if layer == 0:
                in_channels = input_size + hidden_sizes[layer]
            else:
                in_channels = hidden_sizes[layer - 1] + hidden_sizes[layer]
            self.forward_cells.append(_ConvLSTMCell(in_channels=in_channels,
                                                    hidden_size=hidden_sizes[layer],
                                                    kernel_size=kernel_size))
        if bidirectional:
            self.backward_cells = copy.deepcopy(self.forward_cells)
            self.fuser = _Fuser(in_channels=hidden_sizes[-1] * 2, out_channels=hidden_sizes[-1])

    def forward(self, input, states=None):
        if states is None:
            forward_states, backward_states = self._init_states(input)
        else:
            forward_states, backward_states = states

        T = input.size(0)
        _input = input
        for layer in range(self.num_layers):
            h_t, c_t = forward_states[layer]
            hidden_states = []
            for t in range(T):
                if not self.memory:
                    h_t, c_t = forward_states[layer]
                h_t, c_t = self.forward_cells[layer](_input[t], h_t, c_t)
                hidden_states.append(h_t)
            _input = torch.stack(hidden_states, dim=0)
        forward_hidden_states = _input

        if self.bidirectional:
            _input = input
            for layer in range(self.num_layers):
                h_t, c_t = backward_states[layer]
                hidden_states = []
                for t in reversed(range(T)):
                    if not self.memory:
                        h_t, c_t = backward_states[layer]
                    h_t, c_t = self.backward_cells[layer](_input[t], h_t, c_t)
                    hidden_states.insert(0, h_t)
                _input = torch.stack(hidden_states, dim=0)
            backward_hidden_states = _input
            fused_hidden_states = self.fuser(forward_hidden_states, backward_hidden_states)
            return fused_hidden_states
        else:
            return forward_hidden_states

    def update_memory(self, forward_input, backward_input=None):
        forward_states, backward_states = self._init_states(forward_input)

        T = forward_input.size(0)
        for layer in range(self.num_layers):
            h_t, c_t = forward_states[layer]
            hidden_states = []
            for t in range(T):
                h_t, c_t = self.forward_cells[layer](forward_input[t], h_t, c_t)
                hidden_states.append(h_t)
            forward_states[layer] = (h_t, c_t)
            forward_input = torch.stack(hidden_states, dim=0)

        if backward_input:
            for layer in range(self.num_layers):
                h_t, c_t = backward_states[layer]
                hidden_states = []
                for t in reversed(range(T)):
                    h_t, c_t = self.backward_cells[layer](backward_input[t], h_t, c_t)
                    hidden_states.insert(0, h_t)
                backward_states[layer] = (h_t, c_t)
                backward_input = torch.stack(hidden_states, dim=0)
            return forward_states, backward_states
        else:
            return forward_states, None

    def _init_states(self, input):
        _, B, _, H, W = input.size()
        states = []
        for layer in range(self.num_layers):
            C = self.forward_cells[layer].hidden_size
            zeros = torch.zeros(B, C, H, W, dtype=input.dtype, device=input.device)
            states.append((zeros, zeros))
        if self.bidirectional:
            return states, states
        else:
            return states, None


class _ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_size, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=4 * hidden_size,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2)

    def forward(self, input, h_0, c_0):
        features = self.conv(torch.cat([input, h_0], dim=1))
        cc_i, cc_f, cc_o, cc_g = torch.split(features, self.hidden_size, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        g = torch.tanh(cc_g)
        o = torch.sigmoid(cc_o)
        c_1 = f * c_0 + i * g
        h_1 = o * torch.tanh(c_1)
        return h_1, c_1


class _Fuser(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bidirectional_fuser = nn.Conv2d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=1)

    def forward(self, forward_hidden_states, backward_hidden_states):
        fused_hidden_states = torch.stack([self.bidirectional_fuser(torch.cat([fh_t, bh_t], dim=1))
                                           for fh_t, bh_t in zip(forward_hidden_states, backward_hidden_states)], dim=0)
        return fused_hidden_states


class _PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, pos_code):
        _, _, C, H, W = input.size()
        pos_code = pos_code.repeat(C, H, W, 1, 1).permute(4, 3, 0, 1, 2).contiguous()
        return input + pos_code
