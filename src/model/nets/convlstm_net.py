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
        self.positional_encoding = positional_encoding

        if upscale_factor not in [2, 3, 4, 8]:
            raise ValueError(f'The upscale factor should be 2, 3, 4 or 8. Got {upscale_factor}.')

        self.in_block = _InBlock(in_channels, num_features[0])
        self.convlstm_block = _ConvLSTM(input_size=num_features[0],
                                        hidden_sizes=num_features,
                                        kernel_size=3,
                                        num_layers=len(num_features),
                                        bidirectional=bidirectional,
                                        memory=memory,
                                        positional_encoding=positional_encoding)
        self.out_block = _OutBlock(num_features[0], out_channels, upscale_factor)

    def forward(self, inputs, forward_inputs, backward_inputs, pos_code):
        outputs = []
        in_features = torch.stack([self.in_block(input) for input in forward_inputs[-2:]+inputs+backward_inputs[:2]], dim=0)
        forward_features = torch.stack([self.in_block(forward_input)
                                        for forward_input in forward_inputs[:-2]], dim=0)
        if self.bidirectional:
            backward_features = torch.stack([self.in_block(backward_input)
                                             for backward_input in backward_inputs[2:]], dim=0)
        else:
            backward_features = None
        
        for i in range(self.num_stages):
            if self.updated_memory:
                with torch.no_grad():
                    convlstm_features, states = self.convlstm_block.update_memory(forward_features, backward_features)
                    forward_convlstm_features, backward_convlstm_features = convlstm_features
                    forward_features = torch.stack([forward_feature + forward_convlstm_feature
                                                    for forward_feature, forward_convlstm_feature \
                                                    in zip(forward_features, forward_convlstm_features)], dim=0)
                    if self.bidirectional:
                        backward_features = torch.stack([backward_feature + backward_convlstm_feature
                                                         for backward_feature, backward_convlstm_feature \
                                                         in zip(backward_features, backward_convlstm_features)], dim=0)
                    else:
                        backward_features = None
            else:
                states = None
            
            convlstm_features = self.convlstm_block(in_features, states, pos_code)
            if self.bidirectional:
                features = [[in_feature + convlstm_feature
                             for in_feature, convlstm_feature in zip(in_features, _convlstm_features)]
                            for _convlstm_features in convlstm_features]
                outputs.extend([[self.out_block(feature) for feature in _features] for _features in features])
                features = features[-1]
            else:
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
    def __init__(self, input_size, hidden_sizes, kernel_size, num_layers, 
                 bidirectional=False, memory=True, positional_encoding=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.memory = memory
        self.positional_encoding = positional_encoding

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
            if positional_encoding:
                self.bidirectional_fuser = _BidirectionalFuser(in_channels=hidden_sizes[-1] * 2 + 1, 
                                                               out_channels=hidden_sizes[-1], 
                                                               positional_encoding=positional_encoding)
            else:
                self.bidirectional_fuser = _BidirectionalFuser(in_channels=hidden_sizes[-1] * 2, 
                                                               out_channels=hidden_sizes[-1], 
                                                               positional_encoding=positional_encoding)

    def forward(self, inputs, states=None, pos_code=None):
        if states is None:
            forward_states, backward_states = self._init_states(inputs)
        else:
            forward_states, backward_states = states

        T = inputs.size(0)
        _inputs = inputs
        for layer in range(self.num_layers):
            h_t, c_t = forward_states[layer]
            hidden_states = []
            for t in range(T):
                if not self.memory:
                    h_t, c_t = forward_states[layer]
                h_t, c_t = self.forward_cells[layer](_inputs[t], h_t, c_t)
                hidden_states.append(h_t)
            _inputs = torch.stack(hidden_states, dim=0)
        forward_hidden_states = _inputs

        if self.bidirectional:
            _inputs = inputs
            for layer in range(self.num_layers):
                h_t, c_t = backward_states[layer]
                hidden_states = []
                for t in reversed(range(T)):
                    if not self.memory:
                        h_t, c_t = backward_states[layer]
                    h_t, c_t = self.backward_cells[layer](_inputs[t], h_t, c_t)
                    hidden_states.insert(0, h_t)
                _inputs = torch.stack(hidden_states, dim=0)
            backward_hidden_states = _inputs
            fused_hidden_states = self.bidirectional_fuser(forward_hidden_states, backward_hidden_states, pos_code)
            return forward_hidden_states, backward_hidden_states, fused_hidden_states
        else:
            return forward_hidden_states

    def update_memory(self, forward_inputs, backward_inputs=None):
        forward_states, backward_states = self._init_states(forward_inputs)

        T = forward_inputs.size(0)
        for layer in range(self.num_layers):
            h_t, c_t = forward_states[layer]
            hidden_states = []
            for t in range(T):
                h_t, c_t = self.forward_cells[layer](forward_inputs[t], h_t, c_t)
                hidden_states.append(h_t)
            forward_states[layer] = (h_t, c_t)
            forward_inputs = torch.stack(hidden_states, dim=0)
        forward_hidden_states = forward_inputs
        
        if backward_inputs is not None:
            for layer in range(self.num_layers):
                h_t, c_t = backward_states[layer]
                hidden_states = []
                for t in reversed(range(T)):
                    h_t, c_t = self.backward_cells[layer](backward_inputs[t], h_t, c_t)
                    hidden_states.insert(0, h_t)
                backward_states[layer] = (h_t, c_t)
                backward_inputs = torch.stack(hidden_states, dim=0)
            backward_hidden_states = backward_inputs
            return (forward_hidden_states, backward_hidden_states), (forward_states, backward_states)
        else:
            return (forward_hidden_states, None), (forward_states, None)

    def _init_states(self, inputs):
        _, B, _, H, W = inputs.size()
        states = []
        for layer in range(self.num_layers):
            C = self.forward_cells[layer].hidden_size
            zeros = torch.zeros(B, C, H, W, dtype=inputs.dtype, device=inputs.device)
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


class _BidirectionalFuser(nn.Module):
    def __init__(self, in_channels, out_channels, positional_encoding=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.positional_encoding = positional_encoding
        if positional_encoding:
            self.fuser = nn.Sequential()
            self.fuser.add_module('conv1', nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            self.fuser.add_module('prelu1', nn.PReLU(num_parameters=1, init=0.2))
            self.fuser.add_module('conv2', nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1))
            self.fuser.add_module('prelu2', nn.PReLU(num_parameters=1, init=0.2))
        else:
            self.fuser = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1)
        
    def forward(self, forward_hidden_states, backward_hidden_states, pos_code):
        if self.positional_encoding:
            _, _, _, H, W = forward_hidden_states.size()
            pos_code = pos_code.repeat(1, H, W, 1, 1).permute(4, 3, 0, 1, 2).contiguous()
            features = torch.cat([forward_hidden_states, 
                                  backward_hidden_states,
                                  pos_code], dim=2).permute(1, 2, 0, 3, 4).contiguous()
            fused_hidden_states = self.fuser(features).permute(2, 0, 1, 3, 4).contiguous()
        else:
            features = [torch.cat([forward_hidden_state, backward_hidden_state], dim=1)
                        for forward_hidden_state, backward_hidden_state \
                        in zip(forward_hidden_states, backward_hidden_states)]
            fused_hidden_states = torch.stack([self.fuser(feature) for feature in features], dim=0)
        return fused_hidden_states
