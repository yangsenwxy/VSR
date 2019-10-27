import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import copy

from src.model.nets.base_net import BaseNet


class RefineNet(BaseNet):
    """
    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
        num_features (list of int): The number of the internel feature maps.
        bidirectional (bool):
        upscale_factor (int): The upscale factor (2, 3, 4 or 8).
    """
    def __init__(self, in_channels, out_channels, num_frames, num_features, num_stages, upscale_factor, bidirectional=False, 
                 memory=True, updated_memory=False, updated_memory_frame_number=0, positional_encoding=False):
        super().__init__()
        assert num_features[0] == num_features[-1]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_frames = num_frames
        self.num_features = num_features
        self.num_stages = num_stages
        self.bidirectional = bidirectional
        self.upscale_factor = upscale_factor
        self.memory = memory
        self.updated_memory = updated_memory
        self.updated_memory_frame_number = updated_memory_frame_number
        self.positional_encoding = positional_encoding

        if upscale_factor not in [2, 3, 4, 8]:
            raise ValueError(f'The upscale factor should be 2, 3, 4 or 8. Got {upscale_factor}.')

        num_fuse_features = num_features[-1]
        if bidirectional:
            num_fuse_features *= 2
        if positional_encoding:
            num_fuse_features += 1
            
        self.in_block = _InBlock(in_channels, num_features[0])
        self.forward_lstm_block = _ConvLSTM(input_dim=num_features[0],
                                            hidden_dim=num_features,
                                            kernel_size=(3, 3),
                                            num_layers=len(num_features),
                                            bias=True,
                                            memory=memory)
        self.backward_lstm_block = _ConvLSTM(input_dim=num_features[0],
                                             hidden_dim=num_features,
                                             kernel_size=(3, 3),
                                             num_layers=len(num_features),
                                             bias=True,
                                             memory=memory)
        self.fuse_block = _FuseBlock(num_fuse_features, num_features[-1], positional_encoding)
        self.out_block = _OutBlock(num_features[-1], out_channels, upscale_factor) # The output block.

    def forward(self, inputs, pos_codes, start_frame, num_all_frames):
        inputs = torch.stack(inputs, dim=0) # T, N, C, H, W
        pos_codes = pos_codes.permute(1, 0, 2).contiguous()
        inputs, pos_codes, forward_frames, backward_frames = self._get_inputs(inputs, pos_codes, start_frame, num_all_frames)
        
        in_features = torch.stack([self.in_block(input) for input in inputs], dim=0)
        if self.updated_memory:
            with torch.no_grad():
                forward_features = torch.stack([self.in_block(input) for input in forward_frames], dim=0)
                if self.bidirectional:
                    backward_features = torch.stack([self.in_block(input) for input in backward_frames], dim=0)
                else:
                    backward_features = None
        
        outputs = []
        for i in range(self.num_stages):
            # Initialize the convlstm
            self.forward_lstm_block._init_hidden(in_features[0].size(0), in_features[0].size(2), in_features[0].size(3))
            self.backward_lstm_block._init_hidden(in_features[0].size(0), in_features[0].size(2), in_features[0].size(3))
            
            # Update the memory of the convlstm if necessary
            if self.updated_memory:
                with torch.no_grad():
                    for j, feature in enumerate(forward_features):
                        forward_features[j] = self.forward_lstm_block(feature) + feature
                    if self.bidirectional:
                        for j in reversed(range(len(backward_features))):
                            backward_features[j] = self.backward_lstm_block(backward_features[j]) + feature
            
            # Main part
            forward_h_ts, backward_h_ts = [], []
            for feature in in_features:
                forward_h_t = self.forward_lstm_block(feature)
                forward_h_ts.append(forward_h_t)
            fuse_inputs = torch.stack(forward_h_ts, dim=2) # N, C, T, H, W
            
            if self.bidirectional:
                for feature in reversed(in_features):
                    backward_h_t = self.backward_lstm_block(feature)
                    backward_h_ts.insert(0, backward_h_t)
                fuse_inputs = torch.cat((fuse_inputs, torch.stack(backward_h_ts, dim=2)), dim=1)
                
            if self.positional_encoding:
                _pos_codes = pos_codes.permute(1, 2, 0, 3, 4).contiguous()
                fuse_inputs = torch.cat((fuse_inputs, _pos_codes), dim=1)

            print(fuse_inputs.shape)
            outputs.append([self.out_block(forward_h_ts[n] + in_features[n]) for n in range(self.num_frames)])
            if self.bidirectional:
                outputs.append([self.out_block(backward_h_ts[n] + in_features[n]) for n in range(self.num_frames)])
            fuse_features = self.fuse_block(fuse_inputs).permute(2, 0, 1, 3, 4).contiguous()
            outputs.append([self.out_block(fuse_features[n] + in_features[n]) for n in range(self.num_frames)])
            
            # Update the input features
            in_features += fuse_features

        return outputs
        
    def _get_inputs(self, inputs, pos_codes, start_frame, num_all_frames):
        _, N, C, H, W = inputs.shape
        start, end = start_frame, start_frame + self.num_frames
        
        # Declaration of the inputs
        forward_frames = torch.zeros((self.updated_memory_frame_number, N, C, H, W), device=inputs[0].device)
        backward_frames = torch.zeros((self.updated_memory_frame_number, N, C, H, W), device=inputs[0].device)
        input_frames = torch.zeros((self.num_frames, N, C, H, W), device=inputs[0].device)
        input_pos_codes = torch.zeros((self.num_frames, N, 1), device=inputs[0].device)

        for n in range(N):
            # The main inputs of the network
            if end[n] > num_all_frames[n]:
                end[n] %= num_all_frames[n]
                input_frames[:, n] = torch.cat((inputs[start[n]:num_all_frames[n], n], inputs[:end[n], n]), dim=0)
                input_pos_codes[:, n] = torch.cat((pos_codes[start[n]:num_all_frames[n], n], pos_codes[:end[n], n]), dim=0)
            elif start[n] < 0:
                start[n] %= num_all_frames[n]
                input_frames[:, n] = torch.cat((inputs[start[n]:num_all_frames[n], n], inputs[:end[n], n]), dim=0)
                input_pos_codes[:, n] = torch.cat((pos_codes[start[n]:num_all_frames[n], n], pos_codes[:end[n], n]), dim=0)
            else:
                input_frames[:, n] = inputs[start[n]:end[n], n]
                input_pos_codes[:, n] = pos_codes[start[n]:end[n], n]

            # Used for updating memory of the convlstm blocks
            # - Forward convlstm
            _start, _end = start[n] - self.updated_memory_frame_number, start[n]
            if _start < 0:
                _start %= num_all_frames[n]
                forward_frames[:, n] = torch.cat((inputs[_start:num_all_frames[n], n], inputs[:_end, n]), dim=0)
            else:
                forward_frames[:, n] = inputs[_start:_end, n]
            # - Backward convlstm
            _start, _end = end[n], end[n] + self.updated_memory_frame_number
            if _end > num_all_frames[n]:
                _end %= num_all_frames[n]
                backward_frames[:, n] = torch.cat((inputs[_start:num_all_frames[n], n], inputs[:_end, n]), dim=0)
            else:
                backward_frames[:, n] = inputs[_start:_end, n]
        input_pos_codes = input_pos_codes.repeat(H, W, 1, 1, 1).permute(2, 3, 4, 0, 1).contiguous()
        return input_frames, input_pos_codes, forward_frames, backward_frames


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


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, memory):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.memory = memory

        if memory:
            self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)
        else:
            self.conv = nn.Conv2d(in_channels=self.input_dim * 2,
                                  out_channels=4 * self.hidden_dim,
                                  kernel_size=self.kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)

    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state

        if self.memory:
            # concatenate along channel axis
            combined = torch.cat([input_tensor, h_cur], dim=1)
        else:
            combined = torch.cat([input_tensor, input_tensor], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, height, width):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, height, width)).cuda(),
                Variable(torch.zeros(batch_size, self.hidden_dim, height, width)).cuda())


class _ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, bias=True, memory=True):
        super(_ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.hidden_state = None

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                          memory=memory))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor,):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """

        cur_layer_input = input_tensor
        for layer_idx in range(self.num_layers):
            h, c = self.hidden_state[layer_idx]
            h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input, cur_state=[h, c])
            self.hidden_state[layer_idx] = (h, c)
            cur_layer_input = h

        return h

    def _init_hidden(self, batch_size, height, width):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, height, width))
        self.hidden_state = init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class _FuseBlock(nn.Module):
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
            self.fuser = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, inputs):
        return self.fuser(inputs)
