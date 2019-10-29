import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np

from src.model.nets.base_net import BaseNet


class RefineNet(BaseNet):
    """
    Args:
        in_channels (int): The input channels.
        out_channels (int): The output channels.
        num_features (list of int): The number of the internel feature maps.
        upscale_factor (int): The upscale factor (2, 3, 4 or 8).
    """
    def __init__(self, in_channels, out_channels, num_features, refine_window_size, upscale_factor, update_memory=False, memory=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.refine_window_size = refine_window_size
        self.upscale_factor = upscale_factor
        self.update_memory = update_memory
        
        if upscale_factor not in [2, 3, 4, 8]:
            raise ValueError(f'The upscale factor should be 2, 3, 4 or 8. Got {upscale_factor}.')
        
        num_feature = num_features[0]
        self.in_block = _InBlock(in_channels, num_feature) # The input block.
        self.forward_lstm_block = _ConvLSTM(input_dim=num_feature,
                                            hidden_dim=num_features,
                                            kernel_size=(3, 3),
                                            num_layers=len(num_features),
                                            bias=True,
                                            memory=memory)
        self.backward_lstm_block = _ConvLSTM(input_dim=num_feature,
                                             hidden_dim=num_features,
                                             kernel_size=(3, 3),
                                             num_layers=len(num_features),
                                             bias=True,
                                             memory=memory)
        self.refine_block = _RefineBlock(refine_window_size * num_features[-1] * 2,
                                         num_features[-1],
                                         refine_window_size)
        self.out_block = _OutBlock(num_feature, out_channels, upscale_factor)
        
    def forward(self, inputs, all_imgs, start, num_all_frames, pos_codes):
        in_features, all_in_features, refined_features, forward_features, backward_features = [], [], [], [], []
        outputs = []
        num_frames = len(inputs)
        
        #####################################################################################################
        # Get all frames feature maps of bidirectional convlstm
        #####################################################################################################
        
        with torch.no_grad():
            in_feature = self.in_block(all_imgs[0])
            self.forward_lstm_block._init_hidden(in_feature.size(0), in_feature.size(2), in_feature.size(3))
            self.backward_lstm_block._init_hidden(in_feature.size(0), in_feature.size(2), in_feature.size(3))
            if self.update_memory:
                for img in (all_imgs[-6:]):
                    in_feature = self.in_block(img)
                    forward_feature = self.forward_lstm_block(in_feature)
                for img in reversed(all_imgs[:6]):
                    in_feature = self.in_block(img)
                    backward_feature = self.backward_lstm_block(in_feature)
            for img in (all_imgs):
                in_feature = self.in_block(img)
                all_in_features.append(in_feature)
                forward_feature = self.forward_lstm_block(in_feature)
                forward_features.append(forward_feature)
            for in_feature in reversed(all_in_features):
                backward_feature = self.backward_lstm_block(in_feature)
                backward_features.insert(0, backward_feature)
        
        #######################################################################################################
        # Start of first forward
        #######################################################################################################
        
        self.forward_lstm_block._init_hidden(in_feature.size(0), in_feature.size(2), in_feature.size(3))
        self.backward_lstm_block._init_hidden(in_feature.size(0), in_feature.size(2), in_feature.size(3))
        
        # Forward
        _outputs = []
        if self.update_memory:
            with torch.no_grad():
                for input in inputs[:6]:
                    in_feature = self.in_block(input)
                    in_features.append(in_feature)
                    feature = self.forward_lstm_block(in_feature)
        for input in inputs[6:-6]:
            in_feature = self.in_block(input)
            in_features.append(in_feature)
            feature = self.forward_lstm_block(in_feature)
            output = self.out_block(feature+in_feature)
            _outputs.append(output)
        outputs.append(_outputs)
        if self.update_memory:
            with torch.no_grad():
                for input in inputs[-6:]:
                    in_feature = self.in_block(input)
                    in_features.append(in_feature)
        
        # Backward
        _outputs = []
        if self.update_memory:
            with torch.no_grad():
                for in_feature in reversed(in_features[-6:]):
                    feature = self.backward_lstm_block(in_feature)
        for in_feature in reversed(in_features[6:-6]):
            feature = self.backward_lstm_block(in_feature)
            output = self.out_block(feature+in_feature)
            _outputs.insert(0, output)
        outputs.append(_outputs)
        
        #  Fused
        _outputs = []
        for i in range(num_frames):
            in_feature = in_features[i]
            refine_map = self.refine_block(forward_features, backward_features, pos_codes, \
                                           (start+i) % num_all_frames, num_all_frames)
            output = self.out_block(refine_map+in_feature)
            refined_features.append(refine_map+in_feature)
            _outputs.append(output)
        outputs.append(_outputs[6:-6])
        
        ########################################################################################################
        # End of first forward
        ########################################################################################################
        
        return tuple(outputs)
        
        ########################################################################################################
        # Get all frames "refined" feature maps of bidirectional convlstm
        ########################################################################################################
        
        refined_forward_features, refined_backward_features, refine_maps = [], [], []
        with torch.no_grad():
            in_feature = self.in_block(all_imgs[0])
            self.forward_lstm_block._init_hidden(in_feature.size(0), in_feature.size(2), in_feature.size(3))
            self.backward_lstm_block._init_hidden(in_feature.size(0), in_feature.size(2), in_feature.size(3))
            for i in range(len(all_in_features)):
                refine_maps.append(self.refine_block(forward_features, backward_features, pos_codes, 
                                                     (torch.zeros_like(num_all_frames)+i) % num_all_frames, num_all_frames))
            for i, in_feature in enumerate(all_in_features):
                forward_feature = self.forward_lstm_block(in_feature+refine_maps[i])
                refined_forward_features.append(forward_feature)
            for i, in_feature in enumerate(reversed(all_in_features)):
                backward_feature = self.backward_lstm_block(in_feature+refine_maps[len(all_in_features)-i-1])
                refined_backward_features.insert(0, backward_feature)
            
        ########################################################################################################
        # Start of second forward (refinement)
        ########################################################################################################
        
        self.forward_lstm_block._init_hidden(in_feature.size(0), in_feature.size(2), in_feature.size(3))
        self.backward_lstm_block._init_hidden(in_feature.size(0), in_feature.size(2), in_feature.size(3))
        
        # Forward
        _outputs = []
        if self.update_memory:
            with torch.no_grad():
                for in_feature in refined_features[:6]:
                    feature = self.forward_lstm_block(in_feature)
        for in_feature in refined_features[6:-6]:
            feature = self.forward_lstm_block(in_feature)
            output = self.out_block(feature+in_feature)
            _outputs.append(output)
        outputs.append(_outputs)
        
        # Backward
        _outputs = []
        if self.update_memory:
            with torch.no_grad():
                for in_feature in reversed(refined_features[-6:]):
                    feature = self.backward_lstm_block(in_feature)
        for in_feature in reversed(refined_features[6:-6]):
            feature = self.backward_lstm_block(in_feature)
            output = self.out_block(feature+in_feature)
            _outputs.insert(0, output)
        outputs.append(_outputs)
        
        # with fused
        _outputs = []
        for i in range(num_frames):
            in_feature = refined_features[i]
            refine_map = self.refine_block(refined_forward_features, refined_backward_features, pos_codes, 
                                           (start+i) % num_all_frames, num_all_frames)
            output = self.out_block(refine_map+in_feature)
            _outputs.append(output)
        outputs.append(_outputs[6:-6])
        
        ########################################################################################################
        # End of second forward (refinement)
        ########################################################################################################
        
        return tuple(outputs)
    
    
class _RefineBlock(nn.Module):
    def __init__(self, in_channels, num_feature, num_frames):
        super().__init__()
        self.in_channels = in_channels
        self.num_feature = num_feature
        self.num_frames = num_frames
        
        self.body = nn.Sequential()
        self.body.add_module('conv1', nn.Conv2d(in_channels, num_feature, kernel_size=3, padding=1))
        self.body.add_module('prelu1', nn.ReLU(True))
        self.body.add_module('conv2', nn.Conv2d(num_feature, num_feature, kernel_size=3, padding=1))
        self.body.add_module('prelu2', nn.ReLU(True))
        
        
    def forward(self, forward_features, backward_features, pos_codes, t, num_all_frames):
        """
        Args:
            forward_features (list of FloatTensor): the forward hidden features of all frames
            backward_features (list of FloatTensor): the backward hidden features of all frames
            t (int): current frame index
            num_all_frames (int): total number of frames
        """
        n, c, h, w = forward_features[0].shape
        start, end = t - self.num_frames // 2, t + self.num_frames // 2 + 1
        inputs = torch.zeros((n, self.in_channels, h, w), device=forward_features[0].device)

        for b in range(n):
            if end[b] > num_all_frames[b]:
                end[b] %= num_all_frames[b]
                forward_feature = forward_features[start[b]:num_all_frames[b]] + forward_features[:end[b]]
                backward_feature = backward_features[start[b]:num_all_frames[b]] + backward_features[:end[b]]
                pos_code = torch.cat((pos_codes[b, start[b]:num_all_frames[b]], pos_codes[b, :end[b]]), dim=0)
            elif start[b] < 0:
                forward_feature = forward_features[start[b]:] + forward_features[:end[b]]
                backward_feature = backward_features[start[b]:] + backward_features[:end[b]]
                pos_code = torch.cat((pos_codes[b, start[b]:], pos_codes[b, :end[b]]), dim=0)
            else:
                forward_feature = forward_features[start[b]:end[b]]
                backward_feature = backward_features[start[b]:end[b]]
                pos_code = pos_codes[b, start[b]:end[b]]
            
            pos_code = torch.repeat_interleave(pos_code, (h*w), dim=1).view(self.num_frames, 1, h, w)
            forward_feature = torch.stack([feature[b] for feature in forward_feature], dim=0) + pos_code
            backward_feature = torch.stack([feature[b] for feature in backward_feature], dim=0) + pos_code
            inputs[b] = torch.cat((forward_feature, backward_feature), dim=1).view(-1, h, w).contiguous()
        
        return self.body(inputs)
        

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