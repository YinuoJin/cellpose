import torch
import torch.nn as nn

"""
References: 
Code modified with thanks from:
https://github.com/iliasprc/ConvGRU-ConvLSTM-PyTorch
https://github.com/ndrplz/ConvLSTM_pytorch
https://github.com/shreyaspadhy/UNet-Zoo/blob/master/CLSTM.py
https://gist.github.com/halochou/acbd669af86ecb8f988325084ba7a749
"""


class ConvGRUCell(nn.Module):
    """
    Basic CGRU cell.
    """

    def __init__(self, in_channels, hidden_channels, kernel_size, bias):

        super(ConvGRUCell, self).__init__()

        self.input_dim = in_channels
        self.hidden_dim = hidden_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.bias = bias
        self.update_gate = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, out_channels=self.hidden_dim,
                                     kernel_size=self.kernel_size, padding=self.padding,
                                     bias=self.bias)
        self.reset_gate = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, out_channels=self.hidden_dim,
                                    kernel_size=self.kernel_size, padding=self.padding,
                                    bias=self.bias)
        self.out_gate = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, out_channels=self.hidden_dim,
                                  kernel_size=self.kernel_size, padding=self.padding,
                                  bias=self.bias)

    def forward(self, input_tensor, cur_state):

        if not input_tensor.is_cuda:
            input_tensor = input_tensor.cuda()
        if not cur_state.is_cuda:
            cur_state = cur_state.cuda()

        h_cur = cur_state

        # data size is [batch, channel, height, width]
        x_in = torch.cat([input_tensor, h_cur], dim=1)
        update = torch.sigmoid(self.update_gate(x_in))
        reset = torch.sigmoid(self.reset_gate(x_in))
        x_out = torch.tanh(self.out_gate(torch.cat([input_tensor, h_cur * reset], dim=1)))
        h_new = h_cur * (1 - update) + x_out * update

        return h_new

    def init_hidden(self, b, h, w):
        return torch.zeros(b, self.hidden_dim, h, w).cuda()

class ConvGRU(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvGRU, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_channels = self._extend_for_multilayer(hidden_channels, num_layers)
        if not len(kernel_size) == len(hidden_channels) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = in_channels
        self.hidden_dim = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvGRUCell(in_channels=cur_input_dim,
                                         hidden_channels=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_tensor=None):
        """
        Parameters
        ----------
        input_tensor: torch.Tensor
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)

        hidden_tensor: torch.Tensor
            4-D Tensor of shape (b, c, h, w) representing init. hidden state

            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # TODO:
        #  (1). Initialize h0 as either first / last layer in minipatch
        #  (2). Implement fully stateful ConvGRU
        if hidden_tensor is not None:
            hidden_state = self._init_hidden_stateful(hidden_tensor)
        else:
            b, _, _, h, w = input_tensor.shape
            hidden_state = self._init_hidden(b, h, w)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                              cur_state=h)
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(h)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden_stateful(self, h):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(h)
        return init_states

    def _init_hidden(self, b, h, w):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(b, h, w))
        return init_states

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


class ConvBGRU(nn.Module):
    # Constructor
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 kernel_size,
                 num_layers,
                 bias=True,
                 batch_first=False,
                 combine_option='concat',
                 ):

        super(ConvBGRU, self).__init__()

        assert combine_option == 'concat' or combine_option == 'sum' or combine_option == 'avg', \
            "Invalid combination option for fwd & bwd passes, options: (1). concat, (2). sum, (3). avg"
        self.option = combine_option

        h_channels = hidden_channels

        self.forward_net = ConvGRU(in_channels, h_channels, kernel_size,
                                   num_layers, batch_first=batch_first, bias=bias)
        self.reverse_net = ConvGRU(in_channels, h_channels, kernel_size,
                                   num_layers, batch_first=batch_first, bias=bias)

        self.z0_trans = nn.Conv2d(in_channels, h_channels, 1)
        self.zt_trans = nn.Conv2d(in_channels, h_channels, 1)

    def forward(self, xforward, xreverse):
        """
        xforward, xreverse = B T C H W tensors.
        """
        # Pass the first z-slice input as init. hidden state
        z0, zt = self.z0_trans(xforward[:, 0, ...]), self.zt_trans(xforward[:, -1, ...])
        y_out_fwd, _ = self.forward_net(xforward, z0)
        y_out_rev, _ = self.reverse_net(xreverse, zt)

        y_out_fwd = y_out_fwd[-1]  # outputs of last CGRU layer = B, T, C, H, W
        y_out_rev = y_out_rev[-1]  # outputs of last CGRU layer = B, T, C, H, W

        reversed_idx = list(reversed(range(y_out_rev.shape[1])))
        y_out_rev = y_out_rev[:, reversed_idx, ...]  # reverse temporal outputs.

        if self.option == 'concat':
            yout = torch.cat((y_out_fwd, y_out_rev), dim=2)
        elif self.option == 'sum':
            yout = (y_out_fwd + y_out_rev)
        else:  # avg.
            yout = (y_out_fwd + y_out_rev) / 2

        return yout
