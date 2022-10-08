import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable


# Define some constants
# KERNEL_SIZE = 3
# PADDING = KERNEL_SIZE // 2


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, kernel_size, padding):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding)

    def forward(self, input_, prev_state):

        print(f'input {input_}')
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)),
                Variable(torch.zeros(state_size))
            )
        print(f'prev_state {prev_state}')
        prev_hidden, prev_cell = prev_state

        print(f'prev_hidden {prev_hidden.shape}')
        print(f'prev_cell {prev_cell}')
        print(f'input_ {input_.shape}')


        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_.to(device), prev_hidden.to(device)), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate.to(device) * prev_cell.to(device)) + (in_gate.to(device) * cell_gate.to(device))
        hidden = out_gate.to(device) * torch.tanh(cell)

        print(f'hidden {hidden}')
        print(f'cell {cell}')

        print(f'hidden {hidden.shape}')
        print(f'cell {cell.shape}')

        return hidden, cell


