import torch
from torch import nn
import torch.nn.functional as f
from torch.autograd import Variable


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, kernel_size, padding):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(
            in_channels=input_size + hidden_size, out_channels=4 * hidden_size,
            kernel_size=kernel_size, padding=padding)

    def forward(self, input_, prev_state):

#         print(f'input {input_}')
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # print(input_.data.size())

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                Variable(torch.zeros(state_size)),
                Variable(torch.zeros(state_size))
            )

        prev_hidden, prev_cell = prev_state

        if prev_state[0].size()[0] > batch_size:
            prev_hidden = prev_state[0][:batch_size]
            prev_cell = prev_state[1][:batch_size]
#         print(f'prev_state {prev_state}')

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_.to(device), prev_hidden.to(device)), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate.to(device) * prev_cell.to(device)) + (in_gate.to(device) * cell_gate.to(device))
        hidden = out_gate.to(device) * torch.tanh(cell)

        return hidden, cell


