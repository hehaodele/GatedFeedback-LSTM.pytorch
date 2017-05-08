"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.1.11
numpy
visdom
"""
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
from visdom import Visdom
# import matplotlib.pyplot as plt

viz = Visdom()

torch.manual_seed(1)    # reproducible

# Hyper Parameters
TIME_STEP = 5       # rnn time step
INPUT_SIZE = 1      # rnn input size
LR = 0.02           # learning rate

# show data
steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
x_np = np.sin(steps)    # float32 for converting torch FloatTensor
y_np = np.cos(steps)
# plt.plot(steps, y_np, 'r-', label='target (cos)')
# plt.plot(steps, x_np, 'b-', label='input (sin)')
# plt.legend(loc='best')
# plt.show()
viz.line(
    X = np.column_stack((steps, steps)),
    Y = np.column_stack((y_np, x_np)),
    opts = dict(legend=['target (cos)','input (sin)']),
)


from GFLSTM import GFLSTM
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = GFLSTM(
            input_size=1,
            hidden_size=32,     # rnn hidden unit
            num_layers=2,       # number of rnn layer
            batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, output_size)

        r_out, h_state = self.rnn(x, h_state)

        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()

h_state = None      # for initial hidden state
zero_h = Variable(torch.zeros(2, 1, 32))
zero_c = Variable(torch.zeros(2, 1, 32))
h_state = (zero_h, zero_c)

win = None
for step in range(60):
    start, end = step * np.pi, (step+1)*np.pi   # time steps
    # use sin predicts cos
    steps = np.linspace(start, end, 10, dtype=np.float32)
    x_np = np.sin(steps)    # float32 for converting torch FloatTensor
    y_np = np.cos(steps)

    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))    # shape (batch, time_step, input_size)
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))

    prediction, h_state = rnn(x, h_state)   # rnn output
    # !! next step is important !!
    h_state = (Variable(h_state[0].data), Variable(h_state[1].data))        # repack the hidden state, break the connection from last iteration

    loss = loss_func(prediction, y)         # cross entropy loss
    optimizer.zero_grad()                   # clear gradients for this training step
    loss.backward()                         # backpropagation, compute gradients
    optimizer.step()                        # apply gradients

    if win is None:
        win = viz.line(
            X = np.column_stack((steps, steps)),
            Y = np.column_stack((y_np.flatten(), prediction.data.numpy().flatten())),
            opts = dict(
                legend=['target (cos)','output (GFLSTM)'],
                ),
        )
    else:
        win = viz.line(
            X = np.column_stack((steps, steps)),
            Y = np.column_stack((y_np.flatten(), prediction.data.numpy().flatten())),
            win = win,
            update = 'append',
        )
