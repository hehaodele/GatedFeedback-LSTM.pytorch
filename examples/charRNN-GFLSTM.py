import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


with open('anna.txt', 'r') as f:
    text = f.read()

chars = set(text)
int2char = dict(enumerate(chars))
char2int = {ch: ii for ii, ch in int2char.items()}
encoded = np.array([char2int[ch] for ch in text])

def get_batches(arr, n_seqs, n_steps):
    '''Create a generator that returns batches of size
       n_seqs x n_steps from arr.
    '''

    batch_size = n_seqs * n_steps
    n_batches = len(arr)//batch_size

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size]
    # Reshape into n_seqs rows
    arr = arr.reshape((n_seqs, -1))

    for n in range(0, arr.shape[1], n_steps):
        # The features
        x = arr[:, n:n+n_steps]
        # The targets, shifted by one
        y = np.zeros_like(x)
        y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]
        yield torch.from_numpy(x), torch.from_numpy(y)

from GFLSTM import GFLSTM
class CharRNN(nn.Module):
    def __init__(self, tokens, embed_dim=50, n_steps=100,
                               n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super(CharRNN,self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        self.embed = nn.Embedding(len(self.chars), embed_dim)
        self.dropout = nn.Dropout(drop_prob)
        self.lstm = GFLSTM(embed_dim, n_hidden, n_layers,
                            #dropout=drop_prob,
                            batch_first=True)
        self.fc = nn.Linear(n_hidden, len(self.chars))

        self.init_weights()

        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, hc):
        x = self.embed(x)
        x = self.dropout(x)
        x, (h, c) = self.lstm(x, hc)
        x = self.dropout(x)

        # Stack up LSTM outputs
        x = x.view(x.size()[0]*x.size()[1], self.n_hidden)

        x = self.fc(x)

        return x, (h, c)

    def predict(self, char, h=None, cuda=False, top_k=None):
        ''' Given a character, predict the next character.

            Returns the predicted character and the hidden state.
        '''
        if cuda:
            self.cuda()
        else:
            self.cpu()

        if h is None:
            h = self.init_hidden(1)

        x = np.array([[self.char2int[char]]])
        inputs = Variable(torch.from_numpy(x), volatile=True)
        if cuda:
            inputs = inputs.cuda()

        h = tuple([Variable(each.data, volatile=True) for each in h])
        out, h = self.forward(inputs, h)

        p = F.softmax(out).data
        if cuda:
            p = p.cpu()
        p = p.numpy().squeeze()

        if top_k is not None:
            p[np.argsort(p)[:-top_k]] = 0
            p = p/p.sum()

        char = np.random.choice(np.arange(len(self.chars)), p=p)

        return self.int2char[char], h

    def init_weights(self):

        initrange = 0.1
        # Embedding weights as random uniform
        self.embed.weight.data.uniform_(-initrange, initrange)
        # Set bias tensor to all zeros
        self.fc.bias.data.fill_(0)
        # FC weights as random uniform
        self.fc.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, n_seqs):
        # Create two new tensors with sizes n_layers x n_seqs x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.n_layers, n_seqs, self.n_hidden).zero_()),
                Variable(weight.new(self.n_layers, n_seqs, self.n_hidden).zero_()))

def train(net, data, epochs=10, n_seqs=10, n_steps=50, clip=5, cuda=False, print_every=10):
    torch.cuda.set_device(1)
    net.train()
    if cuda:
        net.cuda()
    counter = 0
    for e in range(epochs):
        h = net.init_hidden(n_seqs)
        for x, y in get_batches(data, n_seqs, n_steps):
            counter += 1

            inputs, targets = Variable(x), Variable(y)
            if cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([Variable(each.data) for each in h])

            net.zero_grad()

            output, h = net.forward(inputs, h)
            loss = net.criterion(output, targets.view(n_seqs*n_steps))

            loss.backward()

            #`clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
#             nn.utils.clip_grad_norm(net.parameters(), clip)
#             for p in net.parameters():
#                 p.data.add_(-net.lr, p.grad.data)

            net.opt.step()

            if counter % print_every == 0:
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}".format(loss.data[0]))

net = CharRNN(chars, embed_dim=len(chars), n_hidden=512, lr=0.001, n_layers=2)

n_seqs, n_steps = 128, 100
train(net, encoded, 20, n_seqs=n_seqs, n_steps=n_steps, cuda=True, print_every=10)