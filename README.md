# Gated Feedback RNNs in Pytorch
This is my ongoing PyTorch implementation for "Gated Feedback Recurrent Neural Networks" [[Paper]](https://arxiv.org/abs/1502.02367v4)

## Prerequisites
- Linux or OSX.
- Python 2 or Python 3.
- CPU or NVIDIA GPU + CUDA CuDNN.aa

## Usage
This ```GFLSTM.py``` could be easily used as RNN modules in ```torch.nn``` package.
Following is an simple example
```python
    import torch
    import torch.nn as nn
    from torch.autograd import Variable
    from GFLSTM import GFLSTM
    import numpy as np
  
    input_size = 20
    output_size = 100
    hidden_size = 100
    num_layers = 3
    num_seq = 5
    nbatch = 2
    x = Variable(torch.randn(num_seq, nbatch, input_size))
    xh = Variable(torch.zeros(num_layers, nbatch, hidden_size))
    xc = Variable(torch.zeros(num_layers, nbatch, hidden_size))
    
    gflstm = GFLSTM(input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first = False,)
    print np.sum(map(lambda x: x.numel(), gflstm.parameters()))
    
    lstm = nn.LSTM(input_size=input_size,
                   hidden_size=hidden_size,
                   num_layers=num_layers,
                   batch_first = False,)
    print np.sum(map(lambda x: x.numel(), lstm.parameters()))
```
The output will be 
```bash
608800
210400
```
