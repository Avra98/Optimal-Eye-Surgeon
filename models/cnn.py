import torch
import torch.nn as nn
from .common import *
import itertools

def cnn(
        num_input_channels=2, num_output_channels=1,
        num_layers=4,
        need_bias=True, pad='zero',
        act_fun='LeakyReLU'):
    """Assembles a user-specified N-layer CNN.
    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
    """

    num_channels_base = [16, 32, 64, 128]  # This will repeat every 4 layers
    filter_sizes_base = [3,3,3,3]  # This will repeat every 4 layers

    num_channels = list(itertools.islice(itertools.cycle(num_channels_base), num_layers))
    filter_sizes = list(itertools.islice(itertools.cycle(filter_sizes_base), num_layers))

    model = nn.Sequential()

    input_depth = num_input_channels
    for i in range(num_layers):
        model.add_module(f"conv_{i+1}", conv(input_depth, num_channels[i], filter_sizes[i], bias=need_bias, pad=pad))
        model.add_module(f"bn_{i+1}", bn(num_channels[i]))
        model.add_module(f"act_{i+1}", act(act_fun))

        input_depth = num_channels[i]

    model.add_module("final_conv", conv(num_channels[-1], num_output_channels, 1, bias=need_bias, pad=pad))

    return model
