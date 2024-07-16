import torch
import torch.nn as nn
from collections import OrderedDict
from .common import *

def skip(
        num_input_channels=2, num_output_channels=1, 
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], num_channels_skip=[4, 4, 4, 4, 4], 
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True, 
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
        need1x1_up=True, dtype=torch.float32) -> nn.Sequential:
    """Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    ## print num_channels_down, num_channels_up, num_channels_skip
    #print('num_channels_down:', num_channels_down, 'num_channels_up:', num_channels_up, 'num_channels_skip:', num_channels_skip)
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down) 

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)) :
        upsample_mode   = [upsample_mode]*n_scales

    if not (isinstance(downsample_mode, list)or isinstance(downsample_mode, tuple)):
        downsample_mode   = [downsample_mode]*n_scales
    
    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)) :
        filter_size_down   = [filter_size_down]*n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)) :
        filter_size_up   = [filter_size_up]*n_scales

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(n_scales):

        deeper = nn.Sequential()
        deeper_main = nn.Sequential()

        deeper.add(conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))
        model_tmp.add(deeper)
        
        if i < n_scales-1:
            deeper.add(deeper_main)
            model_tmp.add(nn.BatchNorm2d(num_channels_up[i + 1]))
            k = num_channels_up[i + 1]
        else: # the deepest
            model_tmp.add(nn.BatchNorm2d(num_channels_down[i]))
            k = num_channels_down[i]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add(conv(k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model

def skip_replacement(
    num_input_channels=2, num_output_channels=1, 
    num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128], 
    filter_size_down=3, filter_size_up=3, 
    need_sigmoid=True, need_bias=True, 
    pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU', 
    need1x1_up=True) -> nn.Sequential:

    assert len(num_channels_down) == len(num_channels_up)
    n_scales = len(num_channels_down)

    upsample_mode = [upsample_mode] * n_scales if not isinstance(upsample_mode, (list, tuple)) else upsample_mode
    downsample_mode = [downsample_mode] * n_scales if not isinstance(downsample_mode, (list, tuple)) else downsample_mode
    filter_size_down = [filter_size_down] * n_scales if not isinstance(filter_size_down, (list, tuple)) else filter_size_down
    filter_size_up = [filter_size_up] * n_scales if not isinstance(filter_size_up, (list, tuple)) else filter_size_up

    layers = OrderedDict()
    input_depth = num_input_channels

    for i in range(n_scales):
        deeper = OrderedDict()

        deeper[f'down_conv1_{i}'] = conv(input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad, downsample_mode=downsample_mode[i])
        deeper[f'down_bn1_{i}'] = nn.BatchNorm2d(num_channels_down[i])
        deeper[f'down_act1_{i}'] = act(act_fun)

        deeper[f'down_conv2_{i}'] = conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad)
        deeper[f'down_bn2_{i}'] = nn.BatchNorm2d(num_channels_down[i])
        deeper[f'down_act2_{i}'] = act(act_fun)

        if i < n_scales - 1:
            deeper[f'upsample_{i}'] = nn.Upsample(scale_factor=2, mode=upsample_mode[i])
            layers[f'deeper_{i}'] = nn.Sequential(deeper)
            layers[f'up_bn_{i}'] = nn.BatchNorm2d(num_channels_up[i + 1])
            layers[f'concat_conv_{i}'] = conv(num_channels_up[i + 1], num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad)
        else:
            deeper[f'upsample_{i}'] = nn.Upsample(scale_factor=2, mode=upsample_mode[i])
            layers[f'deeper_{i}'] = nn.Sequential(deeper)
            layers[f'concat_conv_{i}'] = conv(num_channels_down[i], num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad)
        
        layers[f'up_bn_{i}'] = nn.BatchNorm2d(num_channels_up[i])
        layers[f'up_act_{i}'] = act(act_fun)

        if need1x1_up:
            layers[f'conv1x1_{i}'] = conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad)
            layers[f'bn1x1_{i}'] = nn.BatchNorm2d(num_channels_up[i])
            layers[f'act1x1_{i}'] = act(act_fun)

        input_depth = num_channels_down[i]

    layers['final_conv'] = conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad)
    if need_sigmoid:
        layers['sigmoid'] = nn.Sigmoid()

    return nn.Sequential(layers)