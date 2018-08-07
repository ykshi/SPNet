import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import pdb
import math
from data import count_parameters
from torch.autograd import Variable

def clip_grad_value_(parameters, clip_value):
    r"""Clips gradient of an iterable of parameters at specified value.

    Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor]): an iterable of Tensors that will have
            gradients normalized
        clip_value (float or int): maximum allowed value of the gradients
            The gradients are clipped in the range [-clip_value, clip_value]
    """
    clip_value = float(clip_value)
    for p in filter(lambda p: p.grad is not None, parameters):
        p.grad.data.clamp_(min=-clip_value, max=clip_value)

class CNN_spnet(nn.Module):
    def __init__(self, upscale_factor):
        super(CNN_spnet, self).__init__()
        self.upscale_factor = upscale_factor
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(1, 16, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(16, 32, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(32, 128, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(128, 16, (1, 1), (1, 1), (0, 0))
        self.conv10 = nn.Conv2d(24, 2, (3, 3), (1, 1), (1, 1))
        
        self.conv9 = nn.ConvTranspose2d(16, 24, (12, 12), (upscale_factor, upscale_factor), (4, 4))        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(1. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0)
                m.bias.data.zero_()

    def forward(self, input, tag):
        self.x_bilinear = torch.nn.functional.upsample(input, scale_factor = self.upscale_factor, mode='bilinear')

        conv1 = self.relu(self.conv1(input))
        conv2 = self.relu(self.conv2(conv1))
        conv3 = self.relu(self.conv3(conv2))
        conv4 = self.relu(self.conv4(conv3))
        conv9 = self.relu(self.conv9(conv4))
        conv10 = self.conv10(conv9)
        conv10_1 = torch.cat((conv10.narrow(1,0,1), torch.sum(conv10,1,keepdim=True)),1)
        
        return conv10

class CNN(nn.Module):
    def __init__(self, upscale_factor):
        super(CNN, self).__init__()
        self.upscale_factor = upscale_factor
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.conv1 = nn.Conv2d(1, 56, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(56, 12, (1, 1), (1, 1), (0, 0))
        self.conv3 = nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1))
        self.conv6 = nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1))
        self.conv7 = nn.Conv2d(12, 56, (1, 1), (1, 1), (0, 0))
        self.conv8 = nn.ConvTranspose2d(56, 1, (11 + (upscale_factor-3), 11 + (upscale_factor-3)), (upscale_factor, upscale_factor), (4, 4))

    def forward(self, x):
        self.x_bilinear = torch.nn.functional.upsample(x, scale_factor = self.upscale_factor, mode='bilinear')

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.conv7(x)
        x = self.relu(x)
        x = self.conv8(x) + self.x_bilinear

        return x

def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def bilinear_upsample_weights(filter_size, weights):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    f_out = weights.size(0)
    f_in = weights.size(1)
    weights = np.zeros((f_out,
                        f_in,
                        filter_size,
                        filter_size), dtype=np.float32)
    
    upsample_kernel = upsample_filt(filter_size)
    
    for i in xrange(f_out):
        for j in xrange(f_in):
            weights[i, j, :, :] = upsample_kernel
    return torch.Tensor(weights)        

