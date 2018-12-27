import numpy as np

import nnabla as nn
import nnabla.functions as F
from nnabla.parameter import get_parameter_or_create, get_parameter
from nnabla.initializer import (
    calc_uniform_lim_glorot,
    ConstantInitializer, NormalInitializer, UniformInitializer)

from .module import Module

class Deconvolution(Module):
    
    def __init__(self, inmaps, outmaps, kernel,
                pad=None, stride=None, dilation=None, group=1,
                w_init=None, b_init=None,
                base_axis=1, fix_parameters=False, rng=None, with_bias=True):
        if w_init is None:
            w_init = UniformInitializer(
                calc_uniform_lim_glorot(inmaps, outmaps, tuple(kernel)), rng=rng)
        if with_bias and b_init is None:
            b_init = ConstantInitializer()
        w_shape = (outmaps, inmaps // group) + tuple(kernel)        
        w = nn.Variable.from_numpy_array(w_init(w_shape)).apply(need_grad=not fix_parameters)
        b = None
        if with_bias:
            b_shape = (outmaps, )
            b = nn.Variable.from_numpy_array(b_init(b_shape)).apply(need_grad=not fix_parameters)

        self.W = w
        self.b = b
        self.base_axis = base_axis
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.group = group

    def __call__(self, inp):
        return F.deconvolution(inp, self.W, self.b, self.base_axis, 
                             self.pad, self.stride, self.dilation, self.group)

Deconv1d = Deconvolution
Deconv2d = Deconvolution
Deconv3d = Deconvolution
DeconvNd = Deconvolution
