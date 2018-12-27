import numpy as np

import nnabla as nn
import nnabla.functions as F
from nnabla.parameter import get_parameter_or_create, get_parameter
from nnabla.initializer import (
    calc_uniform_lim_glorot,
    ConstantInitializer, NormalInitializer, UniformInitializer)

from .module import Module

class Embed(Module):
    
    def __init__(self, n_inputs, n_features, w_init=None, fix_parameters=False):
        if w_init is None:
            w_init = UniformInitializer((-np.sqrt(3.), np.sqrt(3)))
        w_shape = (n_input, n_features)
        w = nn.Variables.from_numpy_array(w_init()).apply(need_grad=not fix_parameters)
        self.W = w

    def __call__(self, inp):
        return F.embed(inp, self.W)
