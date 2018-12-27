import numpy as np

import nnabla as nn
import nnabla.functions as F
from nnabla.parameter import get_parameter_or_create, get_parameter
from nnabla.initializer import (
    calc_uniform_lim_glorot,
    ConstantInitializer, NormalInitializer, UniformInitializer)

from .module import Module

class Affine(Module):

    def __init__(self, n_inmaps, n_outmaps, base_axis=1, w_init=None, b_init=None, 
                 fix_parameters=False, rng=None, with_bias=True):
        if not hasattr(n_outmaps, '__iter__'):
            n_outmaps = [n_outmaps]
        n_outmaps = list(n_outmaps)
        n_outmap = int(np.prod(n_outmaps))
        if w_init is None:
            w_init = UniformInitializer(
                calc_uniform_lim_glorot(n_inmaps, n_outmap), rng=rng)
        if with_bias and b_init is None:
            b_init = ConstantInitializer()
        w_shape = (n_inmaps, n_outmap)
        w = nn.Variable.from_numpy_array(w_init(w_shape)).apply(need_grad=not fix_parameters)
        b = None
        if with_bias:
            b_shape = (n_outmap, )
            b = nn.Variable.from_numpy_array(b_init(b_shape)).apply(need_grad=not fix_parameters)
        
        self.W = w
        self.b = b
        self.base_axis = base_axis
        
    def __call__(self, inp):
        return F.affine(inp, self.W, self.b, self.base_axis)

Linear = Affine
