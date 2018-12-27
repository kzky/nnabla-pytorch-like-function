import numpy as np

import nnabla as nn
import nnabla.functions as F
from nnabla.parameter import get_parameter_or_create, get_parameter
from nnabla.initializer import (
    calc_uniform_lim_glorot,
    ConstantInitializer, NormalInitializer, UniformInitializer)

from .module import Module

class BatchNormalization(Module):
    
    def __init__(self, n_features, n_dims, axes=[1], decay_rate=0.9, eps=1e-5,
                 batch_stat=True, output_stat=False, fix_parameters=False,
                 param_init=None):
        assert len(axes) == 1
        shape_stat = [1 for _ in range(n_dims)]
        shape_stat[axes[0]] = n_features
    
        if param_init is None:
            param_init = {}
        beta_init = param_init.get('beta', ConstantInitializer(0))
        gamma_init = param_init.get('gamma', ConstantInitializer(1))
        mean_init = param_init.get('mean', ConstantInitializer(0))
        var_init = param_init.get('var', ConstantInitializer(1))
        
        beta = nn.Variable.from_numpy_array(beta_init(shape_stat)).apply(need_grad=not fix_parameters)
        gamma = nn.Variable.from_numpy_array(gamma_init(shape_stat)).apply(need_grad=not fix_parameters)
        mean = nn.Variable.from_numpy_array(mean_init(shape_stat))
        var = nn.Variable.from_numpy_array(var_init(shape_stat))

        self.beta = beta
        self.gamma = gamma
        self.mean = mean
        self.var = var
        self.axes = axes
        self.decay_rate = decay_rate
        self.eps = eps
        self.output_stat = output_stat

    def __call__(self, inp, test=False):
        return F.batch_normalization(inp, self.beta, self.gamma, self.mean, self.var, self.axes,
                                     self.decay_rate, self.eps, not test, self.output_stat)

class BatchNorm1d(BatchNormalization):
    
    def __init__(self, n_features, axes=[1], decay_rate=0.9, eps=1e-5,
                 batch_stat=True, output_stat=False, fix_parameters=False,
                 param_init=None):
        super(BatchNorm1d, self).__init__(n_features, 3, axes=axes, decay_rate=decay_rate, 
                                          eps=1e-5, batch_stat=batch_stat, output_stat=output_stat, 
                                          fix_parameters=fix_parameters,
                                          param_init=param_init)

class BatchNorm2d(BatchNormalization):
    
    def __init__(self, n_features, axes=[1], decay_rate=0.9, eps=1e-5,
                 batch_stat=True, output_stat=False, fix_parameters=False,
                 param_init=None):
        super(BatchNorm2d, self).__init__(n_features, 4, axes=axes, decay_rate=decay_rate, 
                                          eps=1e-5, batch_stat=batch_stat, output_stat=output_stat, 
                                          fix_parameters=fix_parameters,
                                          param_init=param_init)

class BatchNorm3d(BatchNormalization):
    
    def __init__(self, n_features, axes=[1], decay_rate=0.9, eps=1e-5,
                 batch_stat=True, output_stat=False, fix_parameters=False,
                 param_init=None):
        super(BatchNorm3d, self).__init__(n_features, 5, axes=axes, decay_rate=decay_rate, 
                                          eps=1e-5, batch_stat=batch_stat, output_stat=output_stat, 
                                          fix_parameters=fix_parameters,
                                          param_init=param_init)
