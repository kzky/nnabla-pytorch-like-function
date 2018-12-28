import numpy as np

import nnabla as nn
import nnabla.functions as F
from nnabla.parameter import get_parameter_or_create, get_parameter
from nnabla.initializer import (
    calc_uniform_lim_glorot,
    ConstantInitializer, NormalInitializer, UniformInitializer)

from collections import OrderedDict

class Module(object):
    
    def __init__(self):
        pass

    def get_parameters(self, grad_only=True):
        params = OrderedDict()

        for v in self.get_modules():
            if not isinstance(v, tuple):
                continue
            prefix, module = v
            for k, v in module.__dict__.items():
                if not isinstance(v, nn.Variable):
                    continue
                pname = k
                name = "{}/{}".format(prefix, pname)
                if grad_only and v.need_grad == False:
                    continue
                params[name] = v
        return params
    

    def get_modules(self, memo=None, prefix=""):
        if memo is None:
            memo = set()
            
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for k, v in self.__dict__.items():
                if not isinstance(v, Module):
                    continue
                name, module = k, v
                submodule_prefix = "{}/{}".format(prefix, name) if prefix != "" else name
                for m in module.get_modules(memo, submodule_prefix):
                    yield m


    def save_parameters(self, grad_only=False):
        raise NotImplementedError("save_parameters not implemented yet.")
