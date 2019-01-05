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


    def save_parameters(self, path, grad_only=False):
        params = self.get_parameters(grad_only=grad_only)
        nn.save_parameters(path, params)


    def load_parameters(self, path):
        nn.load_parameters(path)
        for v in self.get_modules():
            if not isinstance(v, tuple):
                continue
            prefix, module = v
            for k, v in module.__dict__.items():
                if not isinstance(v, nn.Variable):
                    continue
                pname = k
                name = "{}/{}".format(prefix, pname)
                # Substitute
                param0 = v
                param1 = nn.parameter.pop_parameter(name)
                if param0 is None:
                    raise ValueError("Model does not have {} parameter.".format(name))
                param0.d = param1.d.copy()
                nn.logger.info("`{}` loaded.)".format(name))
                
                
