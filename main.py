import numpy as np

import nnabla as nn
import nnabla.functions as F
import nnabla.solvers as S
from nnabla.ext_utils import get_extension_context

import parametric_function_classes as PFC
from mnist_data import data_iterator_mnist
from nnabla.monitor import Monitor, MonitorSeries

class Model0(PFC.Module):
    def __init__(self):
        self.conv0 = PFC.Conv2d(1, 16, (3, 3), pad=(1, 1), with_bias=False)
        self.conv1 = PFC.Conv2d(16, 32, (3, 3), pad=(1, 1), with_bias=False)
        self.bn0 = PFC.BatchNorm2d(16)
        self.bn1 = PFC.BatchNorm2d(32)
        self.act = F.relu
        self.pool = F.max_pooling
    def __call__(self, x, test=False):
        h = x
        h = self.conv0(h)
        h = self.bn0(h, test)
        h = self.act(h)
        h = self.pool(h, (2, 2))
        h = self.conv1(h)
        h = self.bn1(h, test)
        h = self.act(h)
        h = self.pool(h, (2, 2))
        return h

class Model1(PFC.Module):
    def __init__(self):
        self.conv0 = PFC.Conv2d(32, 64, (3, 3), pad=(1, 1), with_bias=False)
        self.bn0 = PFC.BatchNorm2d(64)
        self.affine = PFC.Affine(64, 10)
        self.act = F.relu
        self.pool = F.average_pooling
    def __call__(self, x, test=False):
        h = x
        h = self.conv0(h)
        h = self.bn0(h, test)
        h = self.act(h)
        h = self.pool(h, h.shape[2:])
        h = self.affine(h)
        return h

class Model(PFC.Module):
    def __init__(self):
        self.model0 = Model0()
        self.model1 = Model1()
    def __call__(self, x, test=False):
        h = x if test else F.image_augmentation(x, flip_lr=True, angle=0.26)
        h = self.model0(h, test)
        h = self.model1(h, test)
        return h

def main():
    # Context
    ctx = get_extension_context("cudnn", device_id="0")
    nn.set_default_context(ctx)
    nn.auto_forward(False)
    # Inputs
    b, c, h, w = 64, 1, 28, 28
    x = nn.Variable([b, c, h, w])
    t = nn.Variable([b, 1])
    vx = nn.Variable([b, c, h, w])
    vt = nn.Variable([b, 1])
    # Model
    model = Model()
    pred = model(x)
    loss = F.softmax_cross_entropy(pred, t)
    vpred = model(vx, test=True)
    verror = F.top_n_error(vpred, vt)
    # Check parameters
    for k, v in model.get_parameters(grad_only=False).items():
        print(k, v)
    # Solver
    solver = S.Adam()
    solver.set_parameters(model.get_parameters())
    # Data Iterator
    tdi = data_iterator_mnist(b, train=True)
    vdi = data_iterator_mnist(b, train=False)
    # Monitor
    monitor = Monitor("tmp.monitor")
    monitor_loss = MonitorSeries("Training loss", monitor, interval=10)
    monitor_verr = MonitorSeries("Test error", monitor, interval=1)

    # Training loop
    for e in range(1):
        for j in range(tdi.size // b):
            i = e * tdi.size // b + j
            x.d, t.d = tdi.next()
            solver.zero_grad()
            loss.forward(clear_no_need_grad=True)
            loss.backward(clear_buffer=True)
            solver.update()
            monitor_loss.add(i, loss.d)
        error = 0.0
        for _ in range(vdi.size // b):
            vx.d, vt.d = vdi.next()
            verror.forward(clear_buffer=True)
            error += verror.d
        error /= vdi.size // b
        monitor_verr.add(i, error)

    # Check Save/Load
    model.save_parameters("model.h5")
    model1 = Model()
    model1.load_parameters("model.h5")
    params0 = model.get_parameters(grad_only=False)
    params1 = model1.get_parameters(grad_only=False)
    for i0, i1 in zip(params0.items(), params1.items()):
        k0, v0 = i0
        k1, v1 = i1
        print(k0, k1, np.allclose(v0.d, v1.d))

if __name__ == '__main__':
    main()
