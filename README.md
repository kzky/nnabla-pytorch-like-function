# [WIP] PyTorch-like function in NNabla

In NNabla, parametric functions (e.g., PF.convolution intenally holding trainable parameters) are defined like TensorFlow V1.x, i.e., trainable parameters are managed in a global scope of a process by using a dictionary of a name of parameter to a trainable parameter and the scope context. It is intuitively straightforward when writing a code while a bit complicated to manage trainable parameters since trainable parameters are not managed in a local scope of e.g., a class but globally, which sometimes hard to see a whole picture of a neural network. If trainable parameters are held in a class like PyTorch or Chainer, it is very easy to see a whole picture of a network while being redundant representation of a network when writing a code since we have to write two lines about a parametric function to be used i.e., in the *\_\_init\_\_* method and *\_\_call\_\_* (or *forward*) method of a class. There are pros and cons for each, but people seem to like PyTorch-like parametric function definition. Thus, here I describe how to write PyTorch-like parametric function in NNabla.


In this repository, I define the following NNabla's parametric functions in a PyTorch-like way.

- Convolution (Conv1d, Conv2d, Conv3d, ConvNd)
- Deconvolution (Deconv1d, Deconv2d, Deconv3d, DeconvNd)
- Affine (Linear)
- Embed
- BatchNorm (BatchNorm1d, BatchNorm2d, BatchNorm3d)


