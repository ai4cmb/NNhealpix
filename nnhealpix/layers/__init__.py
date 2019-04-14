# -*- encoding: utf-8 -*-

import numpy as np
import keras
import os.path
from keras.layers import Conv1D
import keras.backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import nnhealpix as nnh


class OrderMap(Layer):
    """Defines a Keras layer able to operate on HEALPix maps.

    This layer has two purposes:

    * It reorders the input map so that neighbour pixels are all adjacent;
    * It performs 1D convolution using whatever the backend offers.
    """

    def __init__(self, indices, **kwargs):
        self.input_indices = np.array(indices, dtype="int32")
        Kindices = K.variable(indices, dtype="int32")
        self.indices = Kindices
        super(OrderMap, self).__init__(**kwargs)

    def build(self, input_shape):
        "Create the weights for the layer"
        self.in_shape = input_shape
        super(OrderMap, self).build(input_shape)

    def call(self, x):
        "Implement the layer's logic"
        x = tf.to_float(x)
        zero = tf.fill([tf.shape(x)[0], 1, tf.shape(x)[2]], 0.0)
        x1 = tf.concat([x, zero], axis=1)
        reordered = tf.gather(x1, self.indices, axis=1)
        self.output_dim = reordered.shape
        return reordered

    def compute_output_shape(self, input_shape):
        "Compute the shape of the layer's output."
        return (input_shape[0], int(self.output_dim[1]), int(self.output_dim[2]))

    def get_config(self):
        "Return a dictionary containing the configuration for the layer."
        config = super(OrderMap, self).get_config()
        config.update({"indices": self.input_indices})
        return config


def Dgrade(nside_in, nside_out):
    """Keras layer performing a downgrade of input maps

    Parameters
    ----------
    nside_in : integer
        Nside parameter for the input maps.
        Must be a valid healpix Nside value
    nside_out: integer
        Nside parameter for the output maps.
        Must be a valid healpix Nside value
    """

    file_in = os.path.join(
        os.path.dirname(__file__),
        "..",
        "ancillary_files",
        "dgrade_from{}_to{}.npy".format(nside_in, nside_out),
    )
    try:
        pixel_indices = np.load(file_in)
    except:
        pixel_indices = nnh.dgrade(nside_in, nside_out)

    def f(x):
        y = OrderMap(pixel_indices)(x)
        pool_size = int((nside_in / nside_out) ** 2.0)
        y = keras.layers.AveragePooling1D(pool_size=pool_size)(y)
        return y

    return f


def Pooling(nside_in, nside_out, layer1D, *args, **kwargs):
    """Keras layer performing a downgrade+custom pooling of input maps

    Args:
        * nside_in (integer): ``NSIDE`` parameter for the input maps.
        * nside_out (integer): ``NSIDE`` parameter for the output maps.
        * layer1D (layer object): a 1-D layer operation, like
          :code:`kkeras.layers.MaxPooling1D`
        * args (any): Positional arguments to be passed to :code:`layer1D`
        * kwargs: keyword arguments to be passed to
          :code:`layer1D`. The keyword :code:`pool_size` should not be
          included, as it is handled automatically.
    """

    file_in = os.path.join(
        os.path.dirname(__file__),
        "..",
        "ancillary_files",
        "dgrade_from{}_to{}.npy".format(nside_in, nside_out),
    )
    try:
        pixel_indices = np.load(file_in)
    except:
        pixel_indices = nnh.dgrade(nside_in, nside_out)

    def f(x):
        y = OrderMap(pixel_indices)(x)
        pool_size = int((nside_in / nside_out) ** 2.0)
        kwargs["pool_size"] = pool_size
        y = layer1D(*args, **kwargs)(y)
        return y

    return f


def MaxPooling(nside_in, nside_out):
    """Keras layer performing a downgrading+maxpooling of input maps

    Args:
        * nside_in (integer): ``NSIDE`` parameter for the input maps.
        * nside_out (integer): ``NSIDE`` parameter for the output maps.
    """

    return Pooling(nside_in, nside_out, keras.layers.MaxPooling1D)


def AveragePooling(nside_in, nside_out):
    """Keras layer performing a downgrading+averaging of input maps

    Args:
        * nside_in (integer): ``NSIDE`` parameter for the input maps.
        * nside_out (integer): ``NSIDE`` parameter for the output maps.
    """

    return Pooling(nside_in, nside_out, keras.layers.AveragePooling1D)


def DegradeAndConvNeighbours(
    nside_in, nside_out, filters, use_bias=False, trainable=True
):
    """Keras layer performing a downgrading and convolution of input maps.

    Args:
        * nside_in (integer): ``NSIDE`` parameter for the input maps.
        * nside_out (integer): ``NSIDE`` parameter for the output maps.
        * filters (integer): Number of filters to use in the
          convolution
        * use_bias (bool): Whether the layer uses a bias vector or
          not. Default is ``False``.
        * trainable (bool): Wheter this is a trainable layer or
          not. Default is ``True``.

    """

    file_in = os.path.join(
        os.path.dirname(__file__),
        "..",
        "ancillary_files",
        "dgrade_from{}_to{}.npy".format(nside_in, nside_out),
    )
    try:
        pixel_indices = np.load(file_in)
    except:
        pixel_indices = nnh.dgrade(nside_in, nside_out)

    def f(x):
        y = OrderMap(pixel_indices)(x)
        kernel_size = int((nside_in / nside_out) ** 2.0)
        y = keras.layers.Conv1D(
            filters,
            kernel_size=kernel_size,
            strides=kernel_size,
            use_bias=use_bias,
            trainable=trainable,
            kernel_initializer="random_uniform",
        )(y)
        return y

    return f


def ConvNeighbours(nside, kernel_size, filters, use_bias=False, trainable=True):
    """Keras layer to perform pixel neighbour convolution.

    Args:
        * nside(integer): ``NSIDE`` parameter for the input maps.
        * kernel_size(integer): Dimension of the kernel. Currently,
          NNhealpix only supports ``kernelsize = 9`` (first-order
          convolution).
        * filters(integer): Number of filters to use in the
          convolution
        * use_bias(bool): Whether the layer uses a bias vector or
          not. Default is ``False``.
        * trainable(bool): Wheter this is a trainable layer or
          not. Default is ``True``.

    """

    if kernel_size != 9:
        raise ValueError("kernel size must be 9")

    file_in = nnh.filter_file_name(nside, kernel_size)
    try:
        pixel_indices = np.load(file_in)
    except:
        pixel_indices = nnh.filter(nside)

    def f(x):
        y = OrderMap(pixel_indices)(x)
        y = keras.layers.Conv1D(
            filters,
            kernel_size=kernel_size,
            strides=kernel_size,
            use_bias=use_bias,
            trainable=trainable,
        )(y)
        return y

    return f
