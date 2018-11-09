import numpy as np
import keras
import os.path
from keras.layers import Conv1D
import keras.backend as K
from keras.engine.topology import Layer
import tensorflow as tf
import nnhealpix.map_ordering

class OrderMap(Layer):
    """ defines the keras layer that reorders the inputs map to then perfrom
    convolution on them
    """

    def __init__(self, indices, **kwargs):
        self.input_indices = np.array(indices, dtype='int32')
        Kindices = K.variable(indices, dtype='int32')
        self.indices = Kindices
        super(OrderMap, self).__init__(**kwargs)
    def build(self, input_shape):
        self.in_shape = input_shape
        super(OrderMap, self).build(input_shape)
    def call(self, x):
        x = tf.to_float(x)
        zero = tf.fill([tf.shape(x)[0], 1, tf.shape(x)[2]], 0.)
        x1 = tf.concat([x, zero], axis=1)
        reordered = tf.gather(x1, self.indices, axis=1)
        self.output_dim = reordered.shape
        return reordered
    def compute_output_shape(self, input_shape):
        return (
            input_shape[0], int(self.output_dim[1]), int(self.output_dim[2]))
    def get_config(self):
        config = super(OrderMap, self).get_config()
        config.update({'indices': self.input_indices})
        return config

def Dgrade(nside_in, nside_out):
    """ keras layer performing a downgrade of input maps

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
        '../ancillary_files/dgrade_from{}_to{}'.format(nside_in, nside_out))
    try:
        pixel_indices = np.load(file_in)
    except:
        pixel_indices = nnhealpix.map_ordering.dgrade(nside_in, nside_out)
    def f(x):
        y = OrderMap(pixel_indices)(x)
        pool_size=int((nside_in/nside_out)**2.)
        y = keras.layers.AveragePooling1D(pool_size=pool_size)(y)
        return y
    return f

def MaxPooling(nside_in, nside_out):
        """ keras layer performing a downgrade+maxpooling of input maps

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
            '../ancillary_files/dgrade_from{}_to{}'.format(nside_in, nside_out))
        try:
            pixel_indices = np.load(file_in)
        except:
            pixel_indices = nnhealpix.map_ordering.dgrade(nside_in, nside_out)
        def f(x):
            y = OrderMap(pixel_indices)(x)
            pool_size=int((nside_in/nside_out)**2.)
            y = keras.layers.MaxPooling1D(pool_size=pool_size)(y)
            return y
        return f


def ConvPixel(nside_in, nside_out, filters, use_bias=False, trainable=True):
    """ keras layer performing a downgrade+convolution of input maps

    Parameters
    ----------
    nside_in : integer
        Nside parameter for the input maps.
        Must be a valid healpix Nside value
    nside_out: integer
        Nside parameter for the output maps.
        Must be a valid healpix Nside value
    filters : integer
        number of filters in the convolution
    use_bias : bool (default is False)
        whether the layer uses a bias vector
    trainable : bool (default is True)
        wheter this is a trainable layer
    """

    file_in = os.path.join(
        os.path.dirname(__file__),
        '../ancillary_files/dgrade_from{}_to{}'.format(nside_in, nside_out))
    try:
        pixel_indices = np.load(file_in)
    except:
        pixel_indices = nnhealpix.map_ordering.dgrade(nside_in, nside_out)
    def f(x):
        y = OrderMap(pixel_indices)(x)
        kernel_size = int((nside_in/nside_out)**2.)
        y = keras.layers.Conv1D(
            filters, kernel_size=kernel_size, strides=kernel_size,
            use_bias=use_bias, trainable=trainable)(y)
        return y
    return f

def ConvNeighbours(nside, kernel_size, filters, use_bias=False, trainable=True):
    """ keras layer performing the pixel neighbour covolution

    Parameters
    ----------
    nside : integer
        Nside parameter for the input maps.
        Must be a valid healpix Nside value
    kernel_size: integer
        dimension of the kernel.
        Must be a valid number. For now only kernel_size=9 is admitted,
        corresponding to the first neighbours convolution
    filters : integer
        number of filters in the convolution
    use_bias : bool (default is False)
        whether the layer uses a bias vector
    trainable : bool (default is True)
        wheter this is a trainable layer
    """

    if kernel_size!=9:
        raise ValueError('kernel size must be 9')
    file_in = os.path.join(
        os.path.dirname(__file__),
        '../ancillary_files/filter{}_nside{}.npy'.format(kernel_size, nside))
    try:
        pixel_indices = np.load(file_in)
    except:
        pixel_indices = nnhealpix.map_ordering.filter(nside)
    def f(x):
        y = OrderMap(pixel_indices)(x)
        y = keras.layers.Conv1D(
            filters, kernel_size=kernel_size, strides=kernel_size,
            use_bias=use_bias, trainable=trainable)(y)
        return y
    return f

def ResConvNeighbours(
    nside, kernel_size, filters, use_bias=False, trainable=True):
    """ keras layer performing a residual convolution block, convolving twice
    over neighbour pixels

    Parameters
    ----------
    nside : integer
        Nside parameter for the input maps.
        Must be a valid healpix Nside value
    kernel_size: integer
        dimension of the kernel.
        Must be a valid number. For now only kernel_size=9 is admitted,
        corresponding to the first neighbours convolution
    filters : integer
        number of filters in the convolution
    use_bias : bool (default is False)
        whether the layer uses a bias vector
    trainable : bool (default is True)
        wheter this is a trainable layer
    """

    def f(x):
        shortcut = x
        y = ConvNeighbours(
            nside, kernel_size, filters, use_bias=use_bias, trainable=trainable)(x)
        y = keras.layers.BatchNormalization(axis=1, epsilon=1e-5)(y)
        y = keras.layers.Activation('relu')(y)
        y = ConvNeighbours(
            nside, kernel_size, filters, use_bias=use_bias, trainable=trainable)(y)
        y = keras.layers.BatchNormalization(axis=1, epsilon=1e-5)(y)
        y = keras.layers.Add()([y, shortcut])
        y = keras.layers.Activation('relu')(y)
        return y
    return f
