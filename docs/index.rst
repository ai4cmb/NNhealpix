.. HealpixNN documentation master file, created by
   sphinx-quickstart on Thu Oct 11 16:07:52 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to HealpixNN's documentation!
=====================================

NNHealpix is a Python package for the Keras library that implements
convolution and pooling layers defined on a HEALPix sphere. This
library can be used to define convolutional neural networks that
operate on the sphere; so far it has mainly been used for analysis of
the full/partial celestial sky.

Here is a list of the current features/limitations of the library:

* Convolution over 7/8 neighbours of a pixel, using
  :func:`nnhealpix.layers.ConvNeighbours`;
* Pooling over pixel neighbours, using
  :func:`nnhealpix.layers.Pooling`,
  :func:`nnhealpix.layers.AveragePooling`, and
  :func:`nnhealpix.layers.MaxPooling`;
* NNHealpix implements layers that are *not* rotationally equivariant;
  this can usually be overcomed using data augmentation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorial
   layers
   nnhealpix
   projections
   visualization

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
