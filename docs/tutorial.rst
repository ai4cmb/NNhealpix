Tutorial
========

In this tutorial, we present an overview of *nnhealpix*. The tutorial
is best run in a Jupyter notebook, or using IPython.

A hands-on example: recognizing digits on the sphere
----------------------------------------------------

*nnhealpix* provides several algorithms to work with Convolutional
 Neural Networks (CNNs) on HEALPix maps. In this section we introduce
 the most important algorithm, the *convolution filter*.

 We begin importing the libraries we are going to use::

   import numpy as np, healpy as hp
   import matplotlib.pylab as plt
   import keras

The *nnhealpix* library is structured using sub-modules, so we need
to load a few of them. For this tutorial, we need the convolutional
layers and a few visualization routines::

  import nnhealpix.layers.blocks as blocks
  import nnhealpix.visual as vis

Applying filters to maps
------------------------
