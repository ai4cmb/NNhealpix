Tutorial
========

Welcome to NNHealpix documentation! In this tutorial, we show how to
use the library to perform a classification exercise. We are going to
project handwritten digits on a sphere, and we will build a simple
Convolutional Neural Network (CNN) to recognize digits. We will use
Keras and the MNIST dataset.

Build training and validation sets
----------------------------------

We are going to use the MNIST database to build our training and
validation sets. MINST is a set of 28×28 grayscale images representing
handwritten digits, and it is a widely-used database for
classification problems in Machine Learning. The MNIST dataset is
available in Keras, so we simply load it:

.. code:: ipython3

    import keras
    from keras.datasets import mnist
    from keras import backend as K
    K.set_image_dim_ordering('th')

    (X_train_2d, y_train), (X_val_2d, y_val) = mnist.load_data()

As we are going to randomly rotate the digits while projecting them on
the sphere, it would be difficult to distinguish 6 from 9. Therefore,
we are discarding the digit 9 from the test set:

.. code:: ipython3

    import numpy as np

    ok_train = np.where(y_train!=9)
    X_train_2d = X_train_2d[ok_train]
    y_train = y_train[ok_train]
    
    ok_val = np.where(y_val!=9)
    X_val_2d = X_val_2d[ok_val]
    y_val = y_val[ok_val]

The MNIST images are supposed to be flat 28×28 bitmaps, but we need
them projected on a HEALPix sphere. As this is a common task,
NNHealpix provides the function :func:`nnhealpix.visual.projectimages` for
this purpose. We use ``NSIDE=16``, and we apply a random rotation to
each of them. The angular size of each image is random as well.

.. code:: ipython3

    import healpy as hp
    from keras.utils import np_utils
    import nnhealpix as nnhp
        
    NTRAIN, NVAL = 10000, 1000
    NSIDE = 16
    NPIX = hp.nside2npix(NSIDE)
    X_train_hp = np.zeros((NTRAIN, NPIX))
    X_val_hp = np.zeros((NVAL, NPIX))
    y_train_hp = np.zeros(NTRAIN)
    y_val_hp = np.zeros(NVAL)

    # Range of sizes for theta and phi
    dim_theta = [120., 180.]
    dim_phi = [120. , 360.]
    for i, (id_img, hp_img) in enumerate(nnhp.projectimages(X_train_2d, NSIDE, dim_theta, dim_phi, num=NTRAIN)):
        X_train_hp[i, :] = hp_img
        y_train_hp[i] = y_train[id_img]
    for i, (id_img, hp_img) in enumerate(nnhp.projectimages(X_val_2d, NSIDE, dim_theta, dim_phi, num=NVAL)):
        X_val_hp[i, :] = hp_img
        y_val_hp[i] = y_val[id_img]
    y_train = np_utils.to_categorical(y_train_hp)
    y_val = np_utils.to_categorical(y_val_hp)


Plot projected map

.. code:: ipython3

    import matplotlib.pylab as plt
    %matplotlib inline
    
    NINDEX = np.random.randint(NTRAIN)
    fig = plt.figure(figsize=(14,4))
    hp.mollview(X_train_hp[NINDEX], sub=131, max=255, title='Mollview projection')
    hp.orthview(X_train_hp[NINDEX], sub=132, max=255, title='Orthographic projection')
    hp.orthview(X_train_hp[NINDEX], rot=[0, 90], sub=133, max=255, title='Orthographic projection (poles)')



.. image:: images/output_9_0.png


Reshape and normalize

.. code:: ipython3

    X_train = X_train_hp.reshape(X_train_hp.shape[0], len(X_train_hp[0]), 1).astype('float32')
    X_val = X_val_hp.reshape(X_val_hp.shape[0], len(X_val_hp[0]), 1).astype('float32')
    X_train = X_train / 255
    X_val = X_val / 255
    num_classes = y_train.shape[1]
    shape = (len(X_train_hp[0]), 1)

    
Build neural network and train
------------------------------

.. code:: ipython3

    import keras.layers
    import nnhealpix.layers
    
    inputs = keras.layers.Input(shape)
    x = nnhealpix.layers.ConvNeighbours(NSIDE, filters=32, kernel_size=9)(inputs)
    x = keras.layers.Activation('relu')(x)
    x = nnhealpix.layers.MaxPooling(NSIDE, NSIDE//2)(x)
    x = nnhealpix.layers.ConvNeighbours(NSIDE//2, filters=32, kernel_size=9)(x)
    x = keras.layers.Activation('relu')(x)
    x = nnhealpix.layers.MaxPooling(NSIDE//2, NSIDE//4)(x)
    x = nnhealpix.layers.ConvNeighbours(NSIDE//4, filters=32, kernel_size=9)(x)
    x = keras.layers.Activation('relu')(x)
    x = nnhealpix.layers.MaxPooling(NSIDE//4, NSIDE//8)(x)
    x = nnhealpix.layers.ConvNeighbours(NSIDE//8, filters=32, kernel_size=9)(x)
    x = keras.layers.Activation('relu')(x)
    x = nnhealpix.layers.MaxPooling(NSIDE//8, NSIDE//16)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(48)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dense(num_classes)(x)
    out = keras.layers.Activation('softmax')(x)
    model = keras.models.Model(inputs=inputs, outputs=out)
    opt = keras.optimizers.Adam(lr=0.001)
    model.compile(loss=keras.losses.mse, optimizer=opt, metrics=['accuracy'])
    model.summary()


.. parsed-literal::

    [abriged]
    Total params: 46,857
    Trainable params: 46,857
    Non-trainable params: 0


Train network

.. code:: ipython3

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=1)

Save trained network and history

.. code:: ipython3

    import pickle
    
    model.save('model_MNIST_tutorial.h5')
    pickle.dump(history.history, open('history_MNIST_tutorial.npy', 'wb'))

Check results
-------------

.. code:: ipython3

    from keras.models import load_model
    
    def read_history(history_file):
        with open(history_file, 'rb') as f:
            hy = pickle.load(f)
        return hy

.. code:: ipython3

    hy = read_history('history_MNIST_tutorial.npy')
    model = load_model('model_MNIST_tutorial.h5', custom_objects={'OrderMap': nnhealpix.layers.OrderMap})

.. code:: ipython3

    (X_train_2d, y_train), (X_test_2d, y_test) = mnist.load_data()
    ok_test = np.where(y_test!=9)
    X_test_2d = X_test_2d[ok_test]
    y_test = y_test[ok_test]
    
    NTEST = 1000
    NSIDE = 16
    NPIX = hp.nside2npix(NSIDE)
    X_test_hp = np.zeros((NTEST, NPIX))
    y_test_hp = np.zeros(NTEST)

.. code:: ipython3

    dim_theta = [120., 180.]
    dim_phi = [120. , 360.]
    for i, (id_img, hp_img) in enumerate(nnhp.projectimages(X_test_2d, NSIDE, dim_theta, dim_phi, num=NTEST)):
        X_test_hp[i, :] = hp_img
        y_test_hp[i] = y_test[id_img]
    y_test = np_utils.to_categorical(y_test_hp)

.. code:: ipython3

    X_test = X_test_hp.reshape(X_test_hp.shape[0], len(X_test_hp[0]), 1).astype('float32')
    X_test = X_test / 255

.. code:: ipython3

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("CNN Error: %.2f%%" % (100-scores[1]*100))

.. parsed-literal::

    CNN Error: 34.00%


Results are not good as we are training on a small training set and just
for few epochs.

.. code:: ipython3

    plt.plot(hy['acc'], color='blue', lw=3, label='train')
    plt.plot(hy['val_acc'], color='blue', ls='--', lw=3, label = 'validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()


.. image:: images/output_26_1.png


Load pre-trained model
----------------------

this is the network trained and tested in Krachmalnicoff & Tomasi 2019
(https://arxiv.org/abs/1902.04083).

.. code:: ipython3

    modelPT = load_model('model_MNIST_16x32_8x32_4x32_2x32_NTRAIN100000_HVDn10_rndxrnd.h5', 
                              custom_objects={'OrderMap': nnhealpix.layers.OrderMap})
    modelPT.summary()


.. parsed-literal::

    [abriged]
    Total params: 46,857
    Trainable params: 46,857
    Non-trainable params: 0


.. code:: ipython3

    hyPT = read_history('history_MNIST_16x32_8x32_4x32_2x32_NTRAIN100000_HVDn10_rndxrnd.npy')

.. code:: ipython3

    scoresPT = modelPT.evaluate(X_test, y_test, verbose=0)
    print("CNN Error: %.2f%%" % (100-scoresPT[1]*100))


.. parsed-literal::

    CNN Error: 5.40%


.. code:: ipython3

    plt.plot(hyPT['acc'], color='blue', lw=3, label='train')
    plt.plot(hyPT['val_acc'], color='blue', ls='--', lw=3, label = 'validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

.. image:: images/output_31_1.png


Visualize kernels and filtered maps
-----------------------------------

`nnhealpix.visual` allow to inspect the NN, by visualizing kernels and
fitered maps.

Here we plot the 32 filters of the first convolutional layers (layer
number 2).

.. code:: ipython3

    w = np.array(modelPT.layers[2].get_weights())
    wT = w[0, :, 0, :].T

.. code:: ipython3

    from nnhealpix import visual
    fig = visual.plot_filters(wT, cbar=True, vmin=-0, vmax=0.5, basesize=1)



.. image:: images/output_35_0.png


Here we chose a random map in the test set and we plot the effect of
the above filters of it, therefore the output of layer number 3.

.. code:: ipython3

    NINDEX = np.random.randint(NTEST)
    fig = plt.figure(figsize=(14,4))
    hp.mollview(X_test_hp[NINDEX], sub=131, max=255, title='Mollview projection')
    hp.orthview(X_test_hp[NINDEX], sub=132, max=255, title='Orthographic projection')
    hp.orthview(X_test_hp[NINDEX], rot=[0, 90], sub=133, max=255,
                title='Orthographic projection (poles)')

.. image:: images/output_37_0.png


.. code:: ipython3

    get_layer_output = K.function([modelPT.layers[0].input],
                                      [modelPT.layers[3].output])
    layer_output = get_layer_output([X_test[NINDEX:NINDEX+1]])[0]
    filt_maps = layer_output[0].T

.. code:: ipython3

    fig = visual.plot_layer_output(filt_maps, cbar=True)

.. parsed-literal::

    Active nodes:  29


.. image:: images/output_39_1.png

