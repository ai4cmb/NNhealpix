Tutorial
========

Build training and validation sets
----------------------------------

First we load the MNIST dataset

.. code:: ipython3

    import keras
    from keras.datasets import mnist
    from keras import backend as K
    K.set_image_dim_ordering('th')

    (X_train_2d, y_train), (X_val_2d, y_val) = mnist.load_data()

we remove the number 9 from the dataset, as 6 and 9 cannot be
distinguish once projected on the sphere.

.. code:: ipython3

    import numpy as np

    ok_train = np.where(y_train!=9)
    X_train_2d = X_train_2d[ok_train]
    y_train = y_train[ok_train]
    ok_val = np.where(y_val!=9)
    X_val_2d = X_val_2d[ok_val]
    y_val = y_val[ok_val]

We generate training and validation set by projecting MNIST 2D images
on the Healpix sphere (with Nside=16) in a random position, with
random rotation and dimension.

.. code:: ipython3

    import healpy as hp
    
    Ntrain = 10000
    Nval = 1000
    Nside = 16
    X_train_hp = np.zeros((Ntrain, hp.nside2npix(Nside)))
    X_val_hp = np.zeros((Nval, hp.nside2npix(Nside)))
    y_train_hp = np.zeros(Ntrain)
    y_val_hp = np.zeros(Nval)

.. code:: ipython3

    from keras.utils import np_utils
    import nnhealpix as nnhp
    
    dim_theta = [120., 180.]
    dim_phi = [120. , 360.]
    for i, (id_img, hp_img) in enumerate(nnhp.projectimages(X_train_2d, Nside, dim_theta, dim_phi, num=Ntrain)):
        X_train_hp[i, :] = hp_img
        y_train_hp[i] = y_train[id_img]
    for i, (id_img, hp_img) in enumerate(nnhp.projectimages(X_val_2d, Nside, dim_theta, dim_phi, num=Nval)):
        X_val_hp[i, :] = hp_img
        y_val_hp[i] = y_val[id_img]
    y_train = np_utils.to_categorical(y_train_hp)
    y_val = np_utils.to_categorical(y_val_hp)


Plot projected map

.. code:: ipython3

    import matplotlib.pylab as plt
    %matplotlib inline
    
    Nt = np.random.randint(Ntrain)
    fig = plt.figure(figsize=(14,4))
    hp.mollview(X_train_hp[Nt], sub=131, max=255, title='Mollview projection')
    hp.orthview(X_train_hp[Nt], sub=132, max=255, title='Orthographic projection')
    hp.orthview(X_train_hp[Nt], rot=[0, 90], sub=133, max=255, title='Orthographic projection (poles)')



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
    import nnhealpix.layers.blocks
    
    inputs = keras.layers.Input(shape)
    x = nnhealpix.layers.blocks.ConvNeighbours(Nside, filters=32, kernel_size=9)(inputs)
    x = keras.layers.Activation('relu')(x)
    x = nnhealpix.layers.blocks.MaxPooling(Nside, Nside//2)(x)
    x = nnhealpix.layers.blocks.ConvNeighbours(Nside//2, filters=32, kernel_size=9)(x)
    x = keras.layers.Activation('relu')(x)
    x = nnhealpix.layers.blocks.MaxPooling(Nside//2, Nside//4)(x)
    x = nnhealpix.layers.blocks.ConvNeighbours(Nside//4, filters=32, kernel_size=9)(x)
    x = keras.layers.Activation('relu')(x)
    x = nnhealpix.layers.blocks.MaxPooling(Nside//4, Nside//8)(x)
    x = nnhealpix.layers.blocks.ConvNeighbours(Nside//8, filters=32, kernel_size=9)(x)
    x = keras.layers.Activation('relu')(x)
    x = nnhealpix.layers.blocks.MaxPooling(Nside//8, Nside//16)(x)
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
    model = load_model('model_MNIST_tutorial.h5', custom_objects={'OrderMap': nnhealpix.layers.blocks.OrderMap})

.. code:: ipython3

    (X_train_2d, y_train), (X_test_2d, y_test) = mnist.load_data()
    ok_test = np.where(y_test!=9)
    X_test_2d = X_test_2d[ok_test]
    y_test = y_test[ok_test]
    Ntest = 1000
    Nside = 16
    X_test_hp = np.zeros((Ntest, hp.nside2npix(Nside)))
    y_test_hp = np.zeros(Ntest)

.. code:: ipython3

    dim_theta = [120., 180.]
    dim_phi = [120. , 360.]
    for i, (id_img, hp_img) in enumerate(nnhp.projectimages(X_test_2d, Nside, dim_theta, dim_phi, num=Ntest)):
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

    modelPT = load_model('model_MNIST_16x32_8x32_4x32_2x32_Ntrain100000_HVDn10_rndxrnd.h5', 
                              custom_objects={'OrderMap': nnhealpix.layers.blocks.OrderMap})
    modelPT.summary()


.. parsed-literal::

    [abriged]
    Total params: 46,857
    Trainable params: 46,857
    Non-trainable params: 0


.. code:: ipython3

    hyPT = read_history('history_MNIST_16x32_8x32_4x32_2x32_Ntrain100000_HVDn10_rndxrnd.npy')

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

    Nt = np.random.randint(Ntest)
    fig = plt.figure(figsize=(14,4))
    hp.mollview(X_test_hp[Nt], sub=131, max=255, title='Mollview projection')
    hp.orthview(X_test_hp[Nt], sub=132, max=255, title='Orthographic projection')
    hp.orthview(X_test_hp[Nt], rot=[0, 90], sub=133, max=255,
                title='Orthographic projection (poles)')

.. image:: images/output_37_0.png


.. code:: ipython3

    get_layer_output = K.function([modelPT.layers[0].input],
                                      [modelPT.layers[3].output])
    layer_output = get_layer_output([X_test[Nt:Nt+1]])[0]
    filt_maps = layer_output[0].T

.. code:: ipython3

    fig = visual.plot_layer_output(filt_maps, cbar=True)

.. parsed-literal::

    Active nodes:  29


.. image:: images/output_39_1.png

