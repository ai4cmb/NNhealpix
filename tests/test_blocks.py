# -*- encoding: utf-8 -*-

import nnhealpix
import nnhealpix.layers
import tensorflow as tf
import numpy as np
import healpy


def test_dgrade_block():
    input_nside = 4
    output_nside = 1
    m = np.arange(healpy.nside2npix(input_nside))
    inputs = tf.keras.layers.Input((len(m), 1))
    x = nnhealpix.layers.Dgrade(input_nside, output_nside)(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    model.compile(loss=tf.keras.losses.mse, optimizer=tf.keras.optimizers.SGD(lr=0.01))

    mtensor = m.reshape(1, len(m), 1)
    out = model.predict(mtensor).reshape(healpy.nside2npix(output_nside))

    assert np.allclose(
        out,
        np.array(
            [
                29.0625,
                32.4375,
                35.8125,
                39.1875,
                93.75,
                91.75,
                95.75,
                99.75,
                152.0625,
                155.4375,
                158.8125,
                162.1875,
            ]
        ),
    )


def test_maxpooling_block():
    input_nside = 4
    output_nside = 1
    m = np.arange(healpy.nside2npix(input_nside))
    inputs = tf.keras.layers.Input((len(m), 1))
    x = nnhealpix.layers.MaxPooling(input_nside, output_nside)(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    model.compile(loss=tf.keras.losses.mse, optimizer=tf.keras.optimizers.SGD(lr=0.01))

    mtensor = m.reshape(1, len(m), 1)
    out = model.predict(mtensor).reshape(healpy.nside2npix(output_nside))
    print(out)
    assert np.allclose(
        out,
        np.array(
            [
                74.0,
                78.0,
                82.0,
                86.0,
                136.0,
                140.0,
                144.0,
                148.0,
                188.0,
                189.0,
                190.0,
                191.0,
            ]
        ),
    )


def test_convneighbours_block():
    input_nside = 1
    output_nside = 1
    m = np.arange(healpy.nside2npix(input_nside))
    inputs = tf.keras.layers.Input((len(m), 1))
    x = nnhealpix.layers.ConvNeighbours(input_nside, filters=1, kernel_size=9)(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    model.compile(loss=tf.keras.losses.mse, optimizer=tf.keras.optimizers.SGD(lr=0.01))
    model.layers[2].set_weights(
        [
            np.array(
                [
                    [[1.0]],
                    [[2.0]],
                    [[3.0]],
                    [[4.0]],
                    [[5.0]],
                    [[6.0]],
                    [[7.0]],
                    [[8.0]],
                    [[9.0]],
                ]
            )
        ]
    )

    mtensor = m.reshape(1, len(m), 1)
    out = model.predict(mtensor).reshape(healpy.nside2npix(output_nside))
    print(out)
    assert np.allclose(
        out, np.array([148, 167, 182, 161, 158, 153, 184, 187, 238, 265, 264, 243])
    )
