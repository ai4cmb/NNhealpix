# -*- encoding: utf-8 -*-

"""Neural networks on the Healpix sphere in Keras.

The NNhealpix module implements Keras layers that work on a Healpix
sphere.

"""

import healpy as hp
import numpy as np
import os.path
import numba

__version__ = "0.1.0"

DATADIR = os.path.expanduser(
    os.path.join("~", ".config", "nnhealpix", "ancillary_files")
)
"Path to the cache files used by NNhealpix."


def dgrade_file_name(nside_in, nside_out):
    "Return the full path to the datafile used to perform downgrading."
    return os.path.join(DATADIR, "dgrade_from{}_to{}.npz".format(nside_in, nside_out))


def filter_file_name(nside, order):
    "Return the full path to the datafile used to apply the convolution filter."
    return os.path.join(DATADIR, "filter_nside{0}_order{1}.npz".format(nside, order))


def __make_indices(x, y, xmin, xmax, ymin, ymax):
    num = (xmax - xmin) * (ymax - ymin)
    idx = 0
    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            x[idx] = i
            y[idx] = j

            idx += 1


def dgrade(nside_in, nside_out):
    """Return the list of indexes used to downgrade a HEALPix map

    Args:
        * nside_in (int): ``NSIDE`` for the input map. It must be a
          valid HEALPix value.
        * nside_out (int): ``NSIDE`` for the output map. It must be a
          valid HEALPix value.

    Returns:
        Array defining the re-ordering of the input map to obtain a map with
        ``nside_out`` while performing convolution

    Example::

        import numpy, healpy, nnhealpix

        nside_in = 2
        nside_out = 1
        idx = nnhealpix.dgrade(nside_in, nside_out)

    """

    assert hp.isnsideok(
        nside_in, nest=True
    ), "invalid input nside {0} in call to dgrade".format(nside_in)

    assert hp.isnsideok(
        nside_out, nest=True
    ), "invalid output nside {0} in call to dgrade".format(nside_out)

    assert nside_out < nside_in

    try:
        return __read_dgrade_cache(nside_in, nside_out)
    except FileNotFoundError:
        fact = nside_in // nside_out
        pixels = hp.nside2npix(nside_out)
        stride = fact * fact
        result = np.empty(pixels * stride, dtype="int")
        x, y, f = hp.pix2xyf(nside_out, np.arange(pixels))

        i = np.empty(stride, dtype="int")
        j = np.empty(stride, dtype="int")
        f_spread = np.empty(stride, dtype="int")
        for pixnum in range(pixels):
            __make_indices(
                i,
                j,
                fact * x[pixnum],
                fact * (x[pixnum] + 1),
                fact * y[pixnum],
                fact * (y[pixnum] + 1),
            )
            f_spread[:] = f[pixnum]
            result[(pixnum * stride) : ((pixnum + 1) * stride)] = hp.xyf2pix(
                nside_in, i, j, f_spread
            )

        file_name = dgrade_file_name(nside_in, nside_out)
        __write_ancillary_file(file_name, result)
        return result


def pixel_1st_neighbours(ipix, nside):
    """Return the indexes of the neighbour pixels in a HEALPix ``RING`` map.

    Args:
        * ipix (int): pixel number for which to find the neighbours
        * nside (integer): ``NSIDE`` parameter defining the resolution
          of the HEALPix map. It must be a valid healpix Nside value

    Returns:
        An array of integers containing 9 elements: the first element
        is the argument ``ipix`` itself, and the following 8 are the
        indexes of its neighbours. The order of the array is (SW, W,
        NW, N, ipix, NE, E, SE, S). If the pixel has only 7
        neighbours, the missing position will be filled with -1.

    >>> nnhealpix.pixel_1st_neighbours(1, 16)
    array([ 1,  6,  5,  0,  3,  2,  8,  7, 16])

    """
    assert hp.isnsideok(
        nside, nest=True
    ), "invalid nside {0} in call to pixel_1st_neighbours".format(nside)

    pix_array = np.empty(9, dtype="int")
    pix_array[0] = ipix
    pix_array[1:9] = hp.pixelfunc.get_all_neighbours(nside, ipix)
    return pix_array


def pixel_2nd_neighbours(ipix, nside):
    """Return the indexes of the neighbour pixels in a HEALPix ``RING`` map.

    Args:
        * ipix (int): pixel number for which to find the neighbours
        * nside (integer): ``NSIDE`` parameter defining the resolution
          of the HEALPix map. It must be a valid healpix Nside value

    Returns:
        An array of integers containing 25 elements. Any missing
        position will be filled with -1.

    .. note:: This function does not work yet, see
              https://github.com/NicolettaK/HealpixNN/issues/3

    """
    pixels = set([ipix])
    for centerpix in hp.pixelfunc.get_all_neighbours(nside, ipix):
        if centerpix == -1:
            # This is wrong, we should employ a well-defined strategy here
            continue

        for sidepix in hp.pixelfunc.get_all_neighbours(nside, centerpix):
            pixels.add(sidepix)

    result = np.empty(len(pixels), dtype="int")
    for i, item in enumerate(sorted(pixels)):
        result[i] = item

        return result


def neighbours25(nside, ipix):
    nfn = hp.pixelfunc.get_all_neighbours
    result = np.empty(25, dtype="int")

    # Center of the 5×5 tile
    result[0] = ipix

    # The first ring is easy to find
    result[1:9] = nfn(nside, ipix)

    # For the pixels in the range 9…24 things are more complicated
    # We leave the corners out, as they need more checks

    loc = nfn(nside, result[1])
    result[10:12] = loc[0:2]
    result[9] = loc[7]

    result[13:16] = nfn(nside, result[3])[1:4]
    result[17:20] = nfn(nside, result[5])[3:6]
    result[21:24] = nfn(nside, result[7])[5:8]

    # We are left with the outermost corners: #12, #16, #20, #24 The
    # position of the corner can always be determined by its two
    # adjacent pixels that are not along the N/S/E/W directions

    for corneridx, pair in [
        (12, [(11, 2), (13, 0)]),
        (16, [(15, 4), (17, 2)]),
        (20, [(19, 6), (21, 4)]),
        (24, [(9, 6), (23, 0)]),
    ]:
        first, second = pair
        if result[first[0]] >= 0:
            result[corneridx] = nfn(nside, result[first[0]])[first[1]]
        elif result[second[0]] >= 0:
            result[corneridx] = nfn(nside, result[second[0]])[second[1]]
        else:
            result[corneridx] = -1

    return result


def filter(nside, order=1):
    """map ordering to implement a convolutional neural network with a
    kernel convolving the first neighbour of each pixel on an healpix map

    Args:
        * nside (int): ``NSIDE`` for the input map. It must be a valid
          HEALPix value.
        * order (int): the order level. Currently it can only be 1.

    Returns:
        An array of integers defining the re-ordering of the input map
        to perform the convolution.

    """

    assert hp.isnsideok(
        nside, nest=True
    ), "invalid nside ({0}) in call to filter".format(nside)

    try:
        return __read_filter_cache(nside, order)
    except FileNotFoundError:
        order_fn = {1: pixel_1st_neighbours, 2: pixel_2nd_neighbours}

        assert (
            order in order_fn.keys()
        ), "invalid order ({0}) passed to filter, valid values are {1}".format(
            order, ", ".join([str(x) for x in order_fn.keys()])
        )

        result = np.empty(0, dtype="int")
        fn = order_fn[order]
        for i in range(hp.nside2npix(nside)):
            result = np.concatenate((result, fn(i, nside)))

        result[result == -1] = hp.nside2npix(nside)
        file_name = filter_file_name(nside, order)
        __write_ancillary_file(file_name, result)
        return result


def __write_ancillary_file(file_name, array):
    """Writes an array in a NumPy file.

    This is a low-level function used to write cache files to disk. It
    should never be called directly.

    Args:
        * file_name (str): Name of the file. It should have `.npz` extension
        * array: The NumPy array to be saved on disk

    """

    # Be sure that the path were we want to save the file already exists
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    if not os.path.isfile(file_name):
        np.savez_compressed(file_name, arr=array)


def __read_filter_cache(nside, order):
    """Load the array used to to perform neighbour filtering

    Args:
        * nside (int): desired value for ``NSIDE``.

    Returns:
        An array containing the requested array.    
    """

    file_name = filter_file_name(nside, order)
    with np.load(file_name) as f:
        return f["arr"]


def __read_dgrade_cache(nside_in, nside_out):
    """Load the array used to to downgrade a map

    Args:
        * nside_in (int): desired value for the input ``NSIDE``.
        * nside_out (int): desired value for the output ``NSIDE``.

    Returns:
        An array containing the requested array.    
    """

    file_name = dgrade_file_name(nside_in, nside_out)
    with np.load(file_name) as f:
        return f["arr"]
