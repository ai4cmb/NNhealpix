# -*- encoding: utf-8 -*-

import healpy as hp
import numpy as np
import os.path
import numba

DATADIR = os.path.expanduser(
    os.path.join("~", ".config", "nnhealpix", "ancillary_files")
)

""" Functions to create and save the arrays defining the map ordering to
perfrom convolution.
"""


def dgrade_file_name(nside_in, nside_out):
    return os.path.join(DATADIR, "dgrade_from{}_to{}.npz".format(nside_in, nside_out))


def filter_file_name(nside, order):
    return os.path.join(DATADIR, "filter_nside{0}_order{1}.npz".format(nside, order))


def make_indices(x, y, xmin, xmax, ymin, ymax):
    num = (xmax - xmin) * (ymax - ymin)
    idx = 0
    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            x[idx] = i
            y[idx] = j

            idx += 1


def dgrade(nside_in, nside_out):
    """ map ordering to down grade an healpix map

    Parameters
    ----------
    nside_in : integer
        Nside parameter for the input map.
        Must be a valid healpix Nside value
    nside_out: integer
        Nside parameter for the output map.
        Must be a valid healpix Nside value

    Returns
    -------
    order_out : array
        array defining the re-ordering of the input map to obtain a map with
        nside_out while performing convolution
    """

    assert hp.isnsideok(
        nside_in, nest=True
    ), "invalid input nside {0} in call to dgrade".format(nside_in)

    assert hp.isnsideok(
        nside_out, nest=True
    ), "invalid output nside {0} in call to dgrade".format(nside_out)

    assert nside_out < nside_in

    try:
        return read_dgrade(nside_in, nside_out)
    except FileNotFoundError:
        fact = nside_in // nside_out
        pixels = hp.nside2npix(nside_out)
        stride = fact * fact
        result = np.empty(pixels * stride)
        x, y, f = hp.pix2xyf(nside_out, np.arange(pixels))

        i = np.empty(stride, dtype="int")
        j = np.empty(stride, dtype="int")
        f_spread = np.empty(stride, dtype="int")
        for pixnum in range(pixels):
            make_indices(
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
        write_ancillary_file(file_name, result)
        return result


def pixel_1st_neighbours(ipix, nside):
    """ find first pixel get_all_neighbours in the healpix ring scheme
    Parameters
    ----------
    ipix : integer
        pixel number for which to find the neighbours
    nside : integer
        Nside parameter defining the healpix pixelization scheme.
        Must be a valid healpix Nside value

    Returns
    -------
    pix_num : array
        array with 9 elements, corresponding to the first neighbours and the
        ipix itself. The order of the array is
        (SW, W, NW, N, ipix, NE, E, SE, S)

    Notes
    -------
    If a neighbour does not exist (it can be the case for W, N, E and S) the
    corresponding pixel number will be -1
    """
    assert hp.isnsideok(
        nside, nest=True
    ), "invalid nside {0} in call to pixel_1st_neighbours".format(nside)

    pix_array = np.empty(9, dtype="int")
    pix_array[0] = ipix
    pix_array[1:9] = hp.pixelfunc.get_all_neighbours(nside, ipix)
    return pix_array


def pixel_2nd_neighbours(ipix, nside):
    """ find all the pixels within two circle rings from pixel with index `ipix`
    Parameters
    ----------
    ipix : integer
        pixel number for which to find the neighbours
    nside : integer
        Nside parameter defining the healpix pixelization scheme.
        Must be a valid healpix Nside value

    Returns
    -------
    pix_num : array
        array with 25 elements, corresponding to the neighbours and the
        ipix itself. The pixels are listed in ascending order

    Notes
    -------
    If a neighbour does not exist, the corresponding pixel number will be -1
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
    """ map ordering to implement a convolutional neural network with a
    kernel convolving the first neighbour of each pixel on an healpix map

    Parameters
    ----------
    nside: integer
        Nside parameter for the input map.
        Must be a valid healpix Nside value

    Returns
    -------
    filter : array
        array defining the re-ordering of the input map to perform the
        convolution
    """

    assert hp.isnsideok(
        nside, nest=True
    ), "invalid nside ({0}) in call to filter".format(nside)

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
    write_ancillary_file(file_name, result)
    return result


def write_ancillary_file(file_name, array):
    """ writes the ordering array on disk

    Parameters
    ----------
    file_name: string
        Name of the file. It should have `.npz` extension
    array: array
        numpy array to be saved on disk
    """

    if not os.path.isfile(file_name):
        np.savez_compressed(file_name, arr=array)


def read_filter(nside, order):
    """ reads from disk the ordering array to perform the first neighbours
    filtering

    Parameters
    ----------
    nside: integer
        Nside parameter for which the ordering must be retrieved.
        Must be a valid healpix Nside value
    """

    file_name = filter_file_name(nside, order)
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with np.load(file_name) as f:
        return f["arr"]


def read_dgrade(nside_in, nside_out):
    """ reads from disk the ordering array to perform the down grade of
    an healpix map

    Parameters
    ----------
    nside_in, nside_out: integers
        Nsides parameters for which the ordering must be retrieved.
        Must be valid healpix Nside values
    """

    file_name = dgrade_file_name(nside_in, nside_out)
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with np.load(file_name) as f:
        return f["arr"]
