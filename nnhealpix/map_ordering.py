import healpy as hp
import numpy as np
import os.path

""" Functions to create and save the arrays defining the map ordering to
perfrom convolution.
"""

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

    if hp.isnsideok(nside_in, nest=True)==False:
        raise ValueError('input nside is not valid')
    if hp.isnsideok(nside_out, nest=True)==False:
        raise ValueError('output nside in not valid')
    m_in = np.arange(hp.nside2npix(nside_in))
    m_out = np.arange(hp.nside2npix(nside_out))
    m_out_nside_in = hp.ud_grade(m_out, nside_in)
    order_out = []
    for p in m_out:
        pix_list = m_in[m_out_nside_in==p]
        order_out.append(pix_list)
    order_out = np.array(order_out)
    order_out = order_out.flatten()
    string = 'dgrade_from{}_to{}'.format(nside_in, nside_out)
    write_ancillary_file(string, order_out)
    return order_out

def pixel_first_neighbours(ipix, nside):
    """ find first pixel get_all_neighbours in the healix ring scheme
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
    if hp.isnsideok(nside, nest=True)==False:
        raise ValueError('nside is not valid')
    pix_array = hp.pixelfunc.get_all_neighbours(nside, ipix)
    pix_num = np.insert(pix_array, 4, ipix)
    return pix_num

def filter9(nside):
    """ map ordering to implement a convolutional neural network with a
    kernel convolving the first neighbour of each pixel on an healpix map

    Parameters
    ----------
    nside: integer
        Nside parameter for the input map.
        Must be a valid healpix Nside value

    Returns
    -------
    filter9 : array
        array defining the re-ordering of the input map to perform the
        convolution
    """

    if hp.isnsideok(nside, nest=True)==False:
        raise ValueError('nside is not valid')
    filter9 = []
    for i in range(hp.nside2npix(nside)):
        filter9.append(pixel_first_neighbours(i, nside))
    filter9 = np.array(filter9)
    filter9 = filter9.flatten()
    filter9[filter9==-1] = hp.nside2npix(nside)
    string = 'filter9_nside{}'.format(nside)
    write_ancillary_file(string, filter9)
    return filter9

def write_ancillary_file(string, array):
    """ writes the ordering array on disk

    Parameters
    ----------
    string: string
        file name. The output will be saved as string.npy
    array: array
        numpy array to be saved on disk
    """

    file_out = os.path.join(
        os.path.dirname(__file__),
        'ancillary_files/'+string+'.npy')
    if os.path.isfile(file_out)==False:
        np.save(file_out, array)

def read_filter9(nside):
    """ reads from disk the ordering array to perform the first neighbours
    filtering

    Parameters
    ----------
    nside: integer
        Nside parameter for which the ordering must be retrieved.
        Must be a valid healpix Nside value
    """

    file_in = os.path.join(
        os.path.dirname(__file__),
        'ancillary_files/filter9_nside{}.npy'.format(nside))
    indices = np.load(file_in)
    return indices

def read_dgrade(nside_in, nside_out):
    """ reads from disk the ordering array to perform the down grade of
    an healpix map

    Parameters
    ----------
    nside_in, nside_out: integers
        Nsides parameters for which the ordering must be retrieved.
        Must be valid healpix Nside values
    """

    file_in = os.path.join(
        os.path.dirname(__file__),
        'ancillary_files/dgrade_from{}_to{}.npy'.format(nside_in, nside_out))
    indices = np.load(file_in)
    return indices
