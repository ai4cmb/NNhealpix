import healpy as hp
import numpy as np
import os.path

def dgrade(nside_in, nside_out):
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
    if hp.isnsideok(nside, nest=True)==False:
        raise ValueError('nside is not valid')
    pix_array = hp.pixelfunc.get_all_neighbours(nside, ipix)
    pix_num = np.insert(pix_array, 4, ipix)
    return pix_num

def filter9(nside):
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
    file_out = os.path.join(
        os.path.dirname(__file__),
        'ancillary_files/'+string+'.npy')
    if os.path.isfile(file_out)==False:
        np.save(file_out, array)

def read_filter9(nside):
    file_in = os.path.join(
        os.path.dirname(__file__),
        'ancillary_files/filter9_nside{}.npy'.format(nside))
    indices = np.load(file_in)
    return indices

def read_dgrade(nside_in, nside_out):
    file_in = os.path.join(
        os.path.dirname(__file__),
        'ancillary_files/dgrade_from{}_to{}.npy'.format(nside_in, nside_out))
    indices = np.load(file_in)
    return indices
