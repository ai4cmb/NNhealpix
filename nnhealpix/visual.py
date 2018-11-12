import healpy as hp
import numpy as np
import keras
from keras.models import load_model
from nnhealpix.map_ordering import *
import matplotlib.pyplot as plt
import matplotlib


def filter_img(weights, order=1):
    '''Return a 2D image of a filter.

    Parameters
    ----------
    weights: array
        array of filter weights
    order: integer
        order of the filter (1st, 2nd neighbours)
        only 1 impented!

    Returns
    ------------
    img: array
        2D image of the input filter
    '''

    order_fn = {
        1: pixel_1st_neighbours,
        2: pixel_2nd_neighbours,
    }
    assert order in order_fn.keys(), \
        ("invalid order ({0}) passed to filter, valid values are {1}"
         .format(order, ', '.join([str(x) for x in order_fn.keys()])))
    fn = order_fn[order]
    nside = 16
    ipix = hp.ang2pix(nside, np.pi/2, 0)
    pix_num = fn(ipix, nside)
    m = np.zeros(hp.nside2npix(nside))+np.inf
    m[pix_num] = weights
    img = hp.gnomview(m, reso=5.2, rot=[0, 2],
        notext=True, return_projected_map=True)
    plt.close()
    return img

def map_img(map_in, order=1):
    '''Return a 2D image of a filter.

    Parameters
    ----------
    map: array
        map to plot

    Returns
    ------------
    map_img: array
        2D image of the input filter
    '''

    img = hp.mollview(map_in, notext=True, return_projected_map=True)
    plt.close()
    img = np.flip(img, axis=0)
    return img

def plot_filters(filters, cmap=None, cbar=False, min=None, max=None):
    '''plot a set of filters.

    Parameters
    ----------
    filters: array
        array of filters to plot
    cmap: color map
        if None a pick/black color map will be used in the plot
    cbar: boolean
        whether or not to add colorbar to the plot.
        Default is False

    Returns
    ------------
    fig: figure
    '''

    if not cmap:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#FEFAFA", "black"])
        cmap.set_bad('white')
        cmap.set_under('white', alpha=0)
    if len(filters.shape)==1:
        filters = filters.reshape(1, len(filters))
    nfilt = len(filters)
    filt_min = min
    filt_max = max
    if filt_min==None:
        filt_min = filters.min()
    if filt_max==None:
        filt_max = filters.max()
    if nfilt >= 8:
        ncol = 8
    else:
        ncol = nfilt
    nrow= round(nfilt/ncol+0.5)
    fig = plt.figure(figsize=(8, 4))
    axes = fig.subplots(nrows=nrow, ncols=ncol)
    if nfilt == 1:
        axess = [axes]
    else:
        axess = axes.flat
    for j, ax in enumerate(axess):
        filt = filter_img(filters[j])
        filt[np.where((filt<filt_min) & (filt!=-np.inf))] = filt_min
        filt[np.where((filt>filt_max) & (filt!=np.inf))] = filt_max
        im = ax.imshow(filt, vmin=filt_min, vmax=filt_max)
        ax.set_axis_off()
        im.set_cmap(cmap)
    fig.subplots_adjust(right=0.8)
    if cbar:
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
    return fig

def plot_layer_output(maps, cmap=None, cbar=False, min=None, max=None):
    '''plot a set of filters.

    Parameters
    ----------
    maps: array
        output maps of the layer to plot
    cmap: color map
        if None a pick/black color map will be used in the plot
    cbar: boolean
        whether or not to add colorbar to the plot.
        Default is False

    Returns
    ------------
    fig: figure
    '''

    if not cmap:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#FEFAFA", "black"])
        cmap.set_bad('white')
        cmap.set_under('white', alpha=0)
    if len(maps.shape)==1:
        maps = maps.reshape(1, len(maps))
    nmaps = len(maps)
    maps_min = min
    maps_max = max
    if maps_min==None:
        maps_min = maps.min()
    if maps_max==None:
        maps_max = maps.max()
    if nmaps >= 8:
        ncol = 8
    else:
        ncol = nmaps
    nrow= round(nmaps/ncol+0.5)
    fig = plt.figure(figsize=(2*ncol, 1*nrow))
    axes = fig.subplots(nrows=nrow, ncols=ncol)
    if nmaps == 1:
        axess = [axes]
    else:
        axess = axes.flat
    for j, ax in enumerate(axess):
        try:
            m = map_img(maps[j])
            m[np.where((m<maps_min) & (m!=-np.inf))] = maps_min
            m[np.where((m>maps_max) & (m!=np.inf))] = maps_max
        except:
            m = map_img(maps[0])*0+np.inf
        im = ax.imshow(m, vmin=maps_min, vmax=maps_max)
        ax.set_axis_off()
        im.set_cmap(cmap)
    fig.subplots_adjust(right=0.8)
    if cbar:
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
    return fig
