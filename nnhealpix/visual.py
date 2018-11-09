import healpy as hp
import numpy as np
import keras
from keras.models import load_model
from nnhealpix.map_ordering import *
import matplotlib.pyplot as plt

def filter_img(weights, order=1):
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
    filter_img = hp.gnomview(m, reso=5.2, rot=[0, 2],
        notext=True, return_projected_map=True)
    plt.close()
    return filter_img

def plot_filters(filters, cmap=None):
    if not cmap:
        cmap = plt.cm.viridis
    if len(filters.shape)==1:
        filters = filters.reshape(1, len(filters))
    nfilt = len(filters)
    filt_min = filters.min()
    filt_max = filters.max()
    if nfilt >= 8:
        ncol = 8
    else:
        ncol = nfilt
    nrow= round(nfilt/ncol)
    fig = plt.figure(figsize=(8, 4))
    axes = fig.subplots(nrows=nrow, ncols=ncol)
    if nfilt == 1:
        axess = [axes]
    else:
        axess = axes.flat
    for j, ax in enumerate(axess):
        filt = filter_img(filters[j])
        im = ax.imshow(filt, vmin=filt_min, vmax=filt_max)
        ax.set_axis_off()
        im.set_cmap(cmap)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    return fig
