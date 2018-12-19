import healpy as hp
import numpy as np
import keras
from keras.models import load_model
from nnhealpix.map_ordering import *
import matplotlib.pyplot as plt
import matplotlib
from keras import backend as K


def val2str(val):
    'Convert VAL (number) into a string'
    s = str(val)
    if s[0] == '-':
        # Use the Unicode minus sign
        s = '−' + s[1:]

    return s


def draw_filter(weights, cmap, sub=None, order=1,
                minimum=None, maximum=None,
                show_values=False, val2str=val2str):
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
    hp.gnomview(m, reso=1.8, rot=[0, 2], sub=sub, cmap=cmap,
                min=minimum, max=maximum, notext=True, xsize=600)

    if show_values:
        axis = plt.gcf().gca()
        for i, curpix in enumerate(pix_num):
            theta, phi = hp.pix2ang(nside, curpix)
            axis.projtext(theta, phi, val2str(weights[i]),
                          horizontalalignment='center',
                          verticalalignment='center',
                          color='black',
                          bbox=dict(facecolor='white', linewidth=0, alpha=0.7))


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


def plot_filters(filters, cmap=None, cbar=False, minimum=None, maximum=None,
                 show_values=False, val2str=val2str):
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
    minimum, maximum: float
        min and max value for the color map, if None they are the min and max
        values of the set of filters
    show_values: Boolean (default: False)
        if True, display the value of the filter on each pixel
    val2str: function (default: str)
        function to convert the value of each pixel in the filter into a
        string, used when `show_values` is True.
    Returns
    ------------
    fig: figure
    '''

    if not cmap:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "", ["#FEFAFA", "black"])
        cmap.set_bad('white', alpha=0)
        cmap.set_under('white', alpha=0)
    if len(filters.shape) == 1:
        filters = filters.reshape(1, len(filters))
    nfilt = len(filters)
    filt_min, filt_max = minimum, maximum
    if filt_min is None:
        filt_min = np.min(filters)
    if filt_max is None:
        filt_max = np.max(filters)
    ncol = min(8, nfilt)
    nrow = round(nfilt / ncol + 0.4)
    fig = plt.figure(figsize=(8, 4))
    for j in range(nrow * ncol):
        draw_filter(filters[j], sub=(nrow, ncol, j + 1), cmap=cmap,
                    minimum=filt_min, maximum=filt_max,
                    show_values=show_values, val2str=val2str)
        fig.gca().set_title('Filter #{0}'.format(j))
        fig.gca().set_axis_off()

    fig.subplots_adjust(right=0.8)
    if cbar:
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
    return fig


def plot_layer_output(maps, cmap=None, cbar=False, min=None, max=None):
    '''plot a the effect of filters on maps in a given layer of the network.

    Parameters
    ----------
    maps: array
        output maps of the layer to plot
    cmap: color map
        if None a pick/black color map will be used in the plot
    cbar: boolean
        whether or not to add colorbar to the plot.
        Default is False
    min, max: float
        min and max value for the color map, if None they are the min and max
        values of the set of maps
    count: boolean
        whether to return or not the number of active nodes in the layer,
        default is True

    Returns
    ------------
    fig: figure
    '''

    if not cmap:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "", ["#FEFAFA", "black"])
        cmap.set_bad('white', alpha=0)
        cmap.set_under('white', alpha=0)
    if len(maps.shape) == 1:
        maps = maps.reshape(1, len(maps))
    nmaps = len(maps)
    totactive = nmaps
    maps_min = min
    maps_max = max
    if maps_min == None:
        maps_min = maps.min()
    if maps_max == None:
        maps_max = maps.max()
    if nmaps >= 8:
        ncol = 8
    else:
        ncol = nmaps
    nrow = round(nmaps/ncol+0.4)
    fig = plt.figure(figsize=(2*ncol, 1*nrow))
    axes = fig.subplots(nrows=nrow, ncols=ncol)
    if nmaps == 1:
        axess = [axes]
    else:
        axess = axes.flat
    for j, ax in enumerate(axess):
        try:
            m = map_img(maps[j])
            m[m == -np.inf] = np.inf
            m[np.where(m < maps_min)] = maps_min
            m[np.where((m > maps_max) & (m != np.inf))] = maps_max
        except:
            m = map_img(maps[0])*0+np.inf
        im = ax.imshow(m, vmin=maps_min, vmax=maps_max)
        ax.set_axis_off()
        im.set_cmap(cmap)
        if j < nmaps:
            if np.all(maps[j] == 0):
                totactive -= 1
                line = np.arange(600)+100
                ax.plot(line, line/2, color='black', lw=0.5, alpha=0.5)
    fig.subplots_adjust(right=0.8)
    if cbar:
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
    print('Active nodes: ', totactive)
    return fig


def plot_layer_nodes(model, layer, X_val, binary=False, cmap=None, plot=True):
    '''return a map of the active nodes in a give layer and plot it

    Parameters
    ----------
    model: keras model object
        Neural network model to analyze.
    layer: int
        number defining the layer in model to analyze.
    X_val: array-like
        set of inputs used for network validation.
    binary: boolean
        if True the value of each node will be set to one if the node is active
        if False they are set equal to the rms of the map.
        Default is False.
    cmap: color map
        if None a pick/black color map will be used in the plot.
    plot: boolean
        if True a plot with the map of the active nodes in the layer will be
        shown. Default is True.

    Returns
    ------------
    nodes: array-like
        matrix with the map of active nodes in the layer.
    fig: figure
    '''

    if not cmap:
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",
                                                                   ["white", "black"])
        # cmap = matplotlib.colorbar.cm.magma_r
        cmap.set_bad('white', alpha=0)
        cmap.set_under('white', alpha=0)
    get_layer_output = K.function([model.layers[0].input],
                                  [model.layers[layer].output])
    layer_output = get_layer_output([X_val])[0]
    nfilt = np.shape(layer_output)[2]
    nval = np.shape(layer_output)[0]
    nodes = np.ones((nval, nfilt))
    for i in range(nval):
        for j in range(nfilt):
            nodes[i, j] = np.std(layer_output[i, :, j])
    if binary:
        nodes[nodes > 0] = 1
    if nval > 20 & nfilt > 30:
        fig = plt.figure(figsize=(nfilt//30, nval//20))
    elif nval < 20 & nfilt > 30:
        fig = plt.figure(figsize=(nfilt//30, 1))
    else:
        fig = plt.figure(figsize=(1, 1))
    plt.imshow(nodes, cmap=cmap, vmin=0, aspect='auto')
    plt.xticks(range(nfilt), ['#' + str(i) for i in range(nfilt)])
    plt.yticks(range(nval), [str(i) for i in range(nval)])
    if not plot:
        plt.close()
    return nodes, fig
