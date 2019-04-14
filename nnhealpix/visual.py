# -*- encoding: utf-8 -*-

import healpy as hp
import healpy.projaxes as pa
import numpy as np
import keras
from keras.models import load_model
import nnhealpix as nnh
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from keras import backend as K

CMAP_GRAY_TO_BLACK = LinearSegmentedColormap.from_list("", ["#FEFAFA", "black"])
CMAP_WHITE_TO_BLACK = LinearSegmentedColormap.from_list("", ["#FEFAFA", "black"])


def val2str(val):
    """Convert VAL (number) into a string

    This function works like "str", but it substitutes any trailing
    hyphen sign with a Unicode minus sign, which is typographically
    more correct.
    """
    s = str(val)
    if s[0] == "-":
        # Use the Unicode minus sign
        s = "−" + s[1:]

    return s


def draw_filter(
    fig,
    weights,
    extent,
    cmap,
    sub=None,
    order=1,
    vmin=None,
    vmax=None,
    xsize=600,
    ysize=600,
    show_values=False,
    val2str=val2str,
):
    """Return a 2D image of a filter.

    Args:
        * weights (array): Array of filter weights
        * order (int): Order of the filter (1st, 2nd neighbours). So
          far only ``order=1`` works.

    Returns:
        An array containing the 2-D grayscale image of the input filter.
    """

    order_fn = {1: nnh.pixel_1st_neighbours, 2: nnh.pixel_2nd_neighbours}
    assert (
        order in order_fn.keys()
    ), "invalid order ({0}) passed to filter, valid values are {1}".format(
        order, ", ".join([str(x) for x in order_fn.keys()])
    )
    fn = order_fn[order]
    nside = 16
    ipix = hp.ang2pix(nside, np.pi / 2, 0)
    pix_num = fn(ipix, nside)
    m = np.zeros(hp.nside2npix(nside)) + np.inf
    m[pix_num] = weights

    ax = pa.HpxGnomonicAxes(fig, extent, rot=[0, 2])
    fig.add_axes(ax)

    ax.projmap(m, reso=1.8, vmin=vmin, vmax=vmax, xsize=xsize, ysize=ysize, cmap=cmap)

    if show_values:
        for i, curpix in enumerate(pix_num):
            theta, phi = hp.pix2ang(nside, curpix)
            ax.projtext(
                theta,
                phi,
                val2str(weights[i]),
                horizontalalignment="center",
                verticalalignment="center",
                color="black",
                bbox=dict(facecolor="white", linewidth=0, alpha=0.7),
            )

    return ax


def filter_plot_layout(num_of_filters):
    """Calculate a nice layout for a set of filters to be plotted with `plot_filters`.

    This function returns a tuple containing the number of plots per
    row in each of the rows that are produced via a call to
    `plot_filters`.

    >>> filter_plot_layout(7)
    (4, 3)

    The result indicates that the best way to plot 7 filters in the same plot
    is to group them in two rows: 4 in the first row, and 3 in the second one.
    """

    DEFAULT_LAYOUTS = {
        1: (1,),
        2: (2,),
        3: (2, 1),
        4: (2, 2),
        5: (3, 2),
        6: (3, 3),
        7: (4, 3),
        8: (3, 3, 2),
        9: (3, 3, 3),
        10: (4, 4, 2),
        11: (3, 3, 3, 2),
        # Starting from here, the algorithm below would produce the same
        # result. We include them here explicitly for the sake of clarity
        12: (4, 4, 4),
        13: (4, 3, 3, 3),
        14: (4, 4, 3, 3),
        15: (4, 4, 4, 3),
        16: (4, 4, 4, 4),
    }

    # If the number of filters is low, use the default layouts listed above
    if num_of_filters in DEFAULT_LAYOUTS:
        return DEFAULT_LAYOUTS[num_of_filters]

    # In any other case, use an algorithm to calculate how many plots
    # per row should be created. The algorithm tries to follow these
    # rules:
    #
    #   1. Never place more than 4 plots per row
    #   2. Never use less than 3 plots per row
    #   3. Group 4-plot rows and 3-plot rows together (i.e., do not
    #      alternate between 4-plot and 3-plot rows), so that all
    #      the 4-plot rows come before the 3-plot rows.

    nrows = num_of_filters // 4
    if num_of_filters % 4 > 0:
        nrows += 1

    result = [4] * nrows
    rowidx = nrows - 1
    while sum(result) > num_of_filters:
        result[rowidx] -= 1
        rowidx -= 1

    return result


def filter_plot_axis_extents(layout, cbar_space=False):
    nrows, ncols = len(layout), max(layout)

    width = 1.0
    if cbar_space:
        width -= 0.15

    height = 1.0

    plotwidth = width / ncols
    plotheight = height / nrows

    extents = []
    top = height
    for cols_in_row in layout:
        left = (ncols - cols_in_row) * plotwidth / 2

        for curcol in range(cols_in_row):
            # The 0.8 factor is used to make room for the title
            extents.append((left, top - plotheight, plotwidth, plotheight * 0.8))
            left += plotwidth

        top -= plotheight

    return extents


def filter_plot_size(layout, basesize):
    """Return the size (in inches) of the plot produced by `plot_filters`

    Args:

        * layout (list of tuples): The result of a call to
          :func:`nnhealpix.visual.filter_plot_layout`

        * basesize (float): Size (in inches) to be used to plot each
          of the filters

    Returns:
        A 2-element tuple containing the width and height in inches of
        the plot.
    """

    nrows, ncols = len(layout), max(layout)

    # Each square containing a filter will be placed in a square
    # whose side is "basesize" inches long

    width = min(ncols * basesize, 12)
    height = nrows * basesize

    return (width, height)


def plot_filters(
    filters,
    cmap=None,
    cbar=False,
    vmin=None,
    vmax=None,
    show_titles=False,
    titlefn=None,
    show_values=False,
    val2str=val2str,
    basesize=3,
):
    """Plot a set of filters.

    Args:
        * filters (array): Array of filters to plot
        * cmap (color map): If ``None``, a pick/black color map will
          be used in the plot
        * cbar (boolean): Whether or not to add colorbar to the
          plot. Default is False
        * vmin, vmax (float): Minimum and maximum value for the color
          map. If ``None``, they are computed from the actual values
          in the set of filters.
        * show_titles (Boolean): If True, write a title above each
          filter. Default is ``False``.
        * titlefn (function): A function returning a string for the
          title of each filter. It should take one integer parameter,
          which is the progressive number of the filter starting from
          zero. The default is ``None`` (no title).
        * show_values (Boolean). If True, display the value of the
          filter on each pixel (default is ``False``).
        * val2str (function): Function to convert the value of each
          pixel in the filter into a string, used when `show_values`
          is True (default is ``str``).
        * basesize (float): Size (in inches) of one of the Gnomonic
          views to be displayed. The default is 4 inches.

    Returns:
        The figure containing the filter plots.
    """

    if not cmap:
        cmap = CMAP_GRAY_TO_BLACK
        cmap.set_bad("white", alpha=0)
        cmap.set_under("white", alpha=0)

    nfilt = len(filters)
    filt_min, filt_max = vmin, vmax
    if filt_min is None:
        filt_min = np.min(filters)
    if filt_max is None:
        filt_max = np.max(filters)

    layout = filter_plot_layout(nfilt)
    fig = plt.figure(figsize=filter_plot_size(layout, basesize))

    extents = filter_plot_axis_extents(layout, cbar_space=cbar is not None)

    for j in range(len(extents)):
        ax = draw_filter(
            fig,
            filters[j],
            extents[j],
            cmap=cmap,
            vmin=filt_min,
            vmax=filt_max,
            show_values=show_values,
            val2str=val2str,
        )

        if show_titles:
            if titlefn:
                title = titlefn(j)
            else:
                title = "Filter #{0}".format(j)

            ax.set_title(title)

        ax.set_axis_off()

    if cbar:
        cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        fig.colorbar(sm, cax=cbar_ax)

    return fig


def map_img(map_in, order=1):
    """Return a 2D image of a filter.

    Args:
        * map (array): The map to plot

    Returns:
        A 2-D image of the input filter.
    """

    img = hp.mollview(map_in, notext=True, return_projected_map=True)
    plt.close()
    img = np.flip(img, axis=0)
    return img


def plot_layer_output(maps, cmap=None, cbar=False, vmin=None, vmax=None, verbose=True):
    """plot a the effect of filters on maps in a given layer of the network.

    Args:
        * maps (array): Output maps of the layer to plot
        * cmap (color map): If ``None``, a pick/black color map will
          be used in the plot
        * cbar (Boolean): Whether or not to add a colorbar to the
          plot. Default is ``False``.
        * vmin, vmax (float): The minimum and maximum value for the
          color map. If ``None``, they are computed from the actual
          values of the pixels in `maps`.
        * count (Boolean): Whether to return or not the number of
          active nodes in the layer.  Default is ``True``.
        * verbose (Boolean): If ``True``, print a summary of the
          active nodes.

    Returns:
        A figure containing the plot.
    """

    if not cmap:
        cmap = CMAP_GRAY_TO_BLACK
        cmap.set_bad("white", alpha=0)
        cmap.set_under("white", alpha=0)
    if len(maps.shape) == 1:
        maps = maps.reshape(1, len(maps))
    nmaps = len(maps)
    totactive = nmaps
    maps_min, maps_max = vmin, vmax
    if maps_min == None:
        maps_min = np.min(maps)
    if maps_max == None:
        maps_max = np.max(maps)
    ncol = min(8, nmaps)
    nrow = round(nmaps / ncol + 0.4)
    fig = plt.figure(figsize=(2 * ncol, 1 * nrow))
    axes = fig.subplots(nrows=nrow, ncols=ncol)
    if nmaps == 1:
        axess = [axes]
    else:
        axess = axes.flat
    for j, ax in enumerate(axess):
        try:
            m = map_img(maps[j])
            m[np.isinf(m)] = np.inf
            m[np.where(m < maps_min)] = maps_min
            m[np.where((m > maps_max) & np.isfinite(m))] = maps_max
        except:
            m = map_img(maps[0]) * 0 + np.inf
        im = ax.imshow(m, vmin=maps_min, vmax=maps_max)
        ax.set_axis_off()
        im.set_cmap(cmap)
        if j < nmaps:
            if np.all(maps[j] == 0):
                totactive -= 1
                line = np.arange(600) + 100
                ax.plot(line, line / 2, color="black", lw=0.5, alpha=0.5)
    fig.subplots_adjust(right=0.8)
    if cbar:
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

    if verbose:
        print("Active nodes: ", totactive)

    return fig


def plot_layer_nodes(
    model,
    layer,
    X_val,
    binary=False,
    cmap=None,
    show_titles=False,
    titlefn=None,
    figsize=None,
    plot=True,
):
    """Create a map of the active nodes in a given layer

    Args:
        * model (Keras model object): Neural network model to analyze.
        * layer (int): number of the layer to analyze.
        * X_val (array-like): Set of inputs used for network
          validation.
        * binary (Boolean): If ``True``, the value of each node will
          be set to one if the node is active.  If ``False``, they are
          set equal to the RMS of the map. The default is ``False``.
        * cmap (color map): If ``None``, a gray/black color map will
          be used in the plot.
        * show_titles (Boolean): If ``True``, write a title above each
          filter. The default is ``False``.
        * titlefn (function): A function returning a string for the
          title of each filter. It should take one integer parameter,
          which is the progressive number of the filter starting from
          zero. If not specified, a custom title containin the
          progressive number of the plot will be used.
        * figsize (2-element tuple): Size of the figure, in inches. If
          not provided, a sensible size will be figured out.
        * plot (Boolean): If ``True``, a plot with the map of the
          active nodes in the layer will be shown. Default is True.

    Returns:
        A 2-element tuple containing the following elements:

        1. A matrix containing the map of active nodes in the layer;
        2. A figure containing the plot.
    """

    if not cmap:
        cmap = CMAP_WHITE_TO_BLACK
        # cmap = matplotlib.colorbar.cm.magma_r
        cmap.set_bad("white", alpha=0)
        cmap.set_under("white", alpha=0)
    get_layer_output = K.function([model.layers[0].input], [model.layers[layer].output])
    layer_output = get_layer_output([X_val])[0]
    nfilt = np.shape(layer_output)[2]
    nval = np.shape(layer_output)[0]
    nodes = np.ones((nval, nfilt))
    for i in range(nval):
        for j in range(nfilt):
            nodes[i, j] = np.std(layer_output[i, :, j])
    if binary:
        nodes[nodes > 0] = 1

    if figsize:
        intfigsize = figsize
    else:
        if nval > 20 and nfilt > 30:
            intfigsize = (nfilt // 30, nval // 20)
        elif nval < 20 and nfilt > 30:
            intfigsize = (nfilt // 30, 1)
        else:
            intfigsize = (1, 1)

    fig = plt.figure(figsize=intfigsize)
    plt.imshow(nodes, cmap=cmap, vmin=0, aspect="auto")

    if show_titles:
        if titlefn:
            titles = [titlefn(i) for i in range(nfilt)]
        else:
            titles = ["#{0}".format(i) for i in range(nfilt)]

        plt.xticks(range(nfilt), titles)

    plt.yticks(range(nval), [str(i) for i in range(nval)])

    if not plot:
        plt.close()
    return nodes, fig
