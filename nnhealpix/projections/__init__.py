# -*- encoding: utf-8 -*-

import numba
import numpy as np
import healpy as hp

from scipy.interpolate import griddata
from scipy.ndimage.interpolation import zoom


@numba.jit(nopython=True)
def binned_map(signal, pixidx, mappixels, hits, reset_map=True):
    """Project a TOD onto a map.

    This function implements a simple binner to project a TOD into a
    map.

    Args:

        * signal: A TOD containing the signal to be projected (1D vector)
        * pixidx: A TOD containing the index of the pixels (1D vector, same
          length as `signal`)
        * mappixels: A Healpix map that will contain the projected signal
        * hits: A Healpix map of the same resolution as mappixels that will
          contain the number of hits

    """

    assert len(mappixels) == len(hits)
    assert len(signal) == len(pixidx)

    if reset_map:
        mappixels[:] = 0.0
        hits[:] = 0

    for i in range(len(signal)):
        mappixels[pixidx[i]] += signal[i]
        hits[pixidx[i]] += 1

    for i in range(len(mappixels)):
        if hits[i] > 0:
            mappixels[i] /= hits[i]


def img2map(
    img, resultmap, resulthits, delta_theta, delta_phi, rot=np.eye(3), reset_map=True
):
    """Project a 2D image on a Healpix map.

    Args:
        * img: A 2D matrix containing the image to be projected on the map
        * resultmap: A Healpix map where to project the image
        * resulthits: A Healpix map of the same side as `resultmap`, which
          will store the hit count per pixel of `resultmap`
        * delta_theta: the width of the image along the meridian, in degrees
        * delta_phi: the height of the image along the meridian, in degrees
        * rot: Either a 3×3 matrix or a `healpy.rotator.Rotator` object
        * reset_map: If True, both `resultmap` and `resulthits` will be zeroed
          before doing the projection

    Returns:
        A tuple containing the map and the hit map. Unseen pixels in
        the map are set to zero.
    """

    assert img.ndim == 2
    assert len(resultmap) == len(resulthits)
    assert delta_theta > 0.0
    assert delta_phi > 0.0

    nside = hp.npix2nside(len(resultmap))

    delta_theta, delta_phi = [np.deg2rad(x) for x in (delta_theta, delta_phi)]

    if type(rot) is hp.rotator.Rotator:
        rotmatr = rot.mat
    else:
        rotmatr = rot

    # We fire a number of rays close enough not to miss pixels within
    # the image frame. We use as a rule of thumb a spacing that is
    # half the resolution of the map.
    map_resolution = 0.5 * hp.nside2resol(nside, arcmin=False)
    nx, ny = [max(1, int(span / map_resolution)) for span in (delta_theta, delta_phi)]
    theta_proj = np.linspace((np.pi - delta_theta) / 2, (np.pi + delta_theta) / 2, nx)
    phi_proj = np.linspace(delta_phi / 2, -delta_phi / 2, ny)

    # In order to fire so many rays, we need to interpolate between
    # adjacent pixels in the image matrix
    proj_img = zoom(img, (nx / img.shape[1], ny / img.shape[0]), order=0)

    # This 2D mesh grid contains the direction of all the rays we're
    # going to fire around position (θ=π/2, φ=0).
    theta_proj, phi_proj = np.meshgrid(theta_proj, phi_proj)

    # The shape of "dirs" is nx × ny × 3
    dirs = hp.ang2vec(theta_proj, phi_proj)

    # "rotdirs" has the same shape as "dirs". With this operation, we
    # apply the rotation matrix to the rays around position (θ=π/2,
    # φ=0).
    rotdirs = np.tensordot(dirs, rotmatr, (2, 1))

    # "theta" and "phi" are linear vectors
    theta, phi = hp.vec2ang(np.reshape(rotdirs, (-1, 3)))
    pixidx = hp.ang2pix(nside, theta, phi)

    # Run a simple map-maker
    binned_map(np.ravel(proj_img), pixidx, resultmap, resulthits)

    # We're returning nothing, as the result is in "resultmap" and
    # "resulthits"


def img2healpix(img, nside, delta_theta, delta_phi, rot=np.eye(3)):
    """Projection of a 2D image on a Healpix map.

    This function is a wrapper to :func:`nnhealpix.img2map`. Use the
    latter function if you have already allocated a map.

    Args:
        * img: A 2D matrix containing the image to be projected on the
        * map nside (int): The resolution of the Healpix map
        * delta_theta (float): the width of the image along the
        * meridian, in degrees delta_phi (float): the height of the
        * image along the meridian, in degrees rot: Either a 3×3
        * matrix or a `healpy.rotator.Rotator` object

    Returns:
        A tuple containing the map and the hit map. Unseen pixels in
        the map are set to zero.
    """

    assert hp.isnsideok(nside)
    assert delta_theta < 180.0
    assert delta_phi < 180.0

    result = np.zeros(hp.nside2npix(nside))
    hits = np.zeros(result.size, dtype="int")
    img2map(img, result, hits, delta_theta, delta_phi, rot, reset_map=False)

    return result, hits


class projectimages:
    """Project a randomly chosen set of 2D images on Healpix maps.

    This class returns an iterator that produces a set of Healpix maps
    given a number of 2D images. It can be used in :code:`for` loops
    to produce datasets for training convolutional neural networks.

    Args:

        * images (array): 3D tensor with shape ``[n, width, height]``,
          where ``n`` is the number of images and ``width×width`` is
          the size of each 2D image
        * nside (int): resolution of the Healpix maps
        * delta_theta (float, or 2-tuple of floats): Either the size
          along the theta axis of the image (before applying any
          rotation), or a range ``(min, max)``. In the latter case,
          each map will have delta_theta picked randomly within the
          range.
        * delta_phi (float, or 2-tuple of floats): Same as
          :param:`delta_phi`, but along the phi direction
        * num (int): If specified, the iterator will run "num"
          times. Otherwise, it will loop forever.
        * rot: Either a 3×3 matrix or a ``healpy.rotator.Rotator``
          object (optional)

    Returns:
        Each iteration returns a pair (num, pixels) containing the
        index of the image projected on the map and the pixels of the
        map itself.  The value of "num" is always in the range [0,
        images.shape[0] - 1).

    Example::

        import nnhealpix as nnh
        import numpy as np
        import healpy
        from keras.datasets import mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        for idx, pixels in nnh.projectimages(x_train, 64, 30.0, 30.0, num=5):
            print('Image index: {0}, digit is {1}'.format(idx, y_train[idx]))
            healpy.mollview(pixels)

    """

    def __init__(self, images, nside, delta_theta, delta_phi, rot=None, num=None):
        self.images = images
        self.nside = nside
        self.delta_theta = delta_theta
        self.delta_phi = delta_phi
        self.num = num
        self.idx = 0
        self.hitmap = np.zeros(hp.nside2npix(self.nside), dtype="int")
        self.rot = rot

    def __iter__(self):
        return self

    def _get_angle(self, value):
        if isinstance(value, (list, tuple, np.ndarray)):
            assert len(value) == 2
            start, stop = value
            result = np.random.rand() * (stop - start) + start
        else:
            result = float(value)

        return result

    def _get_delta_theta(self):
        return self._get_angle(self.delta_theta)

    def _get_delta_phi(self):
        return self._get_angle(self.delta_phi)

    def __next__(self):
        if self.num and self.idx >= self.num:
            raise StopIteration()

        delta_theta = self._get_delta_theta()
        delta_phi = self._get_delta_phi()

        imgidx = np.random.choice(self.images.shape[0])
        if self.rot:
            rotation = self.rot
        else:
            rotation = hp.rotator.Rotator(
                rot=(
                    np.random.rand() * 360.0,
                    np.random.rand() * 360.0,
                    np.random.rand() * 360.0,
                )
            )
        pixels = np.zeros(hp.nside2npix(self.nside), dtype="float")
        img2map(
            self.images[imgidx],
            pixels,
            self.hitmap,
            delta_theta,
            delta_phi,
            rotation,
            reset_map=False,  # We're not interested in "hits"
        )

        self.idx += 1
        return imgidx, pixels


def img2healpix_planar(img, nside, thetac, phic, delta_theta, delta_phi, rot=None):
    """Project a 2D image on healpix map

    Args:
        * img (array): image to project. It must have shape ``(#img,
          M, N)``
        * nside (int): ``NSIDE`` parameter for the output map.
        * thetac, phic (float): coordinates (in degrees) where to
          project the center of the image on the healpix map. They
          must follow the HEALPix angle convention:
            - ``0 <= thetac <= 180``, with 0 being the N and 180 the S Pole
            - ``0 <= phic <= 360``, with 0 being at the center of the
              map. It increases moving towards W
        * delta_theta, delta_phi (float): angular size of the projected image
        * rot: not implemented yet!

    Returns:
        The HEALPix map containing the projected image.
    """

    imgf = np.flip(img, axis=2)
    imgf = np.array(imgf)

    data = imgf.reshape(img.shape[0], img.shape[1] * img.shape[2])
    xsize = img.shape[1]
    ysize = img.shape[2]
    theta_min = thetac - delta_theta / 2.0
    theta_max = thetac + delta_theta / 2.0
    phi_max = phic + delta_phi / 2.0
    phi_min = phic - delta_phi / 2.0
    theta_min = np.radians(theta_min)
    theta_max = np.radians(theta_max)
    phi_min = np.radians(phi_min)
    phi_max = np.radians(phi_max)
    img_theta_temp = np.linspace(theta_min, theta_max, ysize)
    img_phi_temp = np.linspace(phi_min, phi_max, xsize)
    ipix = np.arange(hp.nside2npix(nside))
    if rot == None:
        theta_r, phi_r = hp.pix2ang(nside, ipix)
    theta1 = theta_min
    theta2 = theta_max
    flg = np.where(theta_r < theta1, 0, 1)
    flg *= np.where(theta_r > theta2, 0, 1)
    if phi_min >= 0:
        phi1 = phi_min
        phi2 = phi_max
        flg *= np.where(phi_r < phi1, 0, 1)
        flg *= np.where(phi_r > phi2, 0, 1)
    else:
        phi1 = 2.0 * np.pi + phi_min
        phi2 = phi_max
        flg *= np.where((phi2 < phi_r) & (phi_r < phi1), 0, 1)
        img_phi_temp[img_phi_temp < 0] = 2 * np.pi + img_phi_temp[img_phi_temp < 0]
    img_phi, img_theta = np.meshgrid(img_phi_temp, img_theta_temp)
    img_phi = img_phi.flatten()
    img_theta = img_theta.flatten()
    ipix = np.compress(flg, ipix)
    pl_theta = np.compress(flg, theta_r)
    pl_phi = np.compress(flg, phi_r)
    points = np.zeros((len(img_theta), 2), "d")
    points[:, 0] = img_theta
    points[:, 1] = img_phi
    npix = hp.nside2npix(nside)
    hp_map = np.zeros((data.shape[0], npix), "d")
    for i in range(data.shape[0]):
        hp_map[i, ipix] = griddata(
            points, data[i, :], (pl_theta, pl_phi), method="nearest"
        )
    return hp_map
