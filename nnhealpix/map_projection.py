# -*- encoding: utf-8 -*-

import numpy as np
import healpy as hp
from scipy.interpolate import griddata
from scipy.ndimage.interpolation import zoom
import numba

@numba.jit(nopython=True)
def binned_map(signal, pixidx, mappixels, hits):
    """Project a TOD onto a map.

    Parameters
    ----------
    signal: A TOD containing the signal to be projected (1D vector)
    pixidx: A TOD containing the index of the pixels (1D vector, same
            length as `signal`)
    mappixels: A Healpix map that will contain the projected signal
    hits: A Healpix map of the same resolution as mappixels that will
          contain the number of hits
    """

    assert len(mappixels) == len(hits)
    assert len(signal) == len(pixidx)
    
    mappixels[:] = 0.0
    hits[:] = 0
    
    for i in range(len(signal)):
        mappixels[pixidx[i]] += signal[i]
        hits[pixidx[i]] += 1
    
    for i in range(len(mappixels)):
        if hits[i] > 0:
            mappixels[i] /= hits[i]


def img2map(
        img,
        resultmap,
        resulthits,
        delta_theta,
        delta_phi,
        rot=np.eye(3),
):
    """Projection of a 2D image on a Healpix map.

    Parameters
    ----------
    img: A 2D matrix containing the image to be projected on the map
    resultmap: A Healpix map where to project the image
    resulthits: A Healpix map of the same side as `resultmap`, which
                will store the hit count per pixel of `resultmap`
    delta_theta: the width of the image along the meridian, in degrees
    delta_phi: the height of the image along the meridian, in degrees
    rot: Either a 3×3 matrix or a `healpy.rotator.Rotator` object

    Result
    ------
    A tuple containing the map and the hit map. Unseen pixels in the map
    are set to zero.
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
    nx, ny = [max(1, int(span / map_resolution))
              for span in (delta_phi, delta_theta)]
    theta_proj = np.linspace(
        (np.pi - delta_theta) / 2,
        (np.pi + delta_theta) / 2,
        nx,
    )
    phi_proj = np.linspace(
        delta_phi / 2,
        -delta_phi / 2,
        ny,
    )

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


def img2healpix2(img, nside, delta_theta, delta_phi, rot=np.eye(3)):
    """Projection of a 2D image on a Healpix map.

    This function is a wrapper to `img2map`. Use the latter function
    if you have already allocated a map.

    Parameters
    ----------
    img: A 2D matrix containing the image to be projected on the map
    nside: The resolution of the Healpix map
    delta_theta: the width of the image along the meridian, in degrees
    delta_phi: the height of the image along the meridian, in degrees
    rot: Either a 3×3 matrix or a `healpy.rotator.Rotator` object

    Result
    ------
    A tuple containing the map and the hit map. Unseen pixels in the map
    are set to zero.
    """

    assert hp.isnsideok(nside)
    assert delta_theta < 180.0
    assert delta_phi < 180.0

    result = np.zeros(hp.nside2npix(nside)) + hp.UNSEEN
    hits = np.zeros(result.size, dtype='int')
    img2map(img, result, hits, delta_theta, delta_phi, rot)
    
    return result, hits


def img2healpix(img, nside, thetac, phic, delta_theta, delta_phi, rot=None):
    """projection of a 2D image on healpix map

    Parameters
    ----------
    img: array-like
        image to project, must have shape = (#img, M, N)
    nside: integer
        Nside parameter for the output map.
        Must be a valid healpix Nside value
    thetac, phic: float
        coordinate (in degrees) where to project the center of the image on the healpix map.
        Must follow the healpix angle convention:
        0 <= thetac <= 180    with 0 being the N and 180 the S Pole
        0 <= phic <= 360      with 0 being at the center of the map, then it
                              increases moving towards W
    delta_theta, delta_phi: float
        dimension of the projected image
    rot: None
        Not yet implemented!

    Returns
    -------
    hp_map : array
        output healpix map with the image projection
    """

    imgf = np.flip(img, axis=2)
    imgf = np.array(imgf)
    print(img.shape[1], img.shape[2])
    data = imgf.reshape(img.shape[0], img.shape[1]*img.shape[2])
    xsize = img.shape[1]
    ysize = img.shape[2]
    theta_min = thetac-delta_theta/2.
    theta_max = thetac+delta_theta/2.
    phi_max = phic+delta_phi/2.
    phi_min = phic-delta_phi/2.
    theta_min = np.radians(theta_min)
    theta_max = np.radians(theta_max)
    phi_min = np.radians(phi_min)
    phi_max = np.radians(phi_max)
    img_theta_temp = np.linspace(theta_min,theta_max,ysize)
    img_phi_temp = np.linspace(phi_min,phi_max,xsize)
    ipix = np.arange(hp.nside2npix(nside))
    if rot == None:
        theta_r, phi_r = hp.pix2ang(nside,ipix)
    theta1 = theta_min
    theta2 = theta_max
    flg = np.where(theta_r<theta1,0,1)
    flg *= np.where(theta_r>theta2,0,1)
    if phi_min >= 0:
        phi1 = phi_min
        phi2 = phi_max
        flg  *= np.where(phi_r<phi1,0,1)
        flg *= np.where(phi_r>phi2,0,1)
    else:
        phi1 = 2.*np.pi+phi_min
        phi2 = phi_max
        flg *= np.where((phi2<phi_r) & (phi_r<phi1),0,1)
        img_phi_temp[img_phi_temp<0] = 2*np.pi+img_phi_temp[img_phi_temp<0]
    img_phi, img_theta = np.meshgrid(img_phi_temp,img_theta_temp)
    img_phi = img_phi.flatten()
    img_theta = img_theta.flatten()
    ipix = np.compress(flg,ipix)
    pl_theta  = np.compress(flg,theta_r)
    pl_phi  = np.compress(flg,phi_r)
    points = np.zeros((len(img_theta),2),'d')
    points[:,0] = img_theta
    points[:,1] = img_phi
    npix = hp.nside2npix(nside)
    hp_map = np.zeros((data.shape[0],npix),'d')
    for i in range(data.shape[0]):
        hp_map[i,ipix] = griddata(
            points, data[i,:], (pl_theta, pl_phi), method='nearest')
    return hp_map
