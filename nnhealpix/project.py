import numpy as np
import healpy as hp
from scipy.interpolate import griddata

def img2healpix(img, nside, thetac, phic, delta_theta, delta_phi, rot=None):
    imgf = np.flip(img, axis=1)
    imgf = np.array(imgf)
    data = imgf.reshape(1, img.shape[0]*img.shape[1])
    xsize = img.shape[0]
    ysize = img.shape[1]
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
    h_data = np.zeros((data.shape[0],npix),'d')
    for i in range(data.shape[0]):
        h_data[i,ipix] = griddata(
            points, data[i,:], (pl_theta, pl_phi), method='nearest')
    return h_data
