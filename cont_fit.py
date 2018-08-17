from __future__ import print_function
import numpy as np
import matplotlib.pylab as plt
from astropy.io import fits
from scipy.spatial import ConvexHull
import sys
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import splev
from scipy.interpolate import splrep
from astropy.stats import sigma_clip
import warnings
warnings.filterwarnings('ignore', category=UserWarning, append=True)


def convex_hull_removal(pixel, wvl):
    """
    Remove the convex-hull of the signal by hull quotient.
    Written by Joe Llama (Lowell Observatory) and Tomas Cabrera (MIT)
    Parameters:
        pixel: `list`
            1D HSI data (p), a pixel.
        wvl: `list`
            Wavelength of each band (p x 1).

    Results: `tck`
        scipy.interpolate spline function of the blaze fits

    """
    from pysptools.spectro.hull_removal import _jarvis
    points = list(zip(wvl, pixel))
    # close the polygone
    poly = [(points[0][0],0)]+points+[(points[-1][0],0)]
    hull = _jarvis.convex_hull(poly)
    # the last two points are on the x axis, remove it
    hull = hull[1:-2]
    x_hull = np.asarray([u for u,v in hull])
    y_hull = np.asarray([v for u,v in hull])
    ii = np.argsort(x_hull)
    x_hull, ind = np.unique(x_hull[ii], return_index=True)
    y_hull = y_hull[ii][ind]
    tck = splrep(x_hull, y_hull, k=1, s=1)
    return tck

if __name__ == "__main__":
    fh = sys.argv[1]
    print("Fitting %s" % fh)
    plot = True
    # plot = True
    # fh = 'SDCK_20180115_0109.spec.fits'
    a0v = fits.open(fh)
    a0f = a0v[0].data
    a0w = a0v[1].data
    nord = a0f.shape[0]
    hull = np.zeros_like(a0w)
    flat = np.zeros_like(a0w)
    if plot == True:
        print('plotting')
        fig, ax = plt.subplots(2, 1, sharex=True)
        for jj in np.arange(0, nord, dtype=np.long):
            sc = sigma_clip(a0f[jj, :], sigma=3, iters=5)
            hull_tck = convex_hull_removal(sc[~sc.mask], a0w[jj, ~sc.mask])
            hull[jj, :] = splev(a0w[jj, :], hull_tck)
            ax[0].plot(a0w[jj, :], a0f[jj, :], c='blue')
            ax[0].plot(a0w[jj, :], hull[jj, :], c='orange')
            flat[jj, :] = sigma_clip(a0f[jj, :] - hull[jj, :],
                sigma=3, iters=5)
            ax[1].plot(a0w[jj, :], flat[jj, :], c='red')
        fig.savefig(fh.replace('spec.fits', 'flat.png'), dpi=300)
        print('Wrote %s to disk' % fh.replace('spec.fits', 'flat.png'))
    a0v.append(fits.PrimaryHDU(hull))
    a0v.append(fits.PrimaryHDU(flat))
    a0v.writeto(fh.replace('spec', 'flat'), overwrite=True)
    print('Wrote %s to disk' % fh.replace('spec', 'flat'))



