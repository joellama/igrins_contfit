import numpy as np
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits
from scipy.spatial import ConvexHull
from scipy.optimize import curve_fit
from scipy.interpolate import splev
from scipy.interpolate import splrep
from astropy.stats import sigma_clip
from astropy.convolution import convolve, Box1DKernel
import warnings
import sys
warnings.filterwarnings('ignore', category=UserWarning, append=True)


def convex_hull_removal(w, f):
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
    points = list(zip(w, f))
    # close the polygone
    poly = [(points[0][0],0)]+points+[(points[-1][0],0)]
    hull = ConvexHull(points)
    # the last two points are on the x axis, remove it
    hull.vertices.sort()
    x_hull = np.asarray(w[hull.vertices[1:-2]])
    y_hull = np.asarray(f[hull.vertices[1:-2]])
    x_hull, ind = np.unique(x_hull, return_index=True)
    y_hull = y_hull[ind]
    # tck = splrep(x_hull, y_hull,  s=1, k=1)
    tck = splrep(w[hull.vertices], f[hull.vertices], k=1)
    return tck

if __name__ == "__main__":
    fh = sys.argv[1]
    print("Fitting %s" % fh)
    spec = fits.open(fh)
    spec_f = spec[0].data
    spec_f[np.isnan(spec_f)] = 0
    spec_w = spec[1].data
    nord = spec_w.shape[0]
    band = spec[0].header['BAND']
    if band == '': # Some McDonald data doesn't record the band in the header
        band = fh.split('SDC')[1][0]
    obs = spec[0].header['TELESCOP']
    if band == 'H':
        orders = np.arange(98, 98 + nord, dtype=np.long)
    else:
        orders = np.arange(72, 72 + nord, dtype=np.long)
    blaze = fits.open('SDC%s_DCT_FLAT.fits' % band)[0].data
    blaze[np.isnan(blaze)] = 0
    flat = np.zeros_like(spec_f)
    pdf_fh = fh.replace('.fits', '_flat.pdf')
    m = [all(z) for z in zip(np.arange(0, 2048) > 150, np.arange(0, 2048) < 1900)]
    with PdfPages(pdf_fh) as pdf:
        for jj in np.arange(0, nord, dtype=np.long):
            hull_tck = convex_hull_removal(spec_w[jj, m], spec_f[jj, m])
            hull_fit = splev(spec_w[jj, m], hull_tck)
            # mid = np.long(len(spec_f[jj, m]) / 2.)
            ratio = np.percentile(hull_fit, 98)
            try:
                # blaze[jj, :] = convolve(blaze[jj, :], Box1DKernel(1))
                # Blaze peaks just right of the center
                ratio /= np.max(blaze[jj, :])
                fit = ratio*blaze[jj, :]
            except:
                ratio /= np.max(blaze[jj-1, :])
                fit = ratio*blaze[jj-1, :]
            flat[jj, :] = spec_f[jj, :] - fit
            fig, ax = plt.subplots(2, 1, sharex=True, figsize=(11.5, 8))
            ax[0].plot(spec_w[jj, :], spec_f[jj, :], label='data')
            ax[0].plot(spec_w[jj, :],  fit, label='Fit')
            ax[0].legend(loc='upper left')
            ax[0].set_ylabel("IGRINS Flux")
            ax[0].set_title("%s - order %02d" % (fh, orders[jj]))
            ax[1].plot(spec_w[jj, :], flat[jj, :], label='Residuals')
            ax[1].axhline(1, lw=2, ls='--', c='k')
            ax[1].set_xlabel("Wavelength (microns)")
            ax[1].set_ylabel("Residual Flux")
            fig.tight_layout()
            pdf.savefig()
            plt.close()
spec.append(fits.PrimaryHDU(flat))
spec.writeto(fh.replace('spec', 'flat'), overwrite=True)
print('Wrote %s to disk' % fh.replace('spec', 'flat'))
