# igrins_contfit
Python code to remove the blaze function from IGRINS spectra

# Required modules 
Designed to be run on Python 3. 
Uses numpy, matplotlib, astropy, scipy 

# Useage 

    python cont_fit.py SDCK_20180115_0109.spec.fits

Will produce 2 additional files:

 SDCK_20180115_0109.flat.fits a fits file with 3 extensions:
  
  1) Original spectrum

  2) Wavelength

  3) Flattened spectrum

 SDCK_20180115_0109.flat.pdf - plot of the original spectrum and the flattened spectrum
