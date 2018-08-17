# igrins_contfit
Python code to remove the blaze function from IGRINS spectra

# Required modules 
Designed to be run on Python 3. 
Uses numpy, matplotlib, astropy, scipy, and pysptools (pip install pysptools)

# Useage 

python cont_fit.py SDCK_20180115_0109.spec.fits

Will produce 2 additional files:

  SDCK_20180115_0109.flat.fits a fits file with 4 extensions:
  
    1) Original spectrum
  
    2) Wavelength
    
    3) Blaze fit (useful if using the routine on A0s to then apply to target stars)
    
    4) Flattened spectrum
