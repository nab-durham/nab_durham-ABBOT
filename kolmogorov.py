"""ABBOT: Poor man's stub for making a phase screen
*NOT* kolmogorov like and ignores r0 value
"""

from __future__ import print_function
import sys

def TwoScreens(nfft, r0, offset=0.,complex=0, flattening=0):
    """Interface equivalent to DFB's kolmogorov code
    """
    print("=-"*30)
    print("WARNING: this code generates an incorrect array, use with extreme care")
    print("=-"*30)
    import numpy
    from numpy import power, meshgrid, arange, where
    from numpy.fft import fftshift,fft2
    # should filter this but for now, not doing that
    core=numpy.random.normal(size=[2]+[nfft]*2)
    cds=power(power(meshgrid(
            fftshift(arange(nfft)-(nfft+1)//2),
            fftshift(arange(nfft)-(nfft+1)//2)
        ),2).sum(0),0.5)
    fftfilter=where(cds==0,0,power(cds,-11/6))
    scaling=nfft**(-0.0)*(nfft/r0)**0.5
    op=fft2( scaling*fftfilter*(core[0]+1.0j*core[1]) )
    return(op.real, op.imag)
