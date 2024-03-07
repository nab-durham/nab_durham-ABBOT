# -*- coding: utf-8 -*-
"""ABBOT : functions to aid other calculations
"""

from __future__ import print_function
import numpy
import sys

def cds(N, roll=False):
   tcds = (numpy.arange(0,N)-(N/2.-0.5))*(N/2.0)**-1.0
   return tcds if not roll else numpy.fft.fftshift(tcds)

def circle(N,fractionalRadius=1):
   '''for N pixels, return a 2D array which has a circle (1 within, 0
   without) and radius a fraction of N.
   '''
   return numpy.add.outer(
         cds(N)**2,cds(N)**2 )<(fractionalRadius**2)

def makeTiltPhase(nPix, fac):
   return -fac*numpy.pi/nPix*numpy.add.outer(
         numpy.arange(nPix),
         numpy.arange(nPix)
      )

def rebin(ip,N):
   '''take the square 2D input and the number of sub-apertures and 
   return a 2D output which is binned over the sub-aperture elements
   and centred.
   '''
   nPix=ip.shape[0]
   sapxls=int( numpy.ceil(nPix*float(N)**-1.0) )
   N_=nPix//sapxls # the guessed number of original sub-apertures
   if N_==N:
      nx=ip
   else:
      newNPix=int( N*sapxls )
      dnp=newNPix-nPix
      nx=numpy.zeros([newNPix]*2,ip.dtype)
      nx[ dnp//2:dnp//2+nPix, dnp//2:dnp//2+nPix ]=ip 
   return nx.reshape([N,sapxls]*2).swapaxes(1,2).sum(axis=-1).sum(axis=-1)
