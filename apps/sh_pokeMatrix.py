# -*- coding: utf-8 -*-
from __future__ import print_function
# What is this?
# Form interaction matrices based on a SH model and a DM model, with the ability
# for each to have independent XY scales and (relative) rotation about Z

import abbot.gradientOperator
import abbot.dm
import abbot.fourierSH as fourierSH
import numpy 
import sys
import Zernike

def _plotFractionalBar(frac,char='#',length=70,
      printFraction=False,
      printPercentage=True,
      spinner=False
   ):
   print(
      "[ "+
      char*int(frac*length)+
      "-"*(length-int(frac*length))+
      " ]"+
      ("" if not printPercentage else " {0:3d}%\r".format(int(frac*100))
         )+
      ("" if not printFraction else " {0:6.3f}\r".format(frac)
         )+
      ("" if not spinner else spinner)
      , end="" )
   sys.stdout.flush()

##
## ---- variables begin --------------------------
##
numpy.random.seed(18071977)
N=8
subApS=2
fftScl=1
dmSize=[(N+1)*subApS]*2  # how big (in pixels) is the DM
dmRot=0#(0.5*N**-1.0)*(1/3.14159*180) # in degrees
dmOffset=(0.,0.) # actuator spacing units
dmSpacing=[(N+1)]*2 # how many actuators (configuration)
dmScaling=(1+N**-1.0,1+N**-1.0) # how the DM is magnified relative to the actuator scale
allActuators = 0 # poke all actuators regardless of whether illuminated?
obscurationD = 0 # fraction of pupil diameter
ifScl = 0.50 # size of influence function
##
## ---- variables end ----------------------------
##

print("VARIABLES::")
print("\tN={:d}".format(N))
print("\tsub-aperture # pixels={:d}".format(subApS))
print("\tspot magnification={:d}".format(fftScl))
print("\tDM size/pixels=({0[0]:d}/{0[1]:d})".format(dmSize))
print("\tDM rotation/degrees={:f}".format(dmRot))
print("\tDM configuration (MxM)=({0[0]:d}/{0[1]:d})".format(dmSpacing))
print()


ei=lambda x : numpy.cos(x)+1.0j*numpy.sin(x)
reshaper=lambda ip :\
      ip.reshape([N,subApS,N,subApS]).swapaxes(1,2)

def getGrads(apWf, oldWay=True, extraData=None):
   if oldWay: raise ImplementationError("DISABLED")
   fSH.makeImgs( apWf, aperture )
   gradsV = fSH.getSlopes()
   focalP = fSH.lastSHImage.swapaxes(1,2).reshape([subApS*N]*2).copy()

   return(gradsV,focalP)

def dmFitter(size,dmSfc,dm):
   '''Take the DM object, a surface representing the DM, and the size to fit
   the surface onto.
   Returns the fitted surface to the specified size.
   '''
   dmSfc = dmSfc.reshape(dm.npix)
   if dm.npix==(size,size):
      return dmSfc
   if dm.npix[0]>size:
      dmSfc=dmSfc[dm.npix[0]//2-size//2:dm.npix[0]//2-size//2+size,:]
   if dm.npix[1]>size:
      dmSfc=dmSfc[:,dm.npix[1]//2-size//2:dm.npix[1]//2-size//2+size]
   #
   thisdmSfc = numpy.zeros( [size]*2, numpy.float32 )
   thisdmSfc[size//2-dmSfc.shape[0]//2:size//2-dmSfc.shape[0]//2+dmSfc.shape[0],
       size//2-dmSfc.shape[1]//2:size//2-dmSfc.shape[1]//2+dmSfc.shape[1]]=(
         dmSfc )
   return thisdmSfc

# ---

assert not dmSize is None, "dmSize cannot be None in this code"

oldWay = False # always set this to be False
if oldWay: raise ImplementationError("Not supported")
extraData = []
size=N*subApS
aperture = Zernike.anyZernike(1,size,size//2)
if obscurationD>0:
   aperture-= Zernike.anyZernike(1,size,size//2*obscurationD)
apMask=( reshaper(aperture).sum(axis=-1).sum(axis=-1) 
            > (0.5*(subApS)**2) ).astype(numpy.bool)
apIdx=apMask.ravel().nonzero()[0]

# \/ configure and setup a DM object
dm = abbot.dm.dm(dmSize,dmSpacing,rotation=dmRot,within=0, ifScl=ifScl,
      lateralScl = dmScaling,
      lateralOffset = dmOffset ) #[ 0.5*-(subApS%2)*subApS**-1.0, 0] )

# \/ configure and setup a fourier Shack-Hartmann object
nPix = subApS*N
binning = 1 # don't bin
LT = 1 # lazy-truncation (fast)
GP = 0 # guard-pixels (none)
radialExtension = 0
magnification = 1
illumFraction = 0.5
fSH = fourierSH.FourierShackHartmann( N, aperture, illumFraction,
      magnification, binning, [0,], LT, GP, radialExtension
   )

# \/ Setup actuators to poke
if allActuators:
   dmActIdx=dm.usableIdx
else:
   # \/ assumption: DM-WFS are relatively well aligned 
   dmActIdx=(Zernike.anyZernike( 1,
         max(dmSpacing), 
         min(dmSpacing)/2,
         ratio=max(dmSpacing)*min(dmSpacing)**-1.0
      )!=0).ravel().nonzero()[0]

# \/ Generate interaction matrix
print("INTERACTION MATRIX GENERATION::") ; sys.stdout.flush()
pokeM=[]
for i,actNo in enumerate( dmActIdx ):
   thisApWf=dmFitter(size,dm.poke( actNo ),dm)
   pokeM.append(getGrads(thisApWf, oldWay, extraData)[0])
   _plotFractionalBar( (i+1)*len(dmActIdx)**-1.0 )

print()

# \/ Analysis
import pylab
pylab.figure(1)
pylab.imshow( pokeM, aspect='auto', cmap='gray' )

   # \/ location of DM actuator coordinates, relative to the centre of the
   #   array
dmActCds = (
      numpy.array(dm.actCds)-(numpy.array(dm.npix)/2.0).reshape([1,2])
   ).take(dmActIdx,axis=0)
dmActDistanceFromCentre = (dmActCds**2.0).sum(axis=1)**0.5
print("Minimum DM actuator distance from centre (relative)={:f}".format(
      min(dmActDistanceFromCentre*2.0)/size ))
print("Maximum DM actuator distance from centre (relative)={:f}".format(
      max(dmActDistanceFromCentre)*2.0/size ))
print("Acceptable tolerance={:f}".format(N**-1.0))

#   # \/ total poked surface
#pokeSurfaces = [ dmFitter(size,dm.poke(j),dm) for j in dmActIdx[::2] ]
#pokedSurface = numpy.sum( pokeSurfaces, axis=0 )
#pylab.figure(2)
#pylab.subplot(1,2,1)
#pylab.imshow( pokedSurface*aperture )
#pylab.title( "inside" )
#pylab.subplot(1,2,2)
#pylab.imshow( pokedSurface*(1-aperture) )
#pylab.title( "outside" )
#
#   # \/ variance of the slope signal per poked actuator
#pylab.figure(3)
#pylab.plot( numpy.var( pokeM, axis=1 ) )
#pylab.title( "Variance of slope signal" )
#pylab.xlabel( "DM actuators, order of those poked" )
