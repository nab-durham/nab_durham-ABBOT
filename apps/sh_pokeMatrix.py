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
N=15
subApS=2
fftScl=1
dmSize=[(N+1)*subApS]*2  # how big (in pixels) is the DM
dmRot=13.00
dmSpacing=[(N+1)]*2 # how many actuators (configuration)
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
#(redundant)   if oldWay: # generally, this is deprecated
#(redundant)      # \/ ----- begin setup -----------------------
#(redundant)      cds=numpy.arange(fftScl*subApS)-(fftScl*subApS-1)/2.0
#(redundant)      corrCds=numpy.arange(subApS)-(subApS-1)/2.0
#(redundant)      fftCorr=ei(-numpy.pi*(fftScl*subApS)**-1.0*numpy.add.outer(corrCds,corrCds)
#(redundant)            ).reshape([1,1]+[subApS]*2) # reuse cds
#(redundant)         # \/ a difference of 1 between the edges of the sub-aperture
#(redundant)         # = 1/subApS*(x) = 2*pi/(fftScl*subApS)*(fftScl)/2/pi*(x)
#(redundant)         # so movement is then fftScl/2/pi of a pixel
#(redundant)      expectedGradGain=4 #fftScl/2/numpy.pi
#(redundant)      # /\ ----- end -------------------------------
#(redundant)      
#(redundant)      # \/ ----- begin creating spots --------------
#(redundant)      focalP = numpy.fft.fftshift (abs(numpy.fft.fft2(
#(redundant)            fftCorr*reshaper(aperture*ei(apWf*2*numpy.pi/1.0))
#(redundant)               ,s=[subApS*fftScl]*2 ))**2.0) * (fftScl*subApS)**-2.0
#(redundant)      focalP=focalP.reshape([-1]+[subApS*fftScl]*2).take(apIdx,axis=0)
#(redundant)      # /\ ----- end -------------------------------
#(redundant)      
#(redundant)      # \/ ----- begin creating gradients ----------
#(redundant)      gradsV = numpy.array(
#(redundant)            [ (focalP*cds.reshape(tShape)).sum(axis=-1).sum(axis=-1)
#(redundant)               /(focalP.sum(axis=-1).sum(axis=-1))
#(redundant)              for tShape in ([1,-1],[-1,1]) 
#(redundant)            ]
#(redundant)         ).ravel()*expectedGradGain**-1.0
#(redundant)      # /\ ----- end -------------------------------
#(redundant)   else:
#(redundant)      ( cntr, refslopes, slopeScaling ) = extraData
   if not oldWay:
##      imgs = fourierSH.makeSHImgs( aperture, apWf, N, mag=fftScl,
##            lazyTruncate=1, binning=1,
##            guardPixels=1 if (aperture.shape[0]/N)==2 else 0,
##            radialExtension=0 )
##      gradsV = fourierSH.getSlopes( imgs, cntr, apIdx, refslopes, slopeScaling)
      fSH.makeImgs( apWf, aperture )
      gradsV = fSH.getSlopes()
      focalP = fSH.lastSHImage.swapaxes(1,2).reshape([subApS*N]*2).copy()
   else:
      raise ImplementationError("DISABLED")

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

oldWay = False
extraData = []
size=N*subApS
aperture = Zernike.anyZernike(1,size,size//2)
aperture-= Zernike.anyZernike(1,size,size//2*(1.2/4.2))
apMask=( reshaper(aperture).sum(axis=-1).sum(axis=-1) 
            > (0.5*(subApS)**2) ).astype(numpy.bool)
apIdx=apMask.ravel().nonzero()[0]

#(redundant)gO=abbot.gradientOperator.gradientOperatorType1(apMask)
#(redundant)gM=gO.returnOp()
#(redundant)reconM=numpy.dot(
#(redundant)   numpy.linalg.inv( numpy.dot( gM.T, gM )+1e-3*numpy.identity(gO.numberPhases) ), gM.T )

#ifScl = 0.5*(N*min(dmSpacing)**-1.0)**0.5
ifScl = 0.60
dm = abbot.dm.dm(dmSize,dmSpacing,rotation=dmRot,within=0, ifScl=ifScl,
      lateralOffset = [ 0.5*-(subApS%2)*subApS**-1.0, 0] )

if not oldWay:
##      cntr = fourierSH.makeCntrArr( subApS )
      nPix = subApS*N
      binning = 1
      LT = 1
      GP = 0
      radialExtension = 0
      #
##      ( slopeScaling, reconScalingFactor, tiltFac, refslopes )=\
##            fourierSH.calibrateSHmodel( aperture, cntr, nPix, N, fftScl, apIdx,
##                  [0,], binning, LT, GP, radialExtension 
##               )
      #
##      extraData = ( cntr, refslopes, slopeScaling )
      fSH = fourierSH.FourierShackHartmann( N, aperture, 0.5, 1, binning,
            [0,], LT, GP, radialExtension
         )
      extraData = fSH # this is a compatibility variable, later remove it

if dmRot==0 or 1==1:
   # \/ only works with the assumption that the DM is aligned with the WFS
   dmActIdx=(Zernike.anyZernike( 1,
         max(dmSpacing), 
         min(dmSpacing)/2,
         ratio=max(dmSpacing)*min(dmSpacing)**-1.0
      )!=0).ravel().nonzero()[0]
else:
   raise RuntimeError("Not implemented")

print("INTERACTION MATRIX GENERATION::") ; sys.stdout.flush()
pokeM=[]
for i,actNo in enumerate( dmActIdx ):
   thisApWf=dmFitter(size,dm.poke( actNo ),dm)
   pokeM.append(getGrads(thisApWf, oldWay, extraData)[0])
   _plotFractionalBar( (i+1)*len(dmActIdx)**-1.0 )

print()

## Below is the analysis
##
#
import pylab
pylab.figure(1)
pylab.imshow( pokeM, aspect='auto' )

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

   # \/ total poked surface
pokeSurfaces = [ dmFitter(size,dm.poke(j),dm) for j in dmActIdx[::2] ]
pokedSurface = numpy.sum( pokeSurfaces, axis=0 )
pylab.figure(2)
pylab.subplot(1,2,1)
pylab.imshow( pokedSurface*aperture )
pylab.title( "inside" )
pylab.subplot(1,2,2)
pylab.imshow( pokedSurface*(1-aperture) )
pylab.title( "outside" )

   # \/ variance of the slope signal per poked actuator
pylab.figure(3)
pylab.plot( numpy.var( pokeM, axis=1 ) )
pylab.title( "Variance of slope signal" )
pylab.xlabel( "DM actuators, order of those poked" )
