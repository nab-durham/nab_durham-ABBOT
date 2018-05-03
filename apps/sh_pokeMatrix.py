# -*- coding: utf-8 -*-
from __future__ import print_function
# What is this?
# Available are a SH model and a DM model, with the ability for each to have
# independent XY scales and (relative) rotation about Z, the DM about an
# arbitrary point.
# Then assume the SH is fixed and allow the DM actuator positions to vary from
# a nominal Fried geometry. Form a synthetic interaction matrix, noiseless.
# Finally use a Levenberg-Marquart ('lm') routine and the SH--DM model to fit
# geometric distortion of the DM relative to the SH.

import abbot.gradientOperator
import abbot.dm
import abbot.fourierSH as fourierSH
import numpy 
import sys
#import Zernike

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

#ei=lambda x : numpy.cos(x)+1.0j*numpy.sin(x)
#reshaper=lambda ip :\
#        ip.reshape([N,subApS,N,subApS]).swapaxes(1,2)

def getGrads(fSH,aperture,apWf):
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

def evaluatePMX(fSH,dmSize,dmSpacing,allActuators,dmActIdx,size,ipVector,verbose=1):
   ipVector=list(ipVector)
#   dmSize=(ipVector.pop(0), ipVector.pop(0))
#   dmSpacing=ipVector.pop(0)
   #
   dmRot=ipVector.pop(0)
   ifScl=ipVector.pop(0)
   dmScaling=(ipVector.pop(0), ipVector.pop(0))
   dmOffset=(ipVector.pop(0), ipVector.pop(0))
   dmRotLoc=(ipVector.pop(0), ipVector.pop(0))
   # \/ configure and setup a DM object
   dm = abbot.dm.dm(dmSize,dmSpacing,rotation=dmRot, ifScl=ifScl,
           lateralScl = dmScaling,
           lateralOffset = dmOffset,
           rotationLoc = dmRotLoc )

   # \/ Generate interaction matrix
   if verbose: print("INTERACTION MATRIX GENERATION::") ; sys.stdout.flush()
   pokeM=[]
   dmActIdx=dmActIdx if dmActIdx is not True else dm.usableIdx
   for i,actNo in enumerate(dmActIdx):
       thisApWf=dmFitter(size,dm.poke( actNo ),dm)
       pokeM.append(getGrads(fSH,aperture,thisApWf)[0])
       if verbose: _plotFractionalBar( (i+1)*len(dmActIdx)**-1.0 )

   if verbose: print()
   return(pokeM,dm)

def evaluateFSH(subApS,N):
   # \/ configure and setup a fourier Shack-Hartmann object
   nPix = subApS*N
   LT = 1 # lazy-truncation (fast)
   GP = 0 # guard-pixels (none)
   radialExtension = 0
   magnification = 1
   fSH = fourierSH.FourierShackHartmann( N, aperture, SHillumFraction,
           magnification, SHbinning, [0,], LT, GP, radialExtension
       )
   return(fSH)

if __name__=='__main__':
   ##
   ## ---- variables begin --------------------------
   ##
   numpy.random.seed(18071977)
   N=4
   subApS=2
   fftScl=1
   dmSize=[(N+1)*subApS]*2  # how big (in pixels) is the DM
   dmRot=0.1#(1*N**-1.0)*(1/3.14159*180) # in degrees
   dmOffset=(0.0123,0.3412) # actuator spacing units
   dmSpacing=[(N+1)]*2 # how many actuators (configuration)
   dmScaling=(1,1)#(1bb**-1.0,1+N**-1.0) # how the DM is magnified relative to the actuator scale
   allActuators = 1 # poke all actuators regardless of whether illuminated?
   obscurationR = 0 # fraction of pupil radius 
   ifScl = 0.50 # size of influence function
   dmRotLoc = (0,0)#N/2,N/2)
   SHbinning = 1 # 1=don't bin
   SHillumFraction = 0.0
   ##
   ## ---- variables end ----------------------------
   ##

#   assert not dmSize is None, "dmSize cannot be None in this code"

   print("VARIABLES::")
   print("\tSH configuration (NxN)={0:d}/{0:d}".format(N))
   print("\tDM configuration (MxM)=({0[0]:d}/{0[1]:d})".format(dmSpacing))
   print("\tsub-aperture # pixels={:d}".format(subApS))
   print("\tspot magnification={:d}".format(fftScl))
   print("\tDM size/pixels=({0[0]:d}/{0[1]:d})".format(dmSize))
   print("\tPoke all actuators?={:s}".format("YES" if allActuators else "No"))
   print("\t -- below are the parameters which are later used for the fitting section")
   print("\tDM rotation/degrees={:f}".format(dmRot))
   print("\tInfluence-fn={:3.1f}".format(ifScl))
   print("\tDM scaling/relative=({0[0]:3.1f}/{0[1]:3.1f})".format(dmScaling))
   print("\tDM offset=({0[0]:3.1f}/{0[1]:3.1f})".format(dmOffset))
   print("\tDM rotation location/relative=({0[0]:3.1f}/{0[1]:3.1f})".format(dmRotLoc))
   print()

   extraData = [] # REDUNDANT, value irrelevant
   size=N*subApS # pixels for DM/WFS representation
   aperture = numpy.ones([size]*2)#fourierSH.circle(size)
   if obscurationR>0:
       aperture-= fourierSH.circle(size,obscurationR)
   # \/ Setup actuators to poke
   if allActuators:
       dmActIdx=True
   else:
       # \/ assumption: DM-WFS are relatively well aligned to each other
       dmActIdx=(fourierSH.circle(
               max(dmSpacing), 
               min(dmSpacing)*max(dmSpacing)**-1.0,
#               ratio=max(dmSpacing)*min(dmSpacing)**-1.0 # this is needed but not supported ATM
           )!=0).ravel().nonzero()[0]

   print("CREATING SH MODEL")
   fSH = evaluateFSH(subApS,N)
   print("CREATING DM MODEL & POKING")
   pokeM,dm = evaluatePMX(fSH,dmSize,dmSpacing,allActuators,dmActIdx,size,
         [dmRot,ifScl]+list(dmScaling)+list(dmOffset)+list(dmRotLoc),
         verbose=1)

   pokeM=numpy.array(pokeM)
   if dmActIdx is True: dmActIdx=dm.usableIdx
   # \/ Analysis
   import pylab
#   pylab.figure(1)
#   pylab.imshow( pokeM, aspect='auto', cmap='gray' )

       # \/ location of DM actuator coordinates, relative to the centre of the
       #    array
   dmActCds = (
           numpy.array(dm.actCds)-(numpy.array(dm.npix)/2.0).reshape([1,2])
       ).take(dmActIdx,axis=0)
   dmActDistanceFromCentre = (dmActCds**2.0).sum(axis=1)**0.5
   print("Minimum DM actuator distance from centre (relative)={:f}".format(
           min(dmActDistanceFromCentre*2.0)/size ))
   print("Maximum DM actuator distance from centre (relative)={:f}".format(
           max(dmActDistanceFromCentre)*2.0/size ))
   print("Acceptable tolerance={:f}".format(N**-1.0))

   #    # \/ total poked surface
   pokeSurfaces = [ dmFitter(size,dm.poke(j),dm) for j in dmActIdx[::2] ]
   pokedSurface = numpy.sum( pokeSurfaces, axis=0 )
   pylab.figure(2)
   pylab.subplot(1,2,1)
   pylab.imshow( pokedSurface*aperture )
   pylab.title( "inside illuminated region" )
   pylab.subplot(1,2,2)
   pylab.imshow( pokedSurface*(1-aperture) )
   pylab.title( "outside illuminated region" )
   #
   #    # \/ variance of the slope signal per poked actuator
   pylab.figure(3)
   pylab.plot( numpy.var( pokeM, axis=1 ) )
   pylab.title( "Variance of slope signal" )
   pylab.xlabel( "DM actuators, order of those poked" )

   print("FITTING MODEL TO THE SYNTHETIC PMX")
   # Now fit to the derived pokeM:-
   # dmRot; ifScl; dmScaling (x2); dmOffset (x2); dmRotLoc (x2)
   #
   guess=(0,0.5, 1.0,1.0, 0.0,0.0, 0.0,0.0)

      # \/ State what the starting guess is: it also effectively overwrites and global
      # variables so this is a useful step to ensure fitting is not assuming existing values
   print("STARTING GUESS (Delta from previously stated values) IS:")
   copyOfGuess=list(guess)
   dmRot=copyOfGuess.pop(0)
   ifScl=copyOfGuess.pop(0)
   dmScaling=(copyOfGuess.pop(0), copyOfGuess.pop(0))
   dmOffset=(copyOfGuess.pop(0), copyOfGuess.pop(0))
   dmRotLoc=(copyOfGuess.pop(0), copyOfGuess.pop(0))
   print("\tDM rotation/degrees={:f}".format(dmRot))
   print("\tInfluence-fn={:3.1f}".format(ifScl))
   print("\tDM scaling/relative=({0[0]:3.1f}/{0[1]:3.1f})".format(dmScaling))
   print("\tDM offset=({0[0]:3.1f}/{0[1]:3.1f})".format(dmOffset))
   print("\tDM rotation location/relative=({0[0]:3.1f}/{0[1]:3.1f})".format(dmRotLoc))
   fitStartPokeM = evaluatePMX(fSH,dmSize,dmSpacing,allActuators,dmActIdx,size,guess,verbose=0
         )[0]

   import scipy.optimize
   def residual(x0):
      print(".",end="") ; sys.stdout.flush()
#      print(str(x0[0]),end="") ; sys.stdout.flush()
      x0_=x0.copy()
#      x0_[-2:]=0 # rotation location forced to (zero,zero)
      npokeM,dm = evaluatePMX(fSH,dmSize,dmSpacing,allActuators,dmActIdx,size,
            x0_, verbose=0)
      retVal=(npokeM-pokeM).ravel()
#      print(": {:5.3f}".format(retVal.std()))
      return( retVal ) 
   result = scipy.optimize.least_squares( residual, guess, verbose=1,
         x_scale=[180,0.5,1,1,1,1,1,1],
         diff_step=[1e-4,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7,1e-7])
  
   
   copyOfGuess=list(result.x)
   fitEndPokeM = evaluatePMX(fSH,dmSize,dmSpacing,allActuators,dmActIdx,size,
         copyOfGuess,verbose=0)[0]
   dmRot=copyOfGuess.pop(0)
   ifScl=copyOfGuess.pop(0)
   dmScaling=(copyOfGuess.pop(0), copyOfGuess.pop(0))
   dmOffset=(copyOfGuess.pop(0), copyOfGuess.pop(0))
   dmRotLoc=(copyOfGuess.pop(0), copyOfGuess.pop(0))
   print("\t -- below are the values after fitting")
   print("\tDM rotation/degrees={:f}".format(dmRot))
   print("\tInfluence-fn={:3.1f}".format(ifScl))
   print("\tDM scaling/relative=({0[0]:3.1f}/{0[1]:3.1f})".format(dmScaling))
   print("\tDM offset=({0[0]:3.1f}/{0[1]:3.1f})".format(dmOffset))
   print("\tDM rotation location/relative=({0[0]:3.1f}/{0[1]:3.1f})".format(dmRotLoc))

   pylab.figure()
   pylab.subplot(1,3,1) ; pylab.imshow( fitStartPokeM ) ; pylab.title("Start")
   pylab.subplot(1,3,2) ; pylab.imshow( pokeM ) ; pylab.title("Target")
   pylab.subplot(1,3,3) ; pylab.imshow( fitEndPokeM ) ; pylab.title("Fit")


