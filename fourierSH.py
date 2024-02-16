# -*- coding: utf-8 -*-
## [bug:1] magnification,binning==1->two-point calibration is insufficient.
"""ABBOT : class-based functions to make realistic SH patterns.

Note that the radial extension feature is not realistic for LGS
spot extension since it emulates the effect of a fully coherent source.
That is, no attempt is made to average over realizations of the reduction
in spatial coherence so it is entirely possible to obtain a restoration of the
spatial coherence (a removal of the elongation) with an input aberration.
Running the script through the python interpreter will test it.
"""

from __future__ import print_function
import numpy
import sys

### CONCEPT FOR OO VERSION
## A simple object, that contains functions to initiate the parameters to
## simulate a SH image, to create a SH image, and to process a SH image
## using a CoG algorithm.
##

## \/ should be in separate file
#
def cds(N, roll=False):
   tcds = (numpy.arange(0,N)-(N/2.-0.5))*(N/2.0)**-1.0
   return tcds if not roll else numpy.fft.fftshift(tcds)

## \/ should be in separate file
#
def circle(N,fractionalRadius=1):
   '''for N pixels, return a 2D array which has a circle (1 within, 0
   without) and radius a fraction of N.
   '''
   return numpy.add.outer(
         cds(N)**2,cds(N)**2 )<(fractionalRadius**2)

## \/ should be in separate file
#
def makeTiltPhase(nPix, fac):
   return -fac*numpy.pi/nPix*numpy.add.outer(
         numpy.arange(nPix),
         numpy.arange(nPix)
      )

## \/ should be in separate file
#
def rebin(ip,N):
   '''take the square 2D input and the number of sub-apertures and 
   return a 2D output which is binned over the sub-aperture elements
   and centred.
   '''
   nPix=ip.shape[0]
   sapxls=int( numpy.ceil(nPix*float(N)**-1.0) )
   N_=nPix//sapxls # the guessed number of original sub-apertures
   if N_==N:
#(2023-06-07, replaced)      return ip.reshape(
#(2023-06-07, replaced)         [N,sapxls,N,sapxls]).swapaxes(1,2).sum(axis=-1).sum(axis=-1)
      nx=ip
   else:
#(2023-06-07, replaced)      newNPix=(N+N_)*sapxls-nPix
      newNPix=int( N*sapxls )
#(2023-06-07, redundant)      assert newNPix%1==0,"newNPix isn't bigger by a multiple of sapxls"
#(2023-06-07, redundant)      newNPix=int(newNPix)
      dnp=newNPix-nPix
      nx=numpy.zeros([newNPix]*2,ip.dtype)
#(2023-06-07, replaced)      nx[ dnp//2:-dnp//2, dnp//2:-dnp//2 ]=ip
      nx[ dnp//2:dnp//2+nPix, dnp//2:dnp//2+nPix ]=ip 
   return nx.reshape([N,sapxls]*2).swapaxes(1,2).sum(axis=-1).sum(axis=-1)
# (2024-02-16, left from merge conflict) =======
# (2024-02-16, left from merge conflict)       return ip.reshape(
# (2024-02-16, left from merge conflict)          [N,sapxls,N,sapxls]).swapaxes(1,2).sum(axis=-1).sum(axis=-1)
# (2024-02-16, left from merge conflict)    else:
# (2024-02-16, left from merge conflict)       newNPix=(N+N_)*sapxls-nPix
# (2024-02-16, left from merge conflict)       assert newNPix%1==0,"newNPix isn't bigger by a multiple of sapxls"
# (2024-02-16, left from merge conflict)       newNPix=int(newNPix)
# (2024-02-16, left from merge conflict)       dnp=newNPix-nPix
# (2024-02-16, left from merge conflict)       nx=numpy.zeros([newNPix]*2,ip.dtype)
# (2024-02-16, left from merge conflict)       nx[ dnp//2:-dnp//2, dnp//2:-dnp//2 ]=ip
# (2024-02-16, left from merge conflict)       return nx.reshape([N,sapxls]*2).swapaxes(1,2).sum(axis=-1).sum(axis=-1)
# (2024-02-16, left from merge conflict) >>>>>>> 784de58c6a60bb07dedfaa8d807867751bb953d4

class FourierShackHartmann(object):
   '''A naïve implementation of a Shack-Hartmann wavefront sensor.
   
   '''

   def __init__( self, N, aperture, illuminationFraction,\
            magnification=2, binning=1, wavelengths=[0,],
            lazyTruncate=1, guardPixels=1, radialExtension=0,
            resampling=1 ):
      '''Parameters for instantiation are,
         N            : scale of sub-apertures
         aperture     : mask of aperture covering lenslets
         0<illuminationFraction<=1: fraction of illumination to accept
         magnification: magnification of each spot
         binning      : how much to expand and then contract each spot 
         resampling   : how much to bin each spot before assigning
         wavelengths  : simulate, crudely, polychromatism via the 'l' variable,
            monochromatic by default
         lazyTruncate : whether to assume each spot can't/won't move into it's
            neighbours region (don't preserve intensity, faster), or to do it
            properly (preserve intensity except at *simulated propagation*
            edges, slower)
         guardPixels  : how many (binned) pixels are ignored along the upper
            right edge of the simulated image, to model a guard-band.
         radialExtension: if >0, reduce spatial coherence to simulate radial
            elongation
            

         Note that binning is first used as a magnification factor and then
         as a factor to bin pixel by.
         This permits finer granulation of magnification.
         The resampling parameter, on the other hand, doesn't do thsi

         Note that guardPixels must be zero if using lazyTruncate; with
         lazyTruncate the effect is implicitly having a guard-band although the
         method used to achieve the effect is rather different from explicit
         lazyTruncate.
      '''
      assert aperture.shape[0]==aperture.shape[1],\
            "aperture array shape should be square"
      ( self.N, self.aperture, self.mag, self.binning, self.wls,
         self.lazyTrunc, self.guardPixels, self.RE, self.resampling
      ) = N, aperture, magnification, binning, wavelengths,  lazyTruncate,\
            guardPixels, radialExtension, resampling

      self.nPix = self.aperture.shape[0]
      self.illuminationFraction = illuminationFraction
      #
      sapxls=list([ int(numpy.ceil(s*N**-1.0)) for s in aperture.shape ])
      assert len(sapxls)==2 and sapxls[0]==sapxls[1], "Require a square, for now"
      self.sapxls = sapxls[0]
      self._makeCntrArr(self.sapxls//self.resampling)
      self.mask = rebin( aperture, N )*(self.sapxls**-2.0)
      self.maskB = (self.mask>=self.illuminationFraction)
      self.numSAs = self.maskB.sum()
      self.maskI = (self.maskB.ravel()).nonzero()[0] # index
      #
      self.slopeScaling, self.refSlopes = 1,0

   def _makeCntrArr( self, P,zeroOffset=False, quantize=1 ):
      '''Produce two 2D arrays which are constant in Y or X and linear increase
      from -1 to +1 in X or Y, respectively. For CoG centroiding.
      Optionally accounting for the single pixel offset so that zero is defined
      at an array element for even-sized arrays.
      quantize = number of pieces to quantize into e.g. 2 is equivalent
         to operating with a quadcell (just a negative or positive value)
      Returns a 4D array compatible with SH images for direct multiplication.
      '''
      linearComponent = numpy.linspace(
            -1, 1-(P**-1.0*2. if zeroOffset else 0), P
         )
      assert (P*quantize**-1.0)%1==0
      
      # quantize. A punishing algorithm that basically,
      #  a. reshapes the array and averages over quantize-sized segments
      #  b. creates a list of quantize-copies of the averaged segments
      #  c. reforms this into an array the same shape as that originally given
#(2017-06-06) Bug?      linearComponent = numpy.array(
#(2017-06-06) Bug?            linearComponent.reshape(
#(2017-06-06) Bug?                  [quantize,P//quantize]
#(2017-06-06) Bug?               ).mean(axis=1).reshape([1,-1]).tolist()*(P//quantize)
#(2017-06-06) Bug?         ).T.ravel()
      linearComponent = numpy.array(
            linearComponent.reshape(
                  [quantize,P//quantize]
               ).mean(axis=0).reshape([1,-1]).tolist()*(quantize)
         ).T.ravel()

      cntr=[ numpy.add.outer( linearComponent, numpy.zeros(P)) ]
      cntr.append( cntr[0].T )
      
      # Create an appropriate shape for multiplication with an array of
      # sub-aperture pixels for each of the two directions
      # Centroid along second axis (''x'') and then the first (''y'')
      self.cntr = numpy.array(cntr).T.reshape([1,1,P,P,2])
      #
      return self

   def makeImgs( self, phs, amplitudeScaling=1 ): 
      '''Produce Shack-Hartmann images.
         amplitudeScaling : an additional factor to aperture, could be constant
         phs : 2D array of phase
      '''
      # TODO got here
      assert type(phs) in (int,float) or ( len(phs)==len(self.aperture) ),\
            "phs should be a constant or same shape as aperture"
      # the number of pixels to use for the imaging
      sapxlsFP = self.sapxls*self.binning*self.mag 
      if self.RE>0:
##(to be finished)         rotatedCoordinate = lambda ang : numpy.add.outer(
##(to be finished)               numpy.cos(ang)*cds(sapxls),numpy.sin(ang)*cds(sapxls )
##(to be finished)            )
##(to be finished)         reduceSpatialCoherence = lambda ang,amp : amp*rotatedCoordinate(ang)**2.0
##(to be finished)         angles = numpy.arctan2( numpy.add.outer(numpy.zeros(N), cds(N)),
##(to be finished)                                 numpy.add.outer(cds(N), numpy.zeros(N))
##(to be finished)            )
##(to be finished)         spatialCoherenceReduction=numpy.empty( [N,N,sapxls,sapxls], numpy.complex128 )
##(to be finished)         for j in range(N):
##(to be finished)            for i in range(N):
##(to be finished)               radius = ( (i-N/2.+0.5)**2 + (j-N/2.0+0.5)**2 )**0.5
##(to be finished)               spatialCoherenceReduction[j,i]=numpy.exp(1.0j*
##(to be finished)                     reduceSpatialCoherence(angles[j,i],radialExtension*radius)
##(to be finished)                  )
##(to be finished)         spatialCoherenceReduction=spatialCoherenceReduction.swapaxes(1,2).reshape([N*sapxls]*2)
         spatialCoherenceReduction=1 # i.e. don't reduce spatial coherence anywhere
      else:
         spatialCoherenceReduction=1 # i.e. don't reduce spatial coherence anywhere
      if (sapxlsFP*self.mag)%1!=0:
         suggested_mags = sapxlsFP//( self.sapxls*self.binning )
         raise ValueError("Magnification {0:f} is too fine-grained, "\
               "suggest using either {1[0]:f} or {1[1]:f}".format(
                     self.mag,
                     (suggested_mag, suggested_mag+(sapxlsFP//self.mag))
                  )
            )
      if self.sapxls%self.resampling!=0:
         raise ValueError("Resampling {0:f} is too fine-grained".format(
                     self.resampling)
            )
      if sapxlsFP-max(self.wls)<1:
         raise ValueError("wls has too large a value, must be < {0:d}".format(
               sapxlsFP-1))
      if self.sapxls*self.mag-max(self.wls)//self.binning<0:
         raise ValueError("wls has too large a value, must be < {0:d}".format(
               self.sapxls*self.mag*self.binning))
      for twl in self.wls:
         if (sapxlsFP-twl)%self.binning:
            raise ValueError("wl {0:d} doesn't work with binning:")
      FFTCentringPhase=lambda twl :\
            makeTiltPhase( self.nPix,-self.nPix*(sapxlsFP-twl)**-1.0)
      A = self.aperture
      C = amplitudeScaling*spatialCoherenceReduction
      polyChromSHImgs = []
      for twl in self.wls:
#(debugging)         print(A.shape,C,phs,FFTCentringPhase(twl).shape)
         tip = rebin( A*C*numpy.exp(
                  -1.0j*( phs+FFTCentringPhase(twl) ) 
                )
               ,self.N*self.sapxls).reshape( [self.N, self.sapxls]*2 ).swapaxes(1,2)
         top = abs( numpy.fft.fftshift(
               numpy.fft.fft2( tip, s=[sapxlsFP+twl]*2
                  ), (2,3) ) )**2.0 # no requirement to preserve phase therefore only one fftshift
         top = top.reshape([ self.N, self.N ]+
                  [  (sapxlsFP+twl)//self.binning, self.binning,
                     (sapxlsFP+twl)//self.binning, self.binning
                  ]
               ).sum(axis=-3).sum(axis=-1)
         polyChromSHImgs.append( top ) # save everything, for now
       
      # n.b. binning has been done at this stage
      if self.lazyTrunc:
         # algorithm: truncate each image to the size of the sub-aperture
         #  and compute an average
         for ( i,twl ) in enumerate( self.wls ):
            idx = ( (self.sapxls*self.mag)//2 + twl//self.binning//2 )+\
                  numpy.arange( -self.sapxls//2, self.sapxls//2 )
            polyChromSHImgs[i] =\
                  polyChromSHImgs[i].take(idx,axis=2).take(idx,axis=3)
         self.lastSHImage = numpy.mean( polyChromSHImgs, axis=0 )
      else:
         # algorithm: make a canvas upon which SH sub-aperture images are
         #  painted, and then place for each sa, the result for each wl, all
         #  made to the same size.
         # If using guardPixels, then make a bigger canvas and then take
         #  out the unecessary columns
         canvas=numpy.zeros(
               [ len(self.wls) ]+
               [ self.nPix+self.sapxls*self.mag+
                 max(self.wls)//self.binning + self.guardPixels*self.N ]*2,
               top.dtype
            )
         for i,twl in enumerate(self.wls):
            width = self.sapxls*self.mag+twl//self.binning
            for l in range(self.N): # vertical
               for m in range(self.N): # horizontal
                  cornerCoords = [
                       (self.sapxls)*int(
                           0.5+v+(max(self.wls)-twl)//self.binning//2)+
                       (self.guardPixels*v) for v in (l,m)
                      ]
                  canvas[i, cornerCoords[0]:cornerCoords[0]+width,
                            cornerCoords[1]:cornerCoords[1]+width
                     ] += polyChromSHImgs[i][l,m]
         offset = (self.sapxls*self.mag)/2+max(self.wls)//self.binning//2
         # trim and remove any guard-pixels
         idx = numpy.add.outer(
                   numpy.arange(self.N)*(self.sapxls+self.guardPixels),
                   offset+numpy.arange(self.sapxls) ).ravel().astype('i')
         canvas = canvas.mean( axis=0 ).take( idx, axis=0 ).take( idx, axis=1 )
         # reshape the canvas, resample, and store
         self.lastSHImage = canvas.reshape(
               [ self.N, self.sapxls//self.resampling,self.resampling ]*2 
            ).sum(axis=2).sum(axis=5-1).swapaxes(1,2)
      #
      return self

   def getSlopes( self ):
      '''Based on the last image produced, and the calibrations, estimate
      the slopes using a CoG algorithm.
      Order is all the slopes along the second axis, and then all the slopes
      along the first. This is (visually) equivalent to x-slopes and then
      y-slopes, if indices run along the x-direction (column-then-row order).
      '''
      rawSlopes=(
            (self.cntr*
               self.lastSHImage.reshape(list(self.lastSHImage.shape)+[1])
               ).sum(axis=2).sum(axis=2)
           /(1e-10+
               self.lastSHImage.reshape(list(self.lastSHImage.shape)+[1])
               ).sum(axis=2).sum(axis=2)
         )
      slopesV = rawSlopes.reshape([-1,2]
            ).take( self.maskI, axis=0 ).T.ravel()
      self.lastSlopesV = numpy.ma.masked_array( slopesV, numpy.isnan(slopesV) )
      return (self.lastSlopesV-self.refSlopes)*self.slopeScaling

   def calibrate( self ):
      """Calibrate the SH model by adding artificial tilts that move the spots
         by 1/4 and 1/2 of the sub-aperture width in both directions
         simultaneously.
      """
## [bug:1] For magnification & binning = 1 (smallest spot) the two-point
##  calibration procedure is insufficient because the spot response to tilt
##  is non-linear. This should be flagged as a potential issue
##
      # Make reference values
      self.makeImgs( 0,1 )
      self.refSlopes = self.getSlopes() # raw slopes
      self.refSHImage = self.lastSHImage.copy() # raw pixels
      assert self.refSlopes.mask.sum()==0, "All refslopes are invalid!"
      assert self.refSlopes.mask.var()==0, "Some refslopes are invalid!"
      #
      # move 1/4 sub-aperture
      self.tiltFac =\
            0.5*self.nPix*(self.sapxls*self.mag//self.resampling)**-1.0 
      tiltphs = makeTiltPhase( self.nPix, self.tiltFac )
      # now, 1/4 sub-aperture for sapxls means that the signal from tiltphs
      # is the tilt for a slope of:
      #  tiltFac*numpy.pi/nPix
      # =0.5*numpy.pi/(sapxls*mag)
      # or the difference between adjacent wavefront grid points of:
      #  0.5*numpy.pi/mag
      # so scaling upon reconstruction should be:
      self.reconScalingFactor = 0.5*numpy.pi*self.mag**-1.0
      # now add a negative factor (don't know why)
      self.reconScalingFactor *= -1
      # and finally the effect of rebinning into sapxls^2
      self.reconScalingFactor *= self.sapxls//self.resampling**2.
      #
      # get unscaled tilt slopes
      ## TILT x1
      self.makeImgs( tiltphs,  1 )
      self.tilt1xSHImage = self.lastSHImage.copy() # tilt 1 pixel
      tilt1xslopes = self.getSlopes()
      self.tilt1xslopes=numpy.ma.masked_array(
            tilt1xslopes.data, tilt1xslopes.mask+tilt1xslopes.data<1e-6 )
      assert self.tilt1xslopes.mask.sum()<len(self.tilt1xslopes),\
            "All tilt1xslopes are invalid!"
      if self.tilt1xslopes.mask.var()!=0: 
         print( "WARNING: Some tilt1xslopes are invalid!" )
      #
      ## TILT x2
      self.makeImgs( tiltphs*2, 1 )
      self.tilt2xSHImage = self.lastSHImage.copy() # tilt 2 pixel
      tilt2xslopes = self.getSlopes()
      self.tilt2xslopes=numpy.ma.masked_array(
            tilt2xslopes.data, tilt2xslopes.mask+tilt2xslopes.data<1e-6 )
      assert self.tilt2xslopes.mask.sum()<len(self.tilt2xslopes),\
            "All tilt2xslopes are invalid!"
      if self.tilt2xslopes.mask.var()!=0:
         print("WARNING: Some tilt1xslopes are invalid!")
      # now generate slope scaling
      self.slopeScaling =\
            0.5*(1/(1e-15+self.tilt1xslopes)+2/(1e-15+self.tilt2xslopes))
      return self


if __name__=="__main__":

   def doFourierShackHartmannObject( N, pupilAp, illuminationFraction, mag,
         binning, defWls, LT, GP, radialExtension
         ):
      import abbot.fourierSH
      fSH = abbot.fourierSH.FourierShackHartmann(
            N, pupilAp, illuminationFraction, mag, binning, defWls, LT, GP,
            radialExtension
         )
      fSH.calibrate()
      return( fSH )

   def doGOpAndRMX(subapMask,dopinv=True):
      import abbot.gradientOperator
      print("GRADIENT OP & RECON MX",end="...");sys.stdout.flush()
      gO=abbot.gradientOperator.gradientOperatorType1( subapMask )
      print("^^^",end="");sys.stdout.flush()
      gM=gO.returnOp()
      if not dopinv:
         print("MX inv",end="");sys.stdout.flush()
         igM=numpy.linalg.inv(
               gM.T.dot(gM)+1e-3*numpy.identity(gO.numberPhases)).dot(gM.T)
      else:
         print("MX SVD",end="");sys.stdout.flush()
         igM=numpy.linalg.pinv(gM,1e-3)
      print("(done)");sys.stdout.flush()
      #
      return gO,gM,igM

   def doLoopOpAndNRMX(subapMask): 
      import abbot.continuity
      print("LOOPS_DEFN & NOISE_REDUC.",end="...");sys.stdout.flush()
      loopsDef=abbot.continuity.loopsIntegrationMatrix( subapMask )
      nrDef=abbot.continuity.loopsNoiseMatrices( subapMask )
      loopIntM=loopsDef.returnOp()
      neM,noiseReductionM=nrDef.returnOp()
      print("(done)");sys.stdout.flush()
      #
      return loopIntM, noiseReductionM

   def doInputPhase(nPix,ips):
      import kolmogorov
      r0,sapxls=ips
      print("PHASE_SCREEN",end="...");sys.stdout.flush()
      scr=kolmogorov.TwoScreens(
            nPix*4,r0,flattening=2*numpy.pi*sapxls**-1.0)[0][:nPix,:nPix]
      print("(done)");sys.stdout.flush()
      return scr

   ##
   ## ===============================================
   ## CODE LOGIC AND CONFIGURATION 
   ## ===============================================
   ##

   def checkMain():
      def doComparison(scr, scrslopes, igM, nrM, pupilAp, N, sapxls,
            gO, fSH, quiet=0
         ):
         reconScalingFactor = fSH.reconScalingFactor
            # \/ reconstruct from the slopes
         inv_scrV = numpy.dot(igM,scrslopes)
            # \/ reconstruct from the noise reduced slopes
         inv_scr_nrV = numpy.dot(igM,nrM.dot(scrslopes))
         delta_inv_scr_nrV = inv_scrV-inv_scr_nrV # difference

         # now make visual comparisons
         wfGridMask = rebin(pupilAp,N+1)!=sapxls**2
         orig_scr = numpy.ma.masked_array(
               rebin(pupilAp*scr,N+1), wfGridMask )
         inv_scr,inv_scr_nr,delta_inv_scr_nr = [
               numpy.ma.masked_array(
                  numpy.empty(gO.n_,numpy.float64), wfGridMask
               ) for dummy in (1,2,3) ]
         for opArray, data in (
                  (inv_scr,inv_scrV),
                  (inv_scr_nr,inv_scr_nrV),
                  (delta_inv_scr_nr,delta_inv_scr_nrV)
               ):
            opArray.ravel()[ gO.illuminatedCornersIdx ]=\
                  (data-data.mean())*reconScalingFactor
         #
         if quiet:
            return (inv_scr,orig_scr,inv_scr_nr)
         else:
            print("COMPARISON:")
            for ip,opStem in (
                     (scrslopes.var(),"var{scrslopes}"),
                     (loopIntM.dot(scrslopes).var()/16.0,
                        "1/16 loopIntM{scrslopes}"),
                     (orig_scr.var(),"var{orig}"),
                     (inv_scr.var(),"var{inv}"),
                     ((inv_scr-orig_scr).var()/orig_scr.var(),
                        "var{inv-orig}/var{orig}"),
                     ((inv_scr_nr-orig_scr).var()/orig_scr.var(),
                        "var{NR{inv}-orig}/var{orig}"),
                     (delta_inv_scr_nr.var()/orig_scr.var(),
                        "var{NR{inv}-inv}/var{orig}"),
                  ):
               print("{0:s}{1:s} = {2:7.5f}".format( " "*(30-len(opStem)),
                     opStem,ip)
                  )
      
      def doComparisonWithNoise(N,shscrimgs,
            igM,nrM,scr,nPix,sapxls,gO,reconScalingFactor,
            noiseSDscaling=1,nLoops=100,quiet=0):
         if not quiet:
            print("NOISE ADDED: SD scaling={0:f}".format(noiseSDscaling))
         noiseSD = (1e-9 if noiseSDscaling<1e-9 else noiseSDscaling)\
               *fSH.refSHImage.reshape([N,N,-1]).sum(axis=-1).max()*N**-2.0
         keys='inv','inv_nr','loopInt','inv-orig'
         variances={}
         #
         for i in range( nLoops ):
               # \/ replace the SH object pixels with a noisy version
            fSH.lastSHImage = numpy.random.normal(shscrimgs, noiseSD) 
            scrslopesNoisy = fSH.getSlopes()
            (inv_scr,orig_scr,inv_scr_nr)=doComparison( scr, scrslopesNoisy,
                  igM, nrM, pupilAp, N, sapxls, gO, fSH, quiet=1
               )
            for key, data in [
                     ('inv', inv_scr), ('inv-orig', inv_scr-orig_scr),
                     ('inv_nr', inv_scr_nr),
                     ('loopInt', loopIntM.dot(scrslopesNoisy)/16.0)
                  ]:
               if key not in variances: variances[key]=[]
               variances[key].append(data.var())
         shscrimgsNoisy = fSH.lastSHImage.copy()
         fSH.lastSHImage = shscrimgs
         #
         if not quiet:
            for key in keys:
               opStem="var{{{0:s}}}".format(key)
               print("{2:s}{3:s}={0:5.3f}+/-{1:5.3f}".format(
                     numpy.mean( variances[key] ),
                     numpy.std( variances[key] ),
                     " "*(20-len(opStem)),
                     opStem )
                  )
         #
         return ( variances, shscrimgsNoisy, orig_scr.var() )
         
      def doPlotting():
            # \/ reconstruct from the slopes
         inv_scrV=numpy.dot(igM,scrslopes)

         # now make visual comparisons
         wfGridMask=rebin(pupilAp,N+1)!=sapxls**2
         orig_scr=numpy.ma.masked_array(
               rebin(pupilAp*scr,N+1), wfGridMask )
         inv_scr=numpy.ma.masked_array(
               numpy.empty(gO.n_,numpy.float64), wfGridMask )
         inv_scr.ravel()[gO.illuminatedCornersIdx] =\
               inv_scrV*fSH.reconScalingFactor
         pyplot.figure(99)
         for i,(thisImg,title) in enumerate([
                  (fSH.refSHImage,"reference"),
                  (fSH.tilt1xSHImage,"tilt1x"),
                  (fSH.tilt2xSHImage,"tilt2x"),
                  (fSH.lastSHImage,"scr"),
               ]):
            pyplot.subplot(2,2,1+i)
            pyplot.imshow( abs(thisImg.swapaxes(1,2).reshape([nPix]*2))**0.5,
                  cmap='cubehelix', vmax=(sapxls)**2.0)
            pyplot.title("sqrt{{{0:s}}}".format(title))

         pyplot.figure(98)
         for ip in inv_scr,orig_scr:
            ip-=ip.mean()

         pyplot.subplot(2,1,1)
         pyplot.plot( orig_scr.ravel(), label="orig_scr")
         pyplot.plot( inv_scr.ravel(), label="inv_scr")
         pyplot.legend(loc=0)
         pyplot.subplot(2,2,3)
         pyplot.imshow( orig_scr, vmin=-orig_scr.ptp()/2,
               vmax=orig_scr.ptp()/2
            )
         pyplot.title( "orig_scr" )
         pyplot.subplot(2,2,4)
         pyplot.imshow( inv_scr, vmin=-orig_scr.ptp()/2,
               vmax=orig_scr.ptp()/2
            )
         pyplot.title( "inv_scr" )

      def doSetup():
         global mask
         print("SETUP",end="...");sys.stdout.flush()
         sapxls=nPix//N
         if 'pupilAp' not in dir():
            pupilAp=circle(nPix,1) # pupil aperture
         mask=rebin(pupilAp,N)*(sapxls**-2.0)
         maskB=(mask>=illuminationFraction)
         numSAs=maskB.sum()
         maskI=(maskB.ravel()).nonzero()[0]
         print("(done)");sys.stdout.flush()
         return sapxls, None, maskB, numSAs, maskI

      ## -- begin variables --
      ##
      defWls = [0,]#range(-4,4+2,4)
      N = 20# how many sub-apertures
      nPix = N*8# total size of pupil as a diameter
      numpy.random.seed(18071977)
      r0 = (nPix/N) # pixels
      mag = 2
      illuminationFraction = 0.1
      pupilAp = circle(nPix,1) # pupil aperture
      binning = 1
      LT = 0 # lazy truncation
      GP = 1 # guardPixels
      noiseSDscalings = (1e-9,1.0,10.0,20.0,30.0,33.0,38.0,40.0)
      radialExtension = 0.3 # if >0, do radial extension of spots
      ##
      ## -- end variables ----
      print("sapxls =\t",end="")
      sapxls=nPix//N
      if sapxls==2:
         print("Quadcells")
      else:
         print(sapxls)
      print("Lazy truncation" if LT else "Proper truncation")
      print(("No" if not GP else "{0:d}".format(GP))+
            " guard pixel"+("s" if GP>1 else ""))
      print(("No" if not radialExtension else "{0:5.3f}x".format(
               radialExtension
            ))+ " radial extension of spots"
         )
         # \/ setup the basic parameters for the SH array
      sapxls, cntr, maskB, numSAs, maskI=doSetup()
         # \/ create the SH object and calibrate it
      fSH = doFourierShackHartmannObject( N, pupilAp, illuminationFraction,
            mag, binning, defWls, LT, GP, radialExtension
         )
         # \/ create the gradient operator and the reconstructor
      gO,gM,igM = doGOpAndRMX(maskB)
      assert gO.numberSubaps==numSAs
         # \/ loop integration operator and noise reducer
      loopIntM, nrM = doLoopOpAndNRMX(maskB)
         # \/ make input
      scr = doInputPhase(nPix,(r0,sapxls))
         # \/ create the SH images and process them
      fSH.makeImgs( scr )
      shscrimgs = fSH.lastSHImage.copy()
      scrslopes=fSH.getSlopes()
      assert scrslopes.mask.sum()<len(scrslopes), "All scrslopes are invalid!"
      if scrslopes.mask.var()>0: "WARNING: Some scrslopes are invalid!"
         # \/ do noiseless comparison
      doComparison( scr, scrslopes, igM, nrM, pupilAp, N, sapxls,
            gO, fSH, quiet=0
         )
         # \/ do noisy comparison (normal noise on images)
      data=[]
      for noiseSDscaling in noiseSDscalings:
         ( variances,lastImg,ipVar ) = doComparisonWithNoise(
               N, shscrimgs, igM, nrM, scr, nPix, sapxls, gO,
               fSH, noiseSDscaling,
            )
         sys.stdout.flush()
         data.append( [ noiseSDscaling, variances, lastImg ] )
         # \/ plot!
      pyplot.figure(97)
      plotData = []
      nRows = int(numpy.ceil(len(noiseSDscalings)/4.0))
      for j,thisData in enumerate( data ):
         ( thisLastImg, thisSD, thisVariances ) =\
               ( thisData[2], thisData[0], thisData[1] )
         thisLastImg = thisLastImg.swapaxes(1,2).reshape([nPix]*2)
         pyplot.subplot(4,nRows,j+1)
         pyplot.imshow( thisLastImg, cmap='cubehelix' )
         for ax in pyplot.gca().get_xaxis(),pyplot.gca().get_yaxis():
            ax.set_visible(0)
         pyplot.title( "{0:g}".format( thisSD ),size=10 )
         plotData.append( [ j, thisSD ]+
               [ [ numpy.mean(thisVariances[k]),
                   numpy.var(thisVariances[k])
                 ] for k in ('loopInt','inv','inv-orig')
               ]
            )
      plotData = numpy.array([ x[:2]+x[2]+x[3]+x[4] for x in plotData ])
      pyplot.figure(96)
      pyplot.subplot(2,1,1)
      pyplot.errorbar( plotData[:,1], plotData[:,2] )#, yerr=plotData[:,3] )
      pyplot.title("loopIntM with noise")
      pyplot.subplot(2,1,2)
      pyplot.errorbar( plotData[:,1],
            plotData[:,4], label='<var{inv}>' )
      pyplot.errorbar( plotData[:,1],
            abs(plotData[:,4]-ipVar), label='|<var{inv}>-var{orig}|' )
      pyplot.semilogy( plotData[:,1],
            plotData[:,6], label='<var{inv-orig}>' )
      pyplot.legend(loc=0)
      pyplot.title("inv with noise")
         # \/ do some plotting to finish up
      doPlotting()


   ## now test the shifting via lazyTruncation
   #
   def checkLazyTruncate():
      aperture=circle(128,0.75)
      tiltphs=makeTiltPhase(128,10.0)
      imgs={'LT':{'tilt':[],'noTilt':[]},'noLT':{'tilt':[],'noTilt':[]}}
      for lt in (0,1):
         for tilt in (0,1):
            fSH = doFourierShackHartmannObject( 16, aperture, 1.0,
                  4, 1, [0,], lt, 0, 0 
               )
            fSH.makeImgs( tiltphs*tilt )
            imgs['LT' if lt else 'noLT']['tilt' if tilt else 'noTilt'].append(
                  fSH.lastSHImage.copy()
               )
      pyplot.figure(80)
      plotNo=1
      for lt in (0,1):
         for tilt in (0,1):
            pyplot.subplot(2,2,plotNo)
            codes='LT' if lt else 'noLT', 'tilt' if tilt else 'noTilt'
            thisImg=imgs[codes[0]][codes[1]][0].swapaxes(1,2).reshape([128]*2)
            if lt==0 and tilt==0:
               refImgSum=float(thisImg.sum())
            pyplot.imshow( thisImg**0.5, cmap='gray' )
            fractionKept=thisImg.sum()/refImgSum
            opStr="{0[0]:s}/{0[1]:s}->{1:4.2f}".format(
                  codes, fractionKept)
            imgs[codes[0]][codes[1]].append( fractionKept )
            print("\t"+opStr);pyplot.title(opStr)
            plotNo+=1
            if lt==0:
               assert fractionKept>0.95, "for no lazy truncation, lost too much"
            if lt==0 and tilt==1:
               assert fractionKept>0.95*imgs['noLT']['noTilt'][1],\
                     "for no lazy truncation and tilt, lost too much"
            if lt==1:
               assert fractionKept<=1, "lazy truncation has too much intensity?"
            if lt==1 and tilt==1:
               assert fractionKept<imgs['LT']['noTilt'][1],\
                     "lazy truncation not losing intensity"

      return imgs

   from matplotlib import pyplot
   print("BEGINS: test code for fourierSHexamples")
   import types
   for objectName in dir():
      print("("+objectName[:len('check')]+")",end="")
      if 'check'==objectName[:len('check')]\
            and type(eval(objectName))==types.FunctionType:
         print("\n\tFound checking function: '{0:s}', "\
               "will call without parameters".format( objectName ), end="\n"*2
            )
         try:
            eval(objectName+"()")
         except:
            print("ERROR:{0:s} FAILED".format(objectName))
            print(sys.exc_info()) 
         else:
            print("PASSED:{0:s}".format(objectName))
   print("ENDS")
