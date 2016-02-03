# -*- coding: utf-8 -*-
"""ABBOT : class-based functions to make realistic SH patterns.
Note that the radial extension feature is not realistic for LGS
spot extension since it emulates the effect of a fully coherent source.
That is, no attempt is made to average over realizations of the reduction
in spatial coherence so it is entirely possible to obtain a restoration of the
spatial coherence (a removal of the elongation) with an input aberration.
"""

from __future__ import print_function
import numpy
import sys

##(redundant?)def _plotFractionalBar(frac,char='#',length=70,
##(redundant?)      printFraction=False,
##(redundant?)      printPercentage=True,
##(redundant?)      spinner=False
##(redundant?)   ):
##(redundant?)   print(
##(redundant?)      "[ "+
##(redundant?)      char*int(frac*length)+
##(redundant?)      "-"*(length-int(frac*length))+
##(redundant?)      " ]"+
##(redundant?)      ("" if not printPercentage else " {0:3d}%\r".format(int(frac*100))
##(redundant?)         )+
##(redundant?)      ("" if not printFraction else " {0:6.3f}\r".format(frac)
##(redundant?)         )+
##(redundant?)      ("" if not spinner else spinner)
##(redundant?)      , end="" )
##(redundant?)   sys.stdout.flush()


### CONCEPT FOR OO VERSION
## A simple object, that contains functions to initiate the parameters to
## simulate a SH image, to create a SH image, and to process a SH image
## using a CoG algorithm.
##

def cds(N, roll=False):
   tcds = (numpy.arange(0,N)-(N/2.-0.5))*(N/2.0)**-1.0
   return tcds if not roll else numpy.fft.fftshift(tcds)

def makeTiltPhase(nPix, fac):
   return -fac*numpy.pi/nPix*numpy.add.outer(
         numpy.arange(nPix),
         numpy.arange(nPix)
      )

class FourierShackHartmann(object):
   '''A naïve implementation of a Shack-Hartmann wavefront sensor.
   
   '''

   def __init__( self, N, aperture, illuminationFraction,\
            magnification=2, binning=1, wavelengths=[0,],
            lazyTruncate=1, guardPixels=1, radialExtension=0 ):
      '''Parameters for instantiation are,
         N            : scale of sub-apertures
         aperture     : mask of aperture covering lenslets
         0<illuminationFraction<=1: fraction of illumination to accept
         magnification: magnification of each spot
         binning      : how much to bin each spot before assigning
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

         Note that guardPixels must be zero if using lazyTruncate; with
         lazyTruncate the effect is implicitly having a guard-band although the
         method used to achieve the effect is rather different from explicit
         lazyTruncate.
      '''
      assert aperture.shape[0]==aperture.shape[1],\
            "aperture array shape should be square"
      ( self.N, self.aperture, self.mag, self.binning, self.wls,
         self.lazyTrunc, self.guardPixels, self.RE
      ) = N, aperture, magnification, binning, wavelengths,  lazyTruncate,\
            guardPixels, radialExtension

      self.nPix = self.aperture.shape[0]
      self.illuminationFraction = illuminationFraction
      #
      sapxls=map(lambda s : int(numpy.ceil(s*N**-1.0)), aperture.shape)
      assert len(sapxls)==2 and sapxls[0]==sapxls[1], "Require a square, for now"
      self.sapxls = sapxls[0]
      self._makeCntrArr(self.sapxls)
      self.mask = self._rebin( aperture, N )*(self.sapxls**-2.0)
      self.maskB = (self.mask>=self.illuminationFraction)
      self.numSAs = self.maskB.sum()
      self.maskI = (self.maskB.ravel()).nonzero()[0] # index
      #
      self.slopeScaling, self.refSlopes = 1,0
   
   def _rebin( self, ip, N ):
      '''take the square 2D input and the number of sub-apertures and 
      return a 2D output which is binned over the sub-aperture elements
      and centred.
      '''
      nPix=ip.shape[0]
      sapxls=map( lambda s : int(numpy.ceil(s*N**-1.0)), self.aperture.shape )
      assert len(sapxls)==2 and sapxls[0]==sapxls[1], "Require a square, for now"
      sapxls=sapxls[0]
      N_ = nPix//sapxls # the guessed number of original sub-apertures
      if N_==N:
         return ip.reshape(
            [N,sapxls,N,sapxls]).swapaxes(1,2).sum(axis=-1).sum(axis=-1)
      else:
         newNPix=(N+N_)*sapxls-nPix
         assert newNPix%1==0,"newNPix isn't bigger by a multiple of sapxls"
         newNPix=int(newNPix)
         dnp=newNPix-nPix
         nx=numpy.zeros([newNPix]*2,ip.dtype)
         nx[ dnp//2:-dnp//2, dnp//2:-dnp//2 ]=ip
         return nx.reshape([N,sapxls]*2).swapaxes(1,2).sum(axis=-1).sum(axis=-1)

   def _makeCntrArr( self, P,zeroOffset=True ):
      '''Produce two 2D arrays which are constant in Y or X and linear increase
      from -1 to +1 in X or Y, respectively. For CoG centroiding.
      Optionally accounting for the single pixel offset so that zero is defined
      at an array element for even-sized arrays.
      Returns a 4D array compatible with SH images for direct multiplication.
      '''
      cntr=[  numpy.add.outer(
            (numpy.linspace(-1,1-(P**-1.0*2. if zeroOffset else 0),P)),
            numpy.zeros(P)) 
         ]
      cntr.append( cntr[0].T )
      self.cntr = numpy.array(cntr).T.reshape([1,1,P,P,2])

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
         suggested_mags=( (sapxls*self.binning)*self.mag )//(sapxls*self.binning)
         raise ValueError("Magnification {0:f} is too fine-grained, "\
               "suggest using either {1[0]:f} or {1[1]:f}".format(
                     self.mag, (suggested_mag, suggested_mag+(sapxls*binnng)))
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
         tip = ( A*C*numpy.exp(
                  -1.0j*( phs+FFTCentringPhase(twl) ) 
                )
               ).reshape( [self.N, self.sapxls]*2 ).swapaxes(1,2)
         top = abs( numpy.fft.fftshift(
               numpy.fft.fft2( tip, s=[sapxlsFP+twl]*2
                  ), (2,3) ) )**2.0 # no requirement to preserve phase therefore only one fftshift
         top = top.reshape([ self.N, self.N ]+
                  [(sapxlsFP+twl)//self.binning,self.binning,
                   (sapxlsFP+twl)//self.binning,self.binning]
               ).sum(axis=-3).sum(axis=-1)
         polyChromSHImgs.append( top ) # save everything, for now
       
      # n.b. binning has been done at this stage
      if self.lazyTrunc:
         # algorithm: truncate each image to the size of the sub-aperture
         #  and compute an average
         for ( i,twl ) in enumerate( self.wls ):
            idx = ( (self.sapxls*self.mag)/2 + twl//self.binning/2 )+\
                  numpy.arange( -self.sapxls/2, self.sapxls/2 )
            polyChromSHImgs[i] =\
                  polyChromSHImgs[i].take(idx,axis=2).take(idx,axis=3)
         self.lastSHImage = numpy.mean( polyChromSHImgs, axis=0 )
         return
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
                  cornerCoords = map( lambda v :
                       (self.sapxls)*(0.5+v+(max(self.wls)-twl)//self.binning//2)+
                       (self.guardPixels*v), (l,m)
                     )
                  canvas[i, cornerCoords[0]:cornerCoords[0]+width,
                            cornerCoords[1]:cornerCoords[1]+width
                     ] += polyChromSHImgs[i][l,m]
         offset = (self.sapxls*self.mag)/2+max(self.wls)//self.binning//2
         # trim and remove any guard-pixels
         idx = numpy.add.outer(
               numpy.arange(self.N)*(self.sapxls+self.guardPixels),
               offset+numpy.arange(self.sapxls) ).ravel()
         canvas = canvas.mean( axis=0 ).take( idx, axis=0 ).take( idx, axis=1 )
         self.lastSHImage = canvas.reshape(
               [ self.N, self.sapxls ]*2 
            ).swapaxes(1,2)
         return

   def getSlopes( self ):
      '''Based on the last image produced, and the calibrations, estimate
      the slopes using a CoG algorithm.
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
      # Make reference values
      self.makeImgs( 0,1 )
      self.refSlopes = self.getSlopes() # raw slopes
      self.refSHImage = self.lastSHImage.copy() # raw pixels
      assert self.refSlopes.mask.sum()==0, "All refslopes are invalid!"
      assert self.refSlopes.mask.var()==0, "Some refslopes are invalid!"
      #
      # move 1/4 sub-aperture
      self.tiltFac = 0.5*self.nPix*(self.sapxls*self.mag)**-1.0 
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
      self.reconScalingFactor *= self.sapxls**2.
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
      return 


#(disabled and non-functional)if __name__=="__main__":
#(disabled and non-functional)
#(disabled and non-functional)   def doGOpAndRMX(subapMask,dopinv=True):
#(disabled and non-functional)      import abbot.gradientOperator
#(disabled and non-functional)      print("GRADIENT OP & RECON MX",end="...");sys.stdout.flush()
#(disabled and non-functional)      gO=abbot.gradientOperator.gradientOperatorType1( subapMask )
#(disabled and non-functional)      print("^^^",end="");sys.stdout.flush()
#(disabled and non-functional)      gM=gO.returnOp()
#(disabled and non-functional)      if not dopinv:
#(disabled and non-functional)         print("MX inv",end="");sys.stdout.flush()
#(disabled and non-functional)         igM=numpy.linalg.inv(
#(disabled and non-functional)               gM.T.dot(gM)+1e-3*numpy.identity(gO.numberPhases)).dot(gM.T)
#(disabled and non-functional)      else:
#(disabled and non-functional)         print("MX SVD",end="");sys.stdout.flush()
#(disabled and non-functional)         igM=numpy.linalg.pinv(gM,1e-3)
#(disabled and non-functional)      print("(done)");sys.stdout.flush()
#(disabled and non-functional)      #
#(disabled and non-functional)      return gO,gM,igM
#(disabled and non-functional)
#(disabled and non-functional)   def doLoopOpAndNRMX(subapMask): 
#(disabled and non-functional)      import abbot.continuity
#(disabled and non-functional)      print("LOOPS_DEFN & NOISE_REDUC.",end="...");sys.stdout.flush()
#(disabled and non-functional)      loopsDef=abbot.continuity.loopsIntegrationMatrix( subapMask )
#(disabled and non-functional)      nrDef=abbot.continuity.loopsNoiseMatrices( subapMask )
#(disabled and non-functional)      loopIntM=loopsDef.returnOp()
#(disabled and non-functional)      neM,noiseReductionM=nrDef.returnOp()
#(disabled and non-functional)      print("(done)");sys.stdout.flush()
#(disabled and non-functional)      #
#(disabled and non-functional)      return loopIntM, noiseReductionM
#(disabled and non-functional)
#(disabled and non-functional)   def doInputPhase(nPix,ips):
#(disabled and non-functional)      import kolmogorov
#(disabled and non-functional)      r0,sapxls=ips
#(disabled and non-functional)      print("PHASE_SCREEN",end="...");sys.stdout.flush()
#(disabled and non-functional)      scr=kolmogorov.TwoScreens(
#(disabled and non-functional)            nPix*4,r0,flattening=2*numpy.pi*sapxls**-1.0)[0][:nPix,:nPix]
#(disabled and non-functional)      print("(done)");sys.stdout.flush()
#(disabled and non-functional)      return scr
#(disabled and non-functional)
#(disabled and non-functional)   def _circle(N,fractionalRadius=1):
#(disabled and non-functional)      '''for N pixels, return a 2D array which has a circle (1 within, 0
#(disabled and non-functional)      without) and radius a fraction of N.
#(disabled and non-functional)      '''
#(disabled and non-functional)      return numpy.add.outer(
#(disabled and non-functional)            cds(N)**2,cds(N)**2 )<(fractionalRadius**2)
#(disabled and non-functional)
#(disabled and non-functional)   def _rebin(ip,N):
#(disabled and non-functional)      '''take the square 2D input and the number of sub-apertures and 
#(disabled and non-functional)      return a 2D output which is binned over the sub-aperture elements
#(disabled and non-functional)      and centred.
#(disabled and non-functional)      '''
#(disabled and non-functional)      nPix=ip.shape[0]
#(disabled and non-functional)      sapxls=int( numpy.ceil(nPix*float(N)**-1.0) )
#(disabled and non-functional)      N_=nPix//sapxls # the guessed number of original sub-apertures
#(disabled and non-functional)      if N_==N:
#(disabled and non-functional)         return ip.reshape(
#(disabled and non-functional)            [N,sapxls,N,sapxls]).swapaxes(1,2).sum(axis=-1).sum(axis=-1)
#(disabled and non-functional)      else:
#(disabled and non-functional)         newNPix=(N+N_)*sapxls-nPix
#(disabled and non-functional)         assert newNPix%1==0,"newNPix isn't bigger by a multiple of sapxls"
#(disabled and non-functional)         newNPix=int(newNPix)
#(disabled and non-functional)         dnp=newNPix-nPix
#(disabled and non-functional)         nx=numpy.zeros([newNPix]*2,ip.dtype)
#(disabled and non-functional)         nx[ dnp//2:-dnp//2, dnp//2:-dnp//2 ]=ip
#(disabled and non-functional)         return nx.reshape([N,sapxls]*2).swapaxes(1,2).sum(axis=-1).sum(axis=-1)
#(disabled and non-functional)
#(disabled and non-functional)
#(disabled and non-functional)   ##
#(disabled and non-functional)   ## ===============================================
#(disabled and non-functional)   ## CODE LOGIC AND CONFIGURATION 
#(disabled and non-functional)   ## ===============================================
#(disabled and non-functional)   ##
#(disabled and non-functional)
#(disabled and non-functional)   def checkMain():
#(disabled and non-functional)      def doComparison(scr,scrslopes,igM,nrM,pupilAp,nPix,N,\
#(disabled and non-functional)            sapxls,gO,reconScalingFactor,quiet=0):
#(disabled and non-functional)            # \/ reconstruct from the slopes
#(disabled and non-functional)         inv_scrV=numpy.dot(igM,scrslopes)
#(disabled and non-functional)            # \/ reconstruct from the noise reduced slopes
#(disabled and non-functional)         inv_scr_nrV=numpy.dot(igM,nrM.dot(scrslopes))
#(disabled and non-functional)         delta_inv_scr_nrV=inv_scrV-inv_scr_nrV # difference
#(disabled and non-functional)
#(disabled and non-functional)         # now make visual comparisons
#(disabled and non-functional)         wfGridMask=_rebin(pupilAp,N+1)!=sapxls**2
#(disabled and non-functional)         orig_scr=numpy.ma.masked_array(
#(disabled and non-functional)               _rebin(pupilAp*scr,N+1), wfGridMask )
#(disabled and non-functional)         inv_scr=numpy.ma.masked_array(
#(disabled and non-functional)               numpy.empty(gO.n_,numpy.float64), wfGridMask )
#(disabled and non-functional)         inv_scr.ravel()[gO.illuminatedCornersIdx]=inv_scrV*reconScalingFactor
#(disabled and non-functional)         inv_scr_nr=numpy.ma.masked_array(
#(disabled and non-functional)               numpy.empty(gO.n_,numpy.float64), wfGridMask )
#(disabled and non-functional)         inv_scr_nr.ravel()[gO.illuminatedCornersIdx]=\
#(disabled and non-functional)               inv_scr_nrV*reconScalingFactor
#(disabled and non-functional)         delta_inv_scr_nr=numpy.ma.masked_array(
#(disabled and non-functional)               numpy.empty(gO.n_,numpy.float64), wfGridMask )
#(disabled and non-functional)         delta_inv_scr_nr.ravel()[gO.illuminatedCornersIdx]=\
#(disabled and non-functional)               delta_inv_scr_nrV*reconScalingFactor
#(disabled and non-functional)         for ip in inv_scr,orig_scr,inv_scr_nr:
#(disabled and non-functional)            ip-=ip.mean()
#(disabled and non-functional)         if quiet:
#(disabled and non-functional)            return (inv_scr,orig_scr,inv_scr_nr)
#(disabled and non-functional)         else:
#(disabled and non-functional)            print("COMPARISON:")
#(disabled and non-functional)            for ip,opStem in (
#(disabled and non-functional)                     (scrslopes.var(),"var{scrslopes}"),
#(disabled and non-functional)                     (loopIntM.dot(scrslopes).var()/16.0,
#(disabled and non-functional)                        "1/16 loopIntM{scrslopes}"),
#(disabled and non-functional)                     (orig_scr.var(),"var{orig}"),
#(disabled and non-functional)                     (inv_scr.var(),"var{inv}"),
#(disabled and non-functional)                     ((inv_scr-orig_scr).var()/orig_scr.var(),
#(disabled and non-functional)                        "var{inv-orig}/var{orig}"),
#(disabled and non-functional)                     ((inv_scr_nr-orig_scr).var()/orig_scr.var(),
#(disabled and non-functional)                        "var{NR{inv}-orig}/var{orig}"),
#(disabled and non-functional)                     (delta_inv_scr_nr.var()/orig_scr.var(),
#(disabled and non-functional)                        "var{NR{inv}-inv}/var{orig}"),
#(disabled and non-functional)                  ):
#(disabled and non-functional)               print("{0:s}{1:s} = {2:7.5f}".format( " "*(30-len(opStem)),
#(disabled and non-functional)                     opStem,ip)
#(disabled and non-functional)                  )
#(disabled and non-functional)      
#(disabled and non-functional)      def doComparisonWithNoise(refimgs,N,shscrimgs,cntr,maskI,refslopes,
#(disabled and non-functional)            slopeScaling,igM,nrM,scr,nPix,sapxls,gO,reconScalingFactor,
#(disabled and non-functional)            noiseSDscaling=1,nLoops=100,quiet=0):
#(disabled and non-functional)         if not quiet:
#(disabled and non-functional)            print("NOISE ADDED: SD scaling={0:f}".format(noiseSDscaling))
#(disabled and non-functional)         noiseSD=(1e-9 if noiseSDscaling<1e-9 else noiseSDscaling)\
#(disabled and non-functional)               *refimgs.reshape([N,N,-1]).sum(axis=-1).max()*N**-2.0
#(disabled and non-functional)         nLoops=100
#(disabled and non-functional)         keys='inv','inv_nr','loopInt','inv-orig'
#(disabled and non-functional)         variances={}
#(disabled and non-functional)         for k in keys:
#(disabled and non-functional)            variances[k]=[]
#(disabled and non-functional)         for i in range(nLoops):
#(disabled and non-functional)            
#(disabled and non-functional)            shscrimgsNoisy=numpy.random.normal(shscrimgs,noiseSD)
#(disabled and non-functional)            scrslopesNoisy=getSlopes(shscrimgsNoisy,cntr,maskI,refslopes,slopeScaling)
#(disabled and non-functional)   ##(redundant)         print("**",noiseSD,numpy.var(scrslopesNoisy))
#(disabled and non-functional)            (inv_scr,orig_scr,inv_scr_nr)=doComparison(
#(disabled and non-functional)                  scr,scrslopesNoisy,igM,nrM,pupilAp,nPix,N,
#(disabled and non-functional)                  sapxls,gO,reconScalingFactor,quiet=1)
#(disabled and non-functional)            variances['inv'].append( inv_scr.var() )
#(disabled and non-functional)            variances['inv-orig'].append( (inv_scr-orig_scr).var() )
#(disabled and non-functional)            variances['inv_nr'].append( inv_scr_nr.var() )
#(disabled and non-functional)            variances['loopInt'].append( loopIntM.dot(scrslopesNoisy).var()/16.0 )
#(disabled and non-functional)         #
#(disabled and non-functional)   #(redundant)      opStem="var{orig}" # always the same
#(disabled and non-functional)   #(redundant)      print("{1:s}{2:s}={0:5.3f}".format(
#(disabled and non-functional)   #(redundant)            orig_scr.var(), " "*(20-len(opStem)), opStem )
#(disabled and non-functional)   #(redundant)         ) 
#(disabled and non-functional)         if not quiet:
#(disabled and non-functional)            for k in variances.keys():
#(disabled and non-functional)               opStem="var{{{0:s}}}".format(k)
#(disabled and non-functional)               print("{2:s}{3:s}={0:5.3f}+/-{1:5.3f}".format(
#(disabled and non-functional)                     numpy.mean(variances[k]), numpy.var(variances[k])**0.5,
#(disabled and non-functional)                     " "*(20-len(opStem)), opStem )
#(disabled and non-functional)                  )
#(disabled and non-functional)         return ( variances, shscrimgsNoisy, orig_scr.var() )
#(disabled and non-functional)         
#(disabled and non-functional)      def doPlotting():
#(disabled and non-functional)            # \/ reconstruct from the slopes
#(disabled and non-functional)         inv_scrV=numpy.dot(igM,scrslopes)
#(disabled and non-functional)
#(disabled and non-functional)         # now make visual comparisons
#(disabled and non-functional)         wfGridMask=rebin(pupilAp,N+1)!=sapxls**2
#(disabled and non-functional)         orig_scr=numpy.ma.masked_array(
#(disabled and non-functional)               rebin(pupilAp*scr,N+1), wfGridMask )
#(disabled and non-functional)         inv_scr=numpy.ma.masked_array(
#(disabled and non-functional)               numpy.empty(gO.n_,numpy.float64), wfGridMask )
#(disabled and non-functional)         inv_scr.ravel()[gO.illuminatedCornersIdx]=inv_scrV*reconScalingFactor
#(disabled and non-functional)         pyplot.figure(99)
#(disabled and non-functional)         for i,(thisImg,title) in enumerate([
#(disabled and non-functional)                  (refimgs,"reference"),
#(disabled and non-functional)                  (tilt1ximgs,"tilt1x"),
#(disabled and non-functional)                  (tilt2ximgs,"tilt2x"),
#(disabled and non-functional)                  (shscrimgs,"scr"),
#(disabled and non-functional)               ]):
#(disabled and non-functional)            pyplot.subplot(2,2,1+i)
#(disabled and non-functional)            pyplot.imshow( abs(thisImg.swapaxes(1,2).reshape([nPix]*2))**0.5,
#(disabled and non-functional)                  cmap='cubehelix', vmax=(sapxls)**2.0)
#(disabled and non-functional)            pyplot.title("sqrt{{{0:s}}}".format(title))
#(disabled and non-functional)
#(disabled and non-functional)         pyplot.figure(98)
#(disabled and non-functional)         for ip in inv_scr,orig_scr:
#(disabled and non-functional)            ip-=ip.mean()
#(disabled and non-functional)
#(disabled and non-functional)         pyplot.subplot(2,1,1)
#(disabled and non-functional)         pyplot.plot( orig_scr.ravel(), label="orig_scr")
#(disabled and non-functional)         pyplot.plot( inv_scr.ravel(), label="inv_scr")
#(disabled and non-functional)         pyplot.legend(loc=0)
#(disabled and non-functional)         pyplot.subplot(2,2,3)
#(disabled and non-functional)         pyplot.imshow( orig_scr, vmin=-orig_scr.ptp()/2, vmax=orig_scr.ptp()/2 )
#(disabled and non-functional)         pyplot.title( "orig_scr" )
#(disabled and non-functional)         pyplot.subplot(2,2,4)
#(disabled and non-functional)         pyplot.imshow( inv_scr, vmin=-orig_scr.ptp()/2, vmax=orig_scr.ptp()/2 )
#(disabled and non-functional)         pyplot.title( "inv_scr" )
#(disabled and non-functional)
#(disabled and non-functional)      def doSetup():
#(disabled and non-functional)         global mask
#(disabled and non-functional)         print("SETUP",end="...");sys.stdout.flush()
#(disabled and non-functional)         sapxls=nPix//N
#(disabled and non-functional)         if 'pupilAp' not in dir():
#(disabled and non-functional)            pupilAp=circle(nPix,1) # pupil aperture
#(disabled and non-functional)         cntr=makeCntrArr(sapxls)
#(disabled and non-functional)         mask=rebin(pupilAp,N)*(sapxls**-2.0)
#(disabled and non-functional)         maskB=(mask>=illuminationFraction)
#(disabled and non-functional)         numSAs=maskB.sum()
#(disabled and non-functional)         maskI=(maskB.ravel()).nonzero()[0]
#(disabled and non-functional)         print("(done)");sys.stdout.flush()
#(disabled and non-functional)         return sapxls, cntr, maskB, numSAs, maskI
#(disabled and non-functional)
#(disabled and non-functional)      ## -- begin variables --
#(disabled and non-functional)      ##
#(disabled and non-functional)      defWls=[0,]#range(-4,4+2,4)
#(disabled and non-functional)      N=20# how many sub-apertures
#(disabled and non-functional)      nPix=N*8# total size of pupil as a diameter
#(disabled and non-functional)      numpy.random.seed(18071977)
#(disabled and non-functional)      r0=(nPix/N) # pixels
#(disabled and non-functional)      mag=2
#(disabled and non-functional)      illuminationFraction=0.1
#(disabled and non-functional)      pupilAp=circle(nPix,1) # pupil aperture
#(disabled and non-functional)      binning=1
#(disabled and non-functional)      LT=0 # lazy truncation
#(disabled and non-functional)      GP=1 # guardPixels
#(disabled and non-functional)      noiseSDscalings=[0,]#numpy.arange(5,30,2)#(1e-9,1.0,10.0,20.0,30.0,33.0,38.0,40.0)
#(disabled and non-functional)      radialExtension=0.3 # if >0, do radial extension of spots
#(disabled and non-functional)      #pupilAp-=circle(nPix,nPix//2-(nPix//N*4))
#(disabled and non-functional)      ##
#(disabled and non-functional)      ## -- end variables ----
#(disabled and non-functional)      assert (nPix/N)%2==0, "Require even number of pixels"
#(disabled and non-functional)      print("sapxls =\t",end="")
#(disabled and non-functional)      sapxls=nPix//N
#(disabled and non-functional)      if sapxls==2:
#(disabled and non-functional)         print("Quadcells")
#(disabled and non-functional)      else:
#(disabled and non-functional)         print(sapxls)
#(disabled and non-functional)      print("Lazy truncation" if LT else "Proper truncation")
#(disabled and non-functional)      print(("No" if not GP else "{0:d}".format(GP))+
#(disabled and non-functional)            " guard pixel"+("s" if GP>1 else ""))
#(disabled and non-functional)      print(("No" if not radialExtension else "{0:5.3f}x".format(radialExtension))+
#(disabled and non-functional)            " radial extension of spots")
#(disabled and non-functional)#(redundant)   assert mag>1, "Magnifcation>=2"
#(disabled and non-functional)#(redundant)   assert not mag%2, "Magnification=2N for N in I+"
#(disabled and non-functional)         # \/ setup the basic parameters for the SH array
#(disabled and non-functional)      sapxls, cntr, maskB, numSAs, maskI=doSetup()
#(disabled and non-functional)         # \/ calibrate the SH model
#(disabled and non-functional)      ( slopeScaling, reconScalingFactor, tiltFac, refslopes )=\
#(disabled and non-functional)            calibrateSHmodel( pupilAp, cntr, nPix, N, mag, maskI,
#(disabled and non-functional)                  defWls, binning, LT, GP, radialExtension 
#(disabled and non-functional)               )
#(disabled and non-functional)         # \/ create the gradient operator and the reconstructor
#(disabled and non-functional)      gO,gM,igM=doGOpAndRMX(maskB)
#(disabled and non-functional)      assert gO.numberSubaps==numSAs
#(disabled and non-functional)         # \/ loop integration operator and noise reducer
#(disabled and non-functional)      loopIntM, nrM=doLoopOpAndNRMX(maskB)
#(disabled and non-functional)         # \/ make input
#(disabled and non-functional)      scr=doInputPhase(nPix,(r0,sapxls))
#(disabled and non-functional)         # \/ create the SH images and process them
#(disabled and non-functional)      shscrimgs=makeSHImgs( pupilAp,scr,N,1,defWls,
#(disabled and non-functional)            mag,binning,LT,GP,radialExtension)
#(disabled and non-functional)         # \/ get noiseless slopes 
#(disabled and non-functional)      scrslopes=getSlopes(shscrimgs,cntr,maskI,refslopes,slopeScaling)
#(disabled and non-functional)      assert scrslopes.mask.sum()<len(scrslopes), "All scrslopes are invalid!"
#(disabled and non-functional)      if scrslopes.mask.var()>0: "WARNING: Some scrslopes are invalid!"
#(disabled and non-functional)         # \/ do noiseless comparison
#(disabled and non-functional)      doComparison(scr,scrslopes,igM,nrM,pupilAp,nPix,N,
#(disabled and non-functional)            sapxls,gO,reconScalingFactor,quiet=0)
#(disabled and non-functional)         # \/ do noisy comparison (normal noise on images)
#(disabled and non-functional)      data=[]
#(disabled and non-functional)      for noiseSDscaling in noiseSDscalings:
#(disabled and non-functional)         variances,lastImg,ipVar=doComparisonWithNoise(
#(disabled and non-functional)               refimgs,N,shscrimgs,cntr,maskI,refslopes,
#(disabled and non-functional)               slopeScaling,igM,nrM,scr,nPix,sapxls,gO,reconScalingFactor,
#(disabled and non-functional)               noiseSDscaling,nLoops=100)
#(disabled and non-functional)         sys.stdout.flush()
#(disabled and non-functional)         data.append((noiseSDscaling,variances,lastImg))
#(disabled and non-functional)      pyplot.figure(97)
#(disabled and non-functional)      plotData=[]
#(disabled and non-functional)      for j,i in enumerate(data):
#(disabled and non-functional)         pyplot.subplot(4,int(numpy.ceil(len(noiseSDscalings)/4.0)),j+1)
#(disabled and non-functional)         pyplot.imshow( data[j][-1].swapaxes(1,2).reshape([nPix]*2),
#(disabled and non-functional)               cmap='cubehelix' )
#(disabled and non-functional)         for ax in pyplot.gca().get_xaxis(),pyplot.gca().get_yaxis():
#(disabled and non-functional)            ax.set_visible(0)
#(disabled and non-functional)         pyplot.title( "{0:g}".format( data[j][0] ),size=10 )
#(disabled and non-functional)         plotData.append([
#(disabled and non-functional)               j, data[j][0]]+
#(disabled and non-functional)               map(lambda k :
#(disabled and non-functional)                     [numpy.mean(data[j][1][k]),numpy.var(data[j][1][k])], 
#(disabled and non-functional)                     ('loopInt','inv','inv-orig') )
#(disabled and non-functional)            )
#(disabled and non-functional)      plotData=numpy.array([ x[:2]+x[2]+x[3]+x[4] for x in plotData ])
#(disabled and non-functional)      pyplot.figure(96)
#(disabled and non-functional)      pyplot.subplot(2,1,1)
#(disabled and non-functional)      pyplot.errorbar( plotData[:,1], plotData[:,2] )#, yerr=plotData[:,3] )
#(disabled and non-functional)      pyplot.title("loopIntM with noise")
#(disabled and non-functional)      pyplot.subplot(2,1,2)
#(disabled and non-functional)      pyplot.errorbar( plotData[:,1],
#(disabled and non-functional)            plotData[:,4], label='<var{inv}>' )
#(disabled and non-functional)      pyplot.errorbar( plotData[:,1],
#(disabled and non-functional)            abs(plotData[:,4]-ipVar), label='|<var{inv}>-var{orig}|' )
#(disabled and non-functional)      pyplot.semilogy( plotData[:,1],
#(disabled and non-functional)            plotData[:,6], label='<var{inv-orig}>' )
#(disabled and non-functional)      pyplot.legend(loc=0)
#(disabled and non-functional)      pyplot.title("inv with noise")
#(disabled and non-functional)         # \/ do some plotting to finish up
#(disabled and non-functional)      doPlotting()
#(disabled and non-functional)
#(disabled and non-functional)      ## now test the shifting via lazyTruncation
#(disabled and non-functional)      ##
#(disabled and non-functional)   
#(disabled and non-functional)   def checkLazyTruncate():
#(disabled and non-functional)      aperture=circle(128,0.75)
#(disabled and non-functional)      tiltphs=makeTiltPhase(128,10.0)
#(disabled and non-functional)      imgs={'LT':{'tilt':[],'noTilt':[]},'noLT':{'tilt':[],'noTilt':[]}}
#(disabled and non-functional)      for lt in (0,1):
#(disabled and non-functional)         for tilt in (0,1):
#(disabled and non-functional)            imgs['LT' if lt else 'noLT']['tilt' if tilt else 'noTilt'].append(
#(disabled and non-functional)                  makeSHImgs(aperture,tiltphs*tilt,16,mag=4,lazyTruncate=lt)
#(disabled and non-functional)               )
#(disabled and non-functional)      pyplot.figure(80)
#(disabled and non-functional)      plotNo=1
#(disabled and non-functional)      for lt in (0,1):
#(disabled and non-functional)         for tilt in (0,1):
#(disabled and non-functional)            pyplot.subplot(2,2,plotNo)
#(disabled and non-functional)            codes='LT' if lt else 'noLT', 'tilt' if tilt else 'noTilt'
#(disabled and non-functional)            thisImg=imgs[codes[0]][codes[1]][0].swapaxes(1,2).reshape([128]*2)
#(disabled and non-functional)            if lt==0 and tilt==0:
#(disabled and non-functional)               refImgSum=float(thisImg.sum())
#(disabled and non-functional)            pyplot.imshow( thisImg**0.5, cmap='gray' )
#(disabled and non-functional)            fractionKept=thisImg.sum()/refImgSum
#(disabled and non-functional)            opStr="{0[0]:s}/{0[1]:s}->{1:4.2f}".format(
#(disabled and non-functional)                  codes, fractionKept)
#(disabled and non-functional)            imgs[codes[0]][codes[1]].append( fractionKept )
#(disabled and non-functional)            print("\t"+opStr);pyplot.title(opStr)
#(disabled and non-functional)            plotNo+=1
#(disabled and non-functional)            if lt==0:
#(disabled and non-functional)               assert fractionKept>0.95, "for no lazy truncation, lost too much"
#(disabled and non-functional)            if lt==0 and tilt==1:
#(disabled and non-functional)               assert fractionKept>0.95*imgs['noLT']['noTilt'][1],\
#(disabled and non-functional)                     "for no lazy truncation and tilt, lost too much"
#(disabled and non-functional)            if lt==1:
#(disabled and non-functional)               assert fractionKept<=1, "lazy truncation has too much intensity?"
#(disabled and non-functional)            if lt==1 and tilt==1:
#(disabled and non-functional)               assert fractionKept<imgs['LT']['noTilt'][1],\
#(disabled and non-functional)                     "lazy truncation not losing intensity"
#(disabled and non-functional)
#(disabled and non-functional)      return imgs
#(disabled and non-functional)
#(disabled and non-functional)   from matplotlib import pyplot
#(disabled and non-functional)   print("BEGINS: test code for fourierSHexamples")
#(disabled and non-functional)   import types
#(disabled and non-functional)   for objectName in dir():
#(disabled and non-functional)      print("("+objectName[:len('check')]+")",end="")
#(disabled and non-functional)      if 'check'==objectName[:len('check')]\
#(disabled and non-functional)            and type(eval(objectName))==types.FunctionType:
#(disabled and non-functional)         print("\n\tFound checking function: '{0:s}', "\
#(disabled and non-functional)               "will call without parameters".format( objectName ), end="\n"*2
#(disabled and non-functional)            )
#(disabled and non-functional)         try:
#(disabled and non-functional)            eval(objectName+"()")
#(disabled and non-functional)         except:
#(disabled and non-functional)            print("ERROR:{0:s} FAILED".format(objectName))
#(disabled and non-functional)            print(sys.exc_info()) 
#(disabled and non-functional)         else:
#(disabled and non-functional)            print("PASSED:{0:s}".format(objectName))
#(disabled and non-functional)   print("ENDS")
