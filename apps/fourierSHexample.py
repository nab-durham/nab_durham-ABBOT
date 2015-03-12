"""ABBOT : useful functions to make realistic SH patterns.
Note that the radial extension feature is not realistic for LGS
spot extension since it doesn't act for incoherent sources.
"""

from __future__ import print_function
import numpy
import sys

def cds(N, roll=False):
   tcds = (numpy.arange(0,N)-(N/2.))*(N/2.0)**-1.0
   return tcds if not roll else numpy.fft.fftshift(tcds)

def circle(N,fractionalRadius=1):
   return numpy.add.outer(
         cds(N)**2,cds(N)**2 )<(fractionalRadius**2)

def makeTiltPhase(nPix, fac):
   return -fac*numpy.pi/nPix*numpy.add.outer(
         numpy.arange(nPix),
         numpy.arange(nPix)
      )

def rebin(ip,N):
   nPix=ip.shape[0]
   sapxls=int( numpy.ceil(nPix*float(N)**-1.0) )
   N_=nPix//sapxls # the guessed number of original sub-apertures
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

def makeSHImgs( aperture, phs, N, amplitudeScaling=1, wl=[0,], mag=2, 
         binning=1, lazyTruncate=1, guardPixels=1, radialExtension=0 ):
   '''aperture : overall aperture
      amplitudeScaling : an additional factor to aperture, could be constant
      phs : 2D array of phase
      N : number of sub-apertures
      wl : simulate, crudely, polychromatism via the 'l' variable
      mag : magnification of each spot
      binning : how much to bin each spot before assinging
      lazyTruncate : whether to assume each spot can't/won't move into it's
         neighbours region (don't preserve intensity, faster), or to do it
         properly (preserve intensity except at aperture edges, slower)
      guardPixels : how many (binned) pixels are ignored along the upper
         right edge of the simulated image, to model a guard-band.
      radialExtension : if >0, reduce spatial coherence to simulate radial
         elongation
         

      Note that binning is first used as a magnification factor and then
      as a factor to bin pixel by.
      This permits finer granulation of magnification.

      Note that guardPixels must be zero if using lazyTruncate; it is 
      sort-of implicit with lazyTruncate depending on the magnification
      chosen, although the method is rather different.
   '''
   sapxls=map(lambda s : s*N**-1.0, aperture.shape)
   assert len(sapxls)==2 and sapxls[0]==sapxls[1], "Require a square, for now"
   sapxls=int(sapxls[0])
   assert type(phs) in (int,float) or ( len(phs)==len(aperture) ),\
         "phs should be a constant or same shape as aperture"
   nPix=aperture.shape[0]
   cos,sin=numpy.cos,numpy.sin
   sapxlsFP=sapxls*binning*mag # the number of pixels to use for the imaging
   if radialExtension>0:
      rotatedCoordinate = lambda ang : numpy.add.outer(
            cos(ang)*cds(sapxls),sin(ang)*cds(sapxls )
         )
      reduceSpatialCoherence = lambda ang,amp : amp*rotatedCoordinate(ang)**2.0
      angles = numpy.arctan2( numpy.add.outer(numpy.zeros(N), cds(N)),
                              numpy.add.outer(cds(N), numpy.zeros(N))
         )
      spatialCoherenceReduction=numpy.empty( [N,N,sapxls,sapxls], numpy.complex128 )
      for j in range(N):
         for i in range(N):
            radius = ( (i-N/2.+0.5)**2 + (j-N/2.0+0.5)**2 )**0.5
            spatialCoherenceReduction[j,i]=numpy.exp(1.0j*
                  reduceSpatialCoherence(angles[j,i],radialExtension*radius)
               )
      spatialCoherenceReduction=spatialCoherenceReduction.swapaxes(1,2).reshape([N*sapxls]*2)
   else:
      spatialCoherenceReduction=1 # i.e. don't reduce spatial coherence anywhere
   if ((sapxls*binning)*mag)%1!=0:
      suggested_mags=int( (sapxls*binning)*mag )/(sapxls*binning)**-1.0,\
                     (int( (sapxls*binning)*mag )+1)/(sapxls*binning)**-1.0
      raise ValueError("Magnification {0:f} is too fine-grained, "\
            "suggest using either {1[0]:f} or {1[1]:f}".format(
                  mag, suggested_mags))

   if sapxlsFP-max(wl)<1:
      raise ValueError("wl has too large a value, must be < {0:d}".format(
            sapxlsFP-1))
   if sapxls*mag-max(wl)//binning<0:
      raise ValueError("wl has too large a value, must be < {0:d}".format(
            sapxls*mag*binning))
   for twl in wl:
      if (sapxlsFP-twl)%binning:
         raise ValueError("wl {0:d} doesn't work with binning:"\
               "({1:d}-{2:d})%{3:d}!=0".format(twl,sapxlsFP,twl,binning))
   FFTCentringPhase=lambda twl : makeTiltPhase(nPix,-nPix*(sapxlsFP-twl)**-1.0)
   A=aperture
   C=amplitudeScaling*spatialCoherenceReduction
   polyChromSHImgs=[]
   for twl in wl:
      tip=(A*C*numpy.exp(
            -1.0j*( phs+FFTCentringPhase(twl) ) 
          )).reshape([N,sapxls,N,sapxls]).swapaxes(1,2)
      top=abs( numpy.fft.fftshift(
            numpy.fft.fft2(tip,s=[sapxlsFP+twl]*2
         ), (2,3) ) )**2.0 # no requirement to preserve phase therefore only one fftshift
      top=top.reshape([N,N]+
               [(sapxlsFP+twl)//binning,binning,(sapxlsFP+twl)//binning,binning]
            ).sum(axis=-3).sum(axis=-1)
      polyChromSHImgs.append( top ) # save everything, for now
  
   # n.b. binning has been done at this stage
   if lazyTruncate:
      # algorithm: truncate each image to the size of the sub-aperture
      #  and compute an average
      for i,twl in enumerate(wl):
         idx=((sapxls*mag)/2+twl//binning/2) + numpy.arange(-sapxls/2,sapxls/2)
         polyChromSHImgs[i]=polyChromSHImgs[i].take(idx,axis=2).take(idx,axis=3)
      return numpy.mean( polyChromSHImgs, axis=0 )
   else:
      # algorithm: make a canvas to paint SH sub-aperture images on
      #  and then place for each sa, the result for each wl made to the same
      #  size.
      # if using guardPixels, then make a bigger canvas and then take
      #  out the unecessary columns
      canvas=numpy.zeros(
            [len(wl)]+[nPix+sapxls*mag+max(wl)//binning+guardPixels*N]*2,
            top.dtype)
      # TODO: guardPixels implementation
#(redundant)      print(">>>:cs,N,nPix,sapxls,mag="+str(
#(redundant)            [canvas.shape, N, nPix, sapxls,mag]))
      for i,twl in enumerate(wl):
         width=sapxls*mag+twl//binning
         for l in range(N): # vertical
            for m in range(N): # horizontal
               cornerCoords=map( lambda v :
                  (sapxls)*(0.5+v+(max(wl)-twl)//binning//2)+
                  (guardPixels*v), (l,m))
               canvas[i, cornerCoords[0]:cornerCoords[0]+width,
                         cornerCoords[1]:cornerCoords[1]+width
                  ]+=polyChromSHImgs[i][l,m]
#(redundant)      print("{"+str(canvas.shape),end="")
      offset=(sapxls*mag)/2+max(wl)//binning//2
      # trim and remove any guard-pixels
      idx=numpy.add.outer( numpy.arange(N)*(sapxls+guardPixels),
            offset+numpy.arange(sapxls) ).ravel()
      canvas=canvas.mean(axis=0).take(idx,axis=0).take(idx,axis=1)
#(redundant)      print("}"+str(canvas.shape))
      return canvas.reshape([N,sapxls]*2).swapaxes(1,2)

def makeCntrArr(P):
   cntr=[  numpy.add.outer(
      (numpy.linspace(-1,1-P**-1.0*2.,P)), numpy.zeros(P)) ]
   cntr.append(cntr[0].T)
   return numpy.array(cntr).T.reshape([1,1,P,P,2])

def getSlopes(shimgs,cntr,maskI,refslopes=0,slopeScaling=1):
   rawSlopes=( (cntr*shimgs.reshape(list(shimgs.shape)+[1])
         ).sum(axis=2).sum(axis=2)/
            (1e-10+shimgs.reshape(list(shimgs.shape)+[1])
         ).sum(axis=2).sum(axis=2)
      )
   slopesV=rawSlopes.reshape([-1,2]).take(maskI,axis=0).T.ravel()
   slopesV=numpy.ma.masked_array(slopesV,numpy.isnan(slopesV))
   return (slopesV-refslopes)*slopeScaling

def calibrateSHmodel( pupilAp, cntr, nPix, N, mag, maskI, defWls, binning,
      LT, GP, radialExtension ):
   """Calibrate the SH model by adding artificial tilts that move the spots
      by 1/4 and 1/2 of the sub-aperture width in both directions
      simultaneously.
   """
   global refimgs,tilt1ximgs,tilt2ximgs,tilt1xslopes,tilt2xslopes
   print("CALIBRATION:\n\t",end="")
   sapxls=nPix//N
   #
   tiltFac=0.5*nPix*(sapxls*mag)**-1.0 # move 1/4 sub-aperture
   tiltphs=makeTiltPhase(nPix, tiltFac)
   # now, 1/4 sub-aperture for sapxls means that the signal from tiltphs
   # is the tilt for a slope of:
   #  tiltFac*numpy.pi/nPix
   # =0.5*numpy.pi/(sapxls*mag)
   # or the difference between adjacent wavefront grid points of:
   #  0.5*numpy.pi/mag
   # so scaling upon reconstruction should be:
   reconScalingFactor=0.5*numpy.pi*mag**-1.0
   # now add a negative factor (don't know why)
   reconScalingFactor*=-1
   # and finally the effect of rebinning into sapxls^2
   reconScalingFactor*=sapxls**2.
   print("reconScaling",end=":")
   #
   tilt1ximgs=makeSHImgs(
         pupilAp,tiltphs,  N,1,defWls,mag,binning,LT,GP,radialExtension)
   tilt2ximgs=makeSHImgs(
         pupilAp,tiltphs*2,N,1,defWls,mag,binning,LT,GP,radialExtension)
   print("SH_Images{tilt1x} & {tilt2x}",end=":")
   #
   # Make reference values
   refimgs=makeSHImgs(pupilAp,0,N,1,defWls,mag,binning,LT,GP,radialExtension)
   refslopes=getSlopes(refimgs,cntr,maskI) # raw slopes
   assert refslopes.mask.sum()==0, "All refslopes are invalid!"
   assert refslopes.mask.var()==0, "Some refslopes are invalid!"
   print("Image intensities:")
   for i,s in (refimgs,"ref"),(tilt1ximgs,"tilt_1x"),(tilt2ximgs,"tilt_2x"):
      print("\t{0:s}{1:s} = {2:d}".format(" "*(15-len(s)), s, int(i.sum()) ))
   #
   # get unscaled tilt slopes
   tilt1xslopes=getSlopes(tilt1ximgs,cntr,maskI,refslopes)
   tilt1xslopes=numpy.ma.masked_array(
         tilt1xslopes.data, tilt1xslopes.mask+tilt1xslopes.data<1e-6 )
   assert tilt1xslopes.mask.sum()<len(tilt1xslopes),\
         "All tilt1xslopes are invalid!"
   if tilt1xslopes.mask.var()!=0: "WARNING: Some tilt1xslopes are invalid!"
   tilt2xslopes=getSlopes(tilt2ximgs,cntr,maskI,refslopes)
   tilt2xslopes=numpy.ma.masked_array(
         tilt2xslopes.data, tilt2xslopes.mask+tilt2xslopes.data<1e-6 )
   assert tilt2xslopes.mask.sum()<len(tilt2xslopes),\
         "All tilt2xslopes are invalid!"
   if tilt2xslopes.mask.var()!=0: "WARNING: Some tilt1xslopes are invalid!"
   # now generate slope scaling
   slopeScaling=0.5*(1/(1e-15+tilt1xslopes)+2/(1e-15+tilt2xslopes))
   print("slopeScaling",end="")
   #
   print()
   return ( slopeScaling, reconScalingFactor, tiltFac, refslopes)


if __name__=="__main__":

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
      def doComparison(scr,scrslopes,igM,nrM,pupilAp,nPix,N,\
            sapxls,gO,reconScalingFactor,quiet=0):
            # \/ reconstruct from the slopes
         inv_scrV=numpy.dot(igM,scrslopes)
            # \/ reconstruct from the noise reduced slopes
         inv_scr_nrV=numpy.dot(igM,nrM.dot(scrslopes))
         delta_inv_scr_nrV=inv_scrV-inv_scr_nrV # difference

         # now make visual comparisons
         wfGridMask=rebin(pupilAp,N+1)!=sapxls**2
         orig_scr=numpy.ma.masked_array(
               rebin(pupilAp*scr,N+1), wfGridMask )
         inv_scr=numpy.ma.masked_array(
               numpy.empty(gO.n_,numpy.float64), wfGridMask )
         inv_scr.ravel()[gO.illuminatedCornersIdx]=inv_scrV*reconScalingFactor
         inv_scr_nr=numpy.ma.masked_array(
               numpy.empty(gO.n_,numpy.float64), wfGridMask )
         inv_scr_nr.ravel()[gO.illuminatedCornersIdx]=\
               inv_scr_nrV*reconScalingFactor
         delta_inv_scr_nr=numpy.ma.masked_array(
               numpy.empty(gO.n_,numpy.float64), wfGridMask )
         delta_inv_scr_nr.ravel()[gO.illuminatedCornersIdx]=\
               delta_inv_scr_nrV*reconScalingFactor
         for ip in inv_scr,orig_scr,inv_scr_nr:
            ip-=ip.mean()
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
      
      def doComparisonWithNoise(refimgs,N,shscrimgs,cntr,maskI,refslopes,
            slopeScaling,igM,nrM,scr,nPix,sapxls,gO,reconScalingFactor,
            noiseSDscaling=1,nLoops=100,quiet=0):
         if not quiet:
            print("NOISE ADDED: SD scaling={0:f}".format(noiseSDscaling))
         noiseSD=(1e-9 if noiseSDscaling<1e-9 else noiseSDscaling)\
               *refimgs.reshape([N,N,-1]).sum(axis=-1).max()*N**-2.0
         nLoops=100
         keys='inv','inv_nr','loopInt','inv-orig'
         variances={}
         for k in keys:
            variances[k]=[]
         for i in range(nLoops):
            
            shscrimgsNoisy=numpy.random.normal(shscrimgs,noiseSD)
            scrslopesNoisy=getSlopes(shscrimgsNoisy,cntr,maskI,refslopes,slopeScaling)
   ##(redundant)         print("**",noiseSD,numpy.var(scrslopesNoisy))
            (inv_scr,orig_scr,inv_scr_nr)=doComparison(
                  scr,scrslopesNoisy,igM,nrM,pupilAp,nPix,N,
                  sapxls,gO,reconScalingFactor,quiet=1)
            variances['inv'].append( inv_scr.var() )
            variances['inv-orig'].append( (inv_scr-orig_scr).var() )
            variances['inv_nr'].append( inv_scr_nr.var() )
            variances['loopInt'].append( loopIntM.dot(scrslopesNoisy).var()/16.0 )
         #
   #(redundant)      opStem="var{orig}" # always the same
   #(redundant)      print("{1:s}{2:s}={0:5.3f}".format(
   #(redundant)            orig_scr.var(), " "*(20-len(opStem)), opStem )
   #(redundant)         ) 
         if not quiet:
            for k in variances.keys():
               opStem="var{{{0:s}}}".format(k)
               print("{2:s}{3:s}={0:5.3f}+/-{1:5.3f}".format(
                     numpy.mean(variances[k]), numpy.var(variances[k])**0.5,
                     " "*(20-len(opStem)), opStem )
                  )
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
         inv_scr.ravel()[gO.illuminatedCornersIdx]=inv_scrV*reconScalingFactor
         pyplot.figure(99)
         for i,(thisImg,title) in enumerate([
                  (refimgs,"reference"),
                  (tilt1ximgs,"tilt1x"),
                  (tilt2ximgs,"tilt2x"),
                  (shscrimgs,"scr"),
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
         pyplot.imshow( orig_scr, vmin=-orig_scr.ptp()/2, vmax=orig_scr.ptp()/2 )
         pyplot.title( "orig_scr" )
         pyplot.subplot(2,2,4)
         pyplot.imshow( inv_scr, vmin=-orig_scr.ptp()/2, vmax=orig_scr.ptp()/2 )
         pyplot.title( "inv_scr" )

      def doSetup():
         global mask
         print("SETUP",end="...");sys.stdout.flush()
         sapxls=nPix//N
         if 'pupilAp' not in dir():
            pupilAp=circle(nPix,1) # pupil aperture
         cntr=makeCntrArr(sapxls)
         mask=rebin(pupilAp,N)*(sapxls**-2.0)
         maskB=(mask>=illuminationFraction)
         numSAs=maskB.sum()
         maskI=(maskB.ravel()).nonzero()[0]
         print("(done)");sys.stdout.flush()
         return sapxls, cntr, maskB, numSAs, maskI

      ## -- begin variables --
      ##
      defWls=[0,]#range(-4,4+2,4)
      N=20# how many sub-apertures
      nPix=N*8# total size of pupil as a diameter
      numpy.random.seed(18071977)
      r0=(nPix/N) # pixels
      mag=2
      illuminationFraction=0.1
      pupilAp=circle(nPix,1) # pupil aperture
      binning=1
      LT=0 # lazy truncation
      GP=1 # guardPixels
      noiseSDscalings=[0,]#numpy.arange(5,30,2)#(1e-9,1.0,10.0,20.0,30.0,33.0,38.0,40.0)
      radialExtension=0.3 # if >0, do radial extension of spots
      #pupilAp-=circle(nPix,nPix//2-(nPix//N*4))
      ##
      ## -- end variables ----
      assert (nPix/N)%2==0, "Require even number of pixels"
      print("sapxls =\t",end="")
      sapxls=nPix//N
      if sapxls==2:
         print("Quadcells")
      else:
         print(sapxls)
      print("Lazy truncation" if LT else "Proper truncation")
      print(("No" if not GP else "{0:d}".format(GP))+
            " guard pixel"+("s" if GP>1 else ""))
      print(("No" if not radialExtension else "{0:5.3f}x".format(radialExtension))+
            " radial extension of spots")
#(redundant)   assert mag>1, "Magnifcation>=2"
#(redundant)   assert not mag%2, "Magnification=2N for N in I+"
         # \/ setup the basic parameters for the SH array
      sapxls, cntr, maskB, numSAs, maskI=doSetup()
         # \/ calibrate the SH model
      ( slopeScaling, reconScalingFactor, tiltFac, refslopes )=\
            calibrateSHmodel( pupilAp, cntr, nPix, N, mag, maskI,
                  defWls, binning, LT, GP, radialExtension 
               )
         # \/ create the gradient operator and the reconstructor
      gO,gM,igM=doGOpAndRMX(maskB)
      assert gO.numberSubaps==numSAs
         # \/ loop integration operator and noise reducer
      loopIntM, nrM=doLoopOpAndNRMX(maskB)
         # \/ make input
      scr=doInputPhase(nPix,(r0,sapxls))
         # \/ create the SH images and process them
      shscrimgs=makeSHImgs( pupilAp,scr,N,1,defWls,
            mag,binning,LT,GP,radialExtension)
         # \/ get noiseless slopes 
      scrslopes=getSlopes(shscrimgs,cntr,maskI,refslopes,slopeScaling)
      assert scrslopes.mask.sum()<len(scrslopes), "All scrslopes are invalid!"
      if scrslopes.mask.var()>0: "WARNING: Some scrslopes are invalid!"
         # \/ do noiseless comparison
      doComparison(scr,scrslopes,igM,nrM,pupilAp,nPix,N,
            sapxls,gO,reconScalingFactor,quiet=0)
         # \/ do noisy comparison (normal noise on images)
      data=[]
      for noiseSDscaling in noiseSDscalings:
         variances,lastImg,ipVar=doComparisonWithNoise(
               refimgs,N,shscrimgs,cntr,maskI,refslopes,
               slopeScaling,igM,nrM,scr,nPix,sapxls,gO,reconScalingFactor,
               noiseSDscaling,nLoops=100)
         sys.stdout.flush()
         data.append((noiseSDscaling,variances,lastImg))
      pyplot.figure(97)
      plotData=[]
      for j,i in enumerate(data):
         pyplot.subplot(4,int(numpy.ceil(len(noiseSDscalings)/4.0)),j+1)
         pyplot.imshow( data[j][-1].swapaxes(1,2).reshape([nPix]*2),
               cmap='cubehelix' )
         for ax in pyplot.gca().get_xaxis(),pyplot.gca().get_yaxis():
            ax.set_visible(0)
         pyplot.title( "{0:g}".format( data[j][0] ),size=10 )
         plotData.append([
               j, data[j][0]]+
               map(lambda k :
                     [numpy.mean(data[j][1][k]),numpy.var(data[j][1][k])], 
                     ('loopInt','inv','inv-orig') )
            )
      plotData=numpy.array([ x[:2]+x[2]+x[3]+x[4] for x in plotData ])
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
      ##
   
   def checkLazyTruncate():
      aperture=circle(128,0.75)
      tiltphs=makeTiltPhase(128,10.0)
      imgs={'LT':{'tilt':[],'noTilt':[]},'noLT':{'tilt':[],'noTilt':[]}}
      for lt in (0,1):
         for tilt in (0,1):
            imgs['LT' if lt else 'noLT']['tilt' if tilt else 'noTilt'].append(
                  makeSHImgs(aperture,tiltphs*tilt,16,mag=4,lazyTruncate=lt)
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
