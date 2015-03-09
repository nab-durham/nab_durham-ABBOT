from __future__ import print_function
import abbot.gradientOperator
import abbot.continuity
import kolmogorov
import numpy
import sys
import Zernike

def makeTiltPhase(nPix, fac):
   return -fac*numpy.pi/nPix*numpy.add.outer(
         numpy.arange(nPix),
         numpy.arange(nPix)
      )

def rebin(ip,nPix,N):
   N_=nPix//sapxls
   if nPix%N==0: # nb this code logic is poor, but sufficient here
      assert N_==N, "Inconsistency"
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
         binning=1, lazyTruncate=1, guardPixels=1 ):
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
         

      Note that binning is first used as a magnification factor and then
      as a factor to bin pixel by.
      This permits finer granulation of magnification.

      Note that guardPixels must be zero if using lazyTruncate; it is 
      sort-of implicit with lazyTruncate depending on the magnification
      chosen, although the method is rather different.
   '''
   if ((sapxls*binning)*mag)%1!=0:
      suggested_mags=int( (sapxls*binning)*mag )/(sapxls*binning)**-1.0,\
                     (int( (sapxls*binning)*mag )+1)/(sapxls*binning)**-1.0
      raise ValueError("Magnification {0:f} is too fine-grained, "\
            "suggest using either {1[0]:f} or {1[1]:f}".format(
                  mag, suggested_mags))

   sapxlsFP=sapxls*binning*mag # the number of pixels to use for the imaging
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
   C=amplitudeScaling
   polyChromSHImgs=[]
   for twl in wl:
      tip=(A*C*numpy.exp(
            -1.0j*( phs+FFTCentringPhase(twl) ) 
          )).reshape([N,sapxls,N,sapxls]).swapaxes(1,2)
      top=abs(numpy.fft.fftshift(
            numpy.fft.fft2(tip,s=[sapxlsFP+twl]*2
         ), (2,3) ))**2.0 # explicitly say we're DFTing the last two axes
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
      print(">>>:cs,N,nPix,sapxls,mag="+str(
            [canvas.shape, N, nPix, sapxls,mag]))
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
      print("{"+str(canvas.shape),end="")
      offset=(sapxls*mag)/2+max(wl)//binning//2
      # trim and remove any guard-pixels
      idx=numpy.add.outer( numpy.arange(N)*(sapxls+guardPixels),
            offset+numpy.arange(sapxls) ).ravel()
      canvas=canvas.mean(axis=0).take(idx,axis=0).take(idx,axis=1)
      print("}"+str(canvas.shape))
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

def calibrateSHmodel( cntr, nPix, N, mag ):
   """Calibrate the SH model by adding artificial tilts that move the spots
      by 1/4 and 1/2 of the sub-aperture width in both directions
      simultaneously.
   """
   global refimgs,tilt1ximgs,tilt2ximgs,tilt1xslopes,tilt2xslopes
   print("CALIBRATION:\n\t",end="")
   sapxls=nPix//N
   #
   tiltFac=0.5*nPix*(sapxls*mag)**-1.0 # move 1/4 sub-aperture
   tiltphs1x=makeTiltPhase(nPix, tiltFac)
   tiltphs2x=makeTiltPhase(nPix, tiltFac*2)
   # now, 1/4 sub-aperture for sapxls means that the signal from tiltphs1x
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
   tilt1ximgs=makeSHImgs(pupilAp,tiltphs1x,N,1,defWls,mag,binning,LT,GP)
   tilt2ximgs=makeSHImgs(pupilAp,tiltphs2x,N,1,defWls,mag,binning,LT,GP)
   print("SH_Images{tilt1x} & {tilt2x}",end=":")
   #
   # Make reference values
   refimgs=makeSHImgs(pupilAp,0,N,1,defWls,mag,binning,LT,GP)
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
   
   def doComparison(scr,scrslopes,igM,nrM,pupilAp,nPix,N,\
         sapxls,gO,reconScalingFactor,quiet=0):
         # \/ reconstruct from the slopes
      inv_scrV=numpy.dot(igM,scrslopes)
         # \/ reconstruct from the noise reduced slopes
      inv_scr_nrV=numpy.dot(igM,nrM.dot(scrslopes))
      delta_inv_scr_nrV=inv_scrV-inv_scr_nrV # difference

      # now make visual comparisons
      wfGridMask=rebin(pupilAp,nPix,N+1)!=sapxls**2
      orig_scr=numpy.ma.masked_array(
            rebin(pupilAp*scr,nPix,N+1), wfGridMask )
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
         print("\tscr.var()\t={0:5.3f}".format( scr.var() ))
         print("\tscrslopes.var()\t={0:5.3f}".format( scrslopes.var() ))
         print("\tloopIntM[scrslopes].var()\t={0:5.3f}".format(
               loopIntM.dot(scrslopes).var()/16 ))
         print("\t{{inv_scr-orig_scr}}_v/{{orig_scr}}_v\t={0:9.7f}".format(
               (inv_scr-orig_scr).var()/orig_scr.var() ))
         print("\t{{inv_scr_nr-orig_scr}}_v/{{orig_scr}}_v\t={0:9.7f}".format(
               (inv_scr_nr-orig_scr).var()/orig_scr.var() ))
         print("\t{{delta_inv_nr}}_v/{{orig_scr}}_v\t={0:9.7G}".format(
               (delta_inv_scr_nr).var()/orig_scr.var() ))
      
   def doPlotting():
         # \/ reconstruct from the slopes
      inv_scrV=numpy.dot(igM,scrslopes)

      # now make visual comparisons
      wfGridMask=rebin(pupilAp,nPix,N+1)!=sapxls**2
      orig_scr=numpy.ma.masked_array(
            rebin(pupilAp*scr,nPix,N+1), wfGridMask )
      inv_scr=numpy.ma.masked_array(
            numpy.empty(gO.n_,numpy.float64), wfGridMask )
      inv_scr.ravel()[gO.illuminatedCornersIdx]=inv_scrV*reconScalingFactor
      pylab.figure(99)
      for i,(thisImg,title) in enumerate([
               (refimgs,"reference"),
               (tilt1ximgs,"tilt1x"),
               (tilt2ximgs,"tilt2x"),
               (shscrimgs,"scr"),
            ]):
         pylab.subplot(2,2,1+i)
         pylab.imshow( abs(thisImg.swapaxes(1,2).reshape([nPix]*2))**0.5 )
         pylab.title("log10{{{0:s}}}".format(title))

      pylab.figure(98)
      for ip in inv_scr,orig_scr:
         ip-=ip.mean()

      pylab.subplot(2,1,1)
      pylab.plot( orig_scr.ravel(), label="orig_scr")
      pylab.plot( inv_scr.ravel(), label="inv_scr")
      pylab.legend(loc=0)
      pylab.subplot(2,2,3)
      pylab.imshow( orig_scr, vmin=-orig_scr.ptp()/2, vmax=orig_scr.ptp()/2 )
      pylab.title( "orig_scr" )
      pylab.subplot(2,2,4)
      pylab.imshow( inv_scr, vmin=-orig_scr.ptp()/2, vmax=orig_scr.ptp()/2 )
      pylab.title( "inv_scr" )

   def doSetup():
      global mask
      print("SETUP",end="...");sys.stdout.flush()
      sapxls=nPix//N
      if 'pupilAp' not in dir():
         pupilAp=Zernike.anyZernike(1,nPix,nPix//2) # pupil aperture
      cntr=makeCntrArr(sapxls)
      mask=rebin(pupilAp,nPix,N)*(sapxls**-2.0)
      maskB=(mask>=illuminationFraction)
      numSAs=maskB.sum()
      maskI=(maskB.ravel()).nonzero()[0]
      print("(done)");sys.stdout.flush()
      return sapxls, cntr, maskB, numSAs, maskI

   def doGOpAndRMX(subapMask):
      print("GRADIENT OP & RECON MX",end="...");sys.stdout.flush()
      gO=abbot.gradientOperator.gradientOperatorType1( subapMask )
      print("^^^",end="");sys.stdout.flush()
      gM=gO.returnOp()
      print("...",end="");sys.stdout.flush()
      igM=numpy.linalg.inv(
            gM.T.dot(gM)+1e-3*numpy.identity(gO.numberPhases)).dot(gM.T)
      ##numpy.linalg.pinv(gM,1e-3)
      print("(done)");sys.stdout.flush()
      #
      return gO,gM,igM

   def doLoopOpAndNRMX(subapMask): 
      print("LOOPS_DEFN & NOISE_REDUC.",end="...");sys.stdout.flush()
      loopsDef=abbot.continuity.loopsIntegrationMatrix( subapMask )
      nrDef=abbot.continuity.loopsNoiseMatrices( subapMask )
      loopIntM=loopsDef.returnOp()
      neM,noiseReductionM=nrDef.returnOp()
      print("(done)");sys.stdout.flush()
      #
      return loopIntM, noiseReductionM

   def doInputPhase(nPix,ips):
      r0,sapxls=ips
      print("PHASE_SCREEN",end="...");sys.stdout.flush()
      scr=kolmogorov.TwoScreens(
            nPix*4,r0,flattening=2*numpy.pi*sapxls**-1.0)[0][:nPix,:nPix]
      print("(done)");sys.stdout.flush()
      return scr

   ##
   ##
   ## ===============================================
   ## CODE LOGIC AND CONFIGURATION 
   ## ===============================================
   ##
   ## -- begin variables --
   ##
   defWls=[0,]#range(-4,4+2,4)
   N=20# how many sub-apertures
   nPix=N*4# total size of pupil as a diameter
   numpy.random.seed(18071977)
   r0=4 # pixels
   mag=2
   illuminationFraction=0.1
   pupilAp=Zernike.anyZernike(1,nPix,nPix//2) # pupil aperture
   binning=1
   LT=0 # lazy truncation
   GP=2 # guardPixels
   #pupilAp-=Zernike.anyZernike(1,nPix,nPix//2-(nPix//N*4))
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
##   assert mag>1, "Magnifcation>=2"
##   assert not mag%2, "Magnification=2N for N in I+"
      # \/ setup the basic parameters for the SH array
   sapxls, cntr, maskB, numSAs, maskI=doSetup()
      # \/ calibrate the SH model
   ( slopeScaling, reconScalingFactor, tiltFac, refslopes )=\
         calibrateSHmodel( cntr, nPix, N, mag )
      # \/ create the gradient operator and the reconstructor
   gO,gM,igM=doGOpAndRMX(maskB)
   assert gO.numberSubaps==numSAs
      # \/ loop integration operator and noise reducer
   loopIntM, nrM=doLoopOpAndNRMX(maskB)
      # \/ make input
   scr=doInputPhase(nPix,(r0,sapxls))
      # \/ create the SH images and process them
   shscrimgs=makeSHImgs(pupilAp,scr,N,1,defWls,mag,binning,LT,GP)
      # \/ get noiseless slopes 
   scrslopes=getSlopes(shscrimgs,cntr,maskI,refslopes,slopeScaling)
   assert scrslopes.mask.sum()<len(scrslopes), "All scrslopes are invalid!"
   if scrslopes.mask.var()>0: "WARNING: Some scrslopes are invalid!"
      # \/ do noiseless comparison
   doComparison(scr,scrslopes,igM,nrM,pupilAp,nPix,N,
         sapxls,gO,reconScalingFactor,quiet=0)
      # \/ do noisy comparison (normal noise on images)
   print("(NOISE ADDED)")
   noiseSD=0.1*refimgs.reshape([N,N,-1]).sum(axis=-1).max()*N**-2.0
   nLoops=100
   variances={'inv':[],'inv_nr':[],'orig':[]}
   for i in range(nLoops):
      shscrimgsNoisy=numpy.random.normal(shscrimgs,noiseSD)
      scrslopesNoisy=getSlopes(shscrimgsNoisy,cntr,maskI,refslopes,slopeScaling)
      (inv_scr,orig_scr,inv_scr_nr)=doComparison(
            scr,scrslopesNoisy,igM,nrM,pupilAp,nPix,N,
            sapxls,gO,reconScalingFactor,quiet=1)
      variances['inv'].append( inv_scr.var() )
      variances['inv_nr'].append( inv_scr_nr.var() )
      variances['orig'].append( orig_scr.var() )
   #
   print("var{{orig}}={0:5.3f}".format(variances['orig'][-1]))
   print("var{{inv}}={0:5.3f}+/-{1:5.3f}".format(
         numpy.mean(variances['inv']),
         numpy.var(variances['inv'])**0.5
      ))
   print("var{{inv_nr}}={0:5.3f}+/-{1:5.3f}".format(
         numpy.mean(variances['inv_nr']),
         numpy.var(variances['inv_nr'])**0.5
      ))
      # \/ do some plotting to finish up
   import pylab
   doPlotting()
