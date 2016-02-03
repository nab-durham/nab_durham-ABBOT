"""ABBOT : useful functions to make realistic SH patterns.
Note that the radial extension feature is not realistic for LGS
spot extension since it emulates the effect of a fully coherent source.
That is, no attempt is made to average over realizations of the reduction
in spatial coherence so it is entirely possible to obtain a restoration of the
spatial coherence (a removal of the elongation) with an input aberration.
"""

from __future__ import print_function
import numpy
import sys

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

def cds(N, roll=False):
   tcds = (numpy.arange(0,N)-(N/2.-0.5))*(N/2.0)**-1.0
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
#(redundant)      opStem="var{orig}" # always the same
#(redundant)      print("{1:s}{2:s}={0:5.3f}".format(
#(redundant)            orig_scr.var(), " "*(20-len(opStem)), opStem )
#(redundant)         ) 
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
##         cntr=makeCntrArr(sapxls) 
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
   noiseSDscalings = [0,]#numpy.arange(5,30,2)#(1e-9,1.0,10.0,20.0,30.0,33.0,38.0,40.0)
   radialExtension = 0.3 # if >0, do radial extension of spots
   ##
   ## -- end variables ----
##      assert (nPix/N)%2==0, "Require even number of pixels"
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
##         # \/ calibrate the SH model
##      ( slopeScaling, reconScalingFactor, tiltFac, refslopes )=\
##            calibrateSHmodel( pupilAp, cntr, nPix, N, mag, maskI,
##                  defWls, binning, LT, GP, radialExtension 
##               )
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
      ( thisLastImg, thisSD, thisVariances )= thisData[2],thisData[0],thisData[1]
      thisLastImg = thisLastImg.swapaxes(1,2).reshape([nPix]*2)
      pyplot.subplot(4,nRows,j+1)
      pyplot.imshow( thisLastImg, cmap='cubehelix' )
      for ax in pyplot.gca().get_xaxis(),pyplot.gca().get_yaxis():
         ax.set_visible(0)
      pyplot.title( "{0:g}".format( thisSD ),size=10 )
      plotData.append( [ j, thisSD ]+
            map( lambda k :
                  [ numpy.mean(thisVariances[k]),
                    numpy.var(thisVariances[k])
                     ], 
                  ('loopInt','inv','inv-orig')
               )
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
         fSH.makeImgs()
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
