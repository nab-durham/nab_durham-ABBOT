from __future__ import print_function
import abbot.gradientOperator
import abbot.continuity
import kolmogorov
import numpy
import sys
import Zernike

## -- begin variables --
##
defWls=[0,]#range(-4,4+2,4)
N=20# how many sub-apertures
nPix=N*2# total size of pupil as a diameter
numpy.random.seed(18071977)
r0=4 # pixels
mag=2
illuminationFraction=0.1
pupilAp=Zernike.anyZernike(1,nPix,nPix//2) # pupil aperture
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
assert mag>1, "Magnifcation>=2"
assert not mag%2, "Magnification=2N for N in I+"

def makeTiltPhase(nPix, fac):
   return -fac*numpy.pi/nPix*numpy.add.outer(
         numpy.arange(nPix),
         numpy.arange(nPix)
      )

def rebin(ip,nPix,N):
   sapxls=nPix//N
   return ip.reshape(
      [N,sapxls,N,sapxls]).swapaxes(1,2).sum(axis=-1).sum(axis=-1)

def rebinplus(ip,nPix,N):
   sapxls=nPix//N
   nx=numpy.zeros([nPix+sapxls]*2,ip.dtype)
   nx[ sapxls//2:-sapxls//2, sapxls//2:-sapxls//2 ]=ip
   return nx.reshape([N+1,sapxls]*2).swapaxes(1,2).sum(axis=-1).sum(axis=-1)

def makeSHImgs( aperture, phs, N, amplitudeScaling=1, wl=defWls, mag=mag):# ):
   '''aperture : overall aperture
      amplitudeScaling : an additional factor to aperture, could be constant
      phs : 2D array of phase
      N : number of sub-apertures
      wl : simulate, crudely, polychromatism via the 'l' variable
      mag : magnification of each spot
   '''
   FFTCentringPhase=lambda twl : makeTiltPhase(nPix,-nPix*(sapxls*mag-twl)**-1.0)
   ##lambda l : ( numpy.add.outer(numpy.arange(256),numpy.arange(256))*numpy.pi*l**-1.0 )
   A=aperture
   C=amplitudeScaling
   polyChromSHImgs=[]
   for twl in wl:
      tip=(A*C*numpy.exp(
            -1.0j*( phs+FFTCentringPhase(twl) ) 
          )).reshape([N,sapxls,N,sapxls]).swapaxes(1,2)
      top=abs(numpy.fft.fftshift(
            numpy.fft.fft2(tip,s=[sapxls*mag-twl]*2
         ), (2,3) ))**2.0
      polyChromSHImgs.append(
            top.take(
               range(sapxls*mag/2-twl/2-sapxls//2,
                     sapxls*mag/2-twl/2+sapxls//2),axis=2
            ).take(
               range(sapxls*mag/2-twl/2-sapxls//2,
                     sapxls*mag/2-twl/2+sapxls//2),axis=3
            )
         )
##            range(N*mag/2-twl/2-N,N*mag/2-twl/2+N),axis=2).take(
##            range(N*mag/2-twl/2-N,N*mag/2-twl/2+N),axis=3)
   monoSHImgs=numpy.mean( polyChromSHImgs, axis=0 )
   #
   return monoSHImgs ##,N*2//N]*2).mean(axis=3).mean(axis=-1)

def makeCntrArr(refimgs):
   cntr=[  numpy.add.outer(
      (numpy.linspace(-1,1-refimgs.shape[2]**-1.0*2.,refimgs.shape[2])),
       numpy.zeros(refimgs.shape[2])) ]
   cntr.append(cntr[0].T)
   return numpy.array(cntr).T.reshape([1,1,refimgs.shape[2],refimgs.shape[2],2])

def getSlopes(shimgs,cntr,maskI,refslopes=0,slopeScaling=1):
   rawSlopes=( (cntr*shimgs.reshape(list(shimgs.shape)+[1])
         ).sum(axis=2).sum(axis=2)/
            (1e-10+shimgs.reshape(list(shimgs.shape)+[1])
         ).sum(axis=2).sum(axis=2)
      )
   slopesV=rawSlopes.reshape([-1,2]).take(maskI,axis=0).T.ravel()
   slopesV=numpy.ma.masked_array(slopesV,numpy.isnan(slopesV))
   return (slopesV-refslopes)*slopeScaling


print("2",end="");sys.stdout.flush()
sapxls=nPix//N
if 'pupilAp' not in dir():
   pupilAp=Zernike.anyZernike(1,nPix,nPix//2) # pupil aperture
refimgs=makeSHImgs(pupilAp,0,N)
cntr=makeCntrArr(refimgs)
mask=rebin(pupilAp,nPix,N)*(sapxls**-2.0)
maskB=(mask>=illuminationFraction)
numSAs=maskB.sum()
maskI=(maskB.ravel()).nonzero()[0]
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
# and finally the effect of rebinplus
reconScalingFactor*=sapxls**2.

   # Make reference values
refslopes=getSlopes(refimgs,cntr,maskI) # raw slopes
assert refslopes.mask.sum()==0, "All refslopes are invalid!"
assert refslopes.mask.var()==0, "Some refslopes are invalid!"

print("4",end="");sys.stdout.flush()
gO=abbot.gradientOperator.gradientOperatorType1( maskB )
assert gO.numberSubaps==numSAs
gM=gO.returnOp()

print("5",end="");sys.stdout.flush()
igM=numpy.linalg.inv(
      gM.T.dot(gM)+1e-3*numpy.identity(gO.numberPhases)).dot(gM.T)
##numpy.linalg.pinv(gM,1e-3)

print("6",end="");sys.stdout.flush()
loopsDef=abbot.continuity.loopsIntegrationMatrix( maskB )
nrDef=abbot.continuity.loopsNoiseMatrices( maskB )
loopIntM=loopsDef.returnOp()
neM,nrM=nrDef.returnOp()

print("7",end="");sys.stdout.flush()
scr=kolmogorov.TwoScreens(
      nPix*4,r0,flattening=2*numpy.pi*sapxls**-1.0)[0][:nPix,:nPix]
### remove relative tilt
##for i in (0,1):
##   scr_meangrad=numpy.polyfit(numpy.arange(256),scr.mean(axis=i),1)[0]
##   scr-=scr_meangrad*numpy.add.outer(
##         numpy.arange(256)*(1-i),numpy.arange(256)*i)
##raise RuntimeError("verify I work")
print(".",end="");sys.stdout.flush()
##print("8",end="");sys.stdout.flush()
##sfmask=lambda s,l,z : numpy.fft.fftshift(
##   numpy.multiply.outer(
##      abs(numpy.arange(-l/2+s/2,l/2-s/2))<((l-s)*z**-1.0),
##      abs(numpy.arange(-l/2+s/2,l/2-s/2))<((l-s)*z**-1.0) ))
##sfscr=numpy.fft.ifft2(
##      numpy.fft.fft2(scr,s=[1024]*2)*sfmask(0,1024,32) )[:256,:256].real
##sfscrB=numpy.fft.ifft2(
##      numpy.fft.fft2(scr,s=[1024]*2)*sfmask(0,1024,64) )[:256,:256].real
##print(".",end="");sys.stdout.flush()
#print("9",end="");sys.stdout.flush()
#randomphs=numpy.array(
#   [ Zernike.anyZernike(j,256,128)*numpy.random.normal()
#         for j in range(2,137) ]).sum(axis=0)
#print(".",end="");sys.stdout.flush()


print("A",end="");sys.stdout.flush()
shscrimgs=makeSHImgs(pupilAp,scr,N)
print(".",end="");sys.stdout.flush()
#print("C",end="");sys.stdout.flush()
#randzimgs=shimgs(1,randomphs*0.05,N)
#print(".",end="");sys.stdout.flush()
print("Da",end="");sys.stdout.flush()
tilt1ximgs=makeSHImgs(pupilAp,tiltphs1x,N)
print(",b",end="");sys.stdout.flush()
tilt2ximgs=makeSHImgs(pupilAp,tiltphs2x,N)
print(".",end="");sys.stdout.flush()

#randzslopes=getSlopes(randzimgs)-refslope,cntr,maskIs
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
slopeScaling=0.5*(1/tilt1xslopes+2/tilt2xslopes) # slope scaling
# get the i/p
scrslopes=getSlopes(shscrimgs,cntr,maskI,refslopes,slopeScaling)
assert scrslopes.mask.sum()<len(scrslopes), "All scrslopes are invalid!"
if scrslopes.mask.var()>0: "WARNING: Some scrslopes are invalid!"

# now try reconstruction
orig_scr=numpy.ma.masked_array(
      rebinplus(pupilAp*scr,nPix,N),
      rebinplus(pupilAp,nPix,N)!=sapxls**2
   )
##orig_tilt=numpy.ma.masked_array(
##      rebinplus(pupilAp*tiltphs,nPix,N),
##      rebinplus(pupilAp,nPix,N)!=sapxls**2
##   )

print("X",end="");sys.stdout.flush()
inv_scrV=numpy.dot(igM,scrslopes)
inv_scr_nrV=numpy.dot(igM,nrM.dot(scrslopes))
#inv_tiltV=numpy.dot(igM,tiltslopes)
print(".",end="");sys.stdout.flush()

##assert (nPix==256 and N==16), "Code is hard-wired for nPix,N=256,16"
##scl=(1.46777047e-04**-1.0) # this should be worked out by the code, automatically

##inv_tilt=numpy.ma.masked_array(
##      numpy.empty(gO.n_,numpy.float64),
##      rebinplus(pupilAp,nPix,N)!=sapxls**2
##   )
inv_scr=numpy.ma.masked_array(
      numpy.empty(gO.n_,numpy.float64),
      rebinplus(pupilAp,nPix,N)!=sapxls**2
   )
inv_scr_nr=numpy.ma.masked_array(
      numpy.empty(gO.n_,numpy.float64),
      rebinplus(pupilAp,nPix,N)!=sapxls**2
   )

##inv_tilt.ravel()[gO.illuminatedCornersIdx]=inv_tiltV*scl
inv_scr.ravel()[gO.illuminatedCornersIdx]=inv_scrV*reconScalingFactor
inv_scr_nr.ravel()[gO.illuminatedCornersIdx]=inv_scr_nrV*reconScalingFactor


print()

print("scr.var()={0:5.3f}".format( scr.var() ))
print("scrslopes.var()={0:5.3f}".format( scrslopes.var() ))
##print("tilt test:{0:5.3g} residual".format((inv_tilt-orig_tilt).var()/orig_tilt.var()))
print("loopIntM[scrslopes].var()={0:5.3f}".format(
      loopIntM.dot(scrslopes).var() ))
print("{{inv_scr-orig_scr}}_v/{{orig_scr}}_v={0:5.3f}".format( (inv_scr-orig_scr).var()/orig_scr.var() ))
print("{{inv_scr_nr-orig_scr}}_v/{{orig_scr}}_v={0:5.3f}".format( (inv_scr_nr-orig_scr).var()/orig_scr.var() ))

import pylab
pylab.figure(99)
for i,(thisImg,title) in enumerate([
         (refimgs,"reference"),
         (tilt1ximgs,"tilt1x"),
         (tilt2ximgs,"tilt2x"),
         (shscrimgs,"scr"),
      ]):
   pylab.subplot(2,2,1+i)
   pylab.imshow( numpy.log10(thisImg.swapaxes(1,2).reshape([nPix]*2)) )
   pylab.title("log10{{{0:s}}}".format(title))

pylab.figure(98)
for ip in inv_scr,inv_scr_nr,orig_scr:
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
