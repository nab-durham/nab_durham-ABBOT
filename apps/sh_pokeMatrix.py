from __future__ import print_function
# What is this?
# Test methods of wavefront reconstruction using a SH spot output from a FT

import abbot.gradientOperator
import abbot.dm
import numpy as np
import sys
import Zernike

ei=lambda x : np.cos(x)+1.0j*np.sin(x)
smilT2=lambda x :\
      np.roll( np.roll( x, x.shape[-1]/2, axis=-1 ), x.shape[-2]/2, axis=-2 )
reshaper=lambda ip :\
      ip.reshape([N,subApS,N,subApS]).swapaxes(1,2)

def getGrads(apWf):
   focalP=smilT2(abs(np.fft.fft2(
         fftCorr*reshaper(aperture*ei(apWf*2*np.pi/1.0))
            ,s=[subApS*fftScl]*2 ))**2.0) * (fftScl*subApS)**-2.0
   focalP=focalP.reshape([-1]+[subApS*fftScl]*2).take(apIdx,axis=0)
   gradsV=np.array([ (focalP*cds.reshape(tShape)).sum(axis=-1).sum(axis=-1)
            /(focalP.sum(axis=-1).sum(axis=-1))
           for tShape in ([1,-1],[-1,1]) ]).ravel()*expectedGradGain**-1.0
   return(gradsV,focalP)

def dmFitter(size,dmSfc,dm):
   dmSfc=dmSfc.reshape(dm.npix)
   if dm.npix==(size,size):
      return dmSfc
   thisdmSfc=np.zeros([size]*2,np.float32)
   if dm.npix[0]>size:
      dmSfc=dmSfc[dm.npix[0]//2-size//2:dm.npix[0]//2-size//2+size,:]
   if dm.npix[1]>size:
      dmSfc=dmSfc[:,dm.npix[1]//2-size//2:dm.npix[1]//2-size//2+size]
   thisdmSfc[size//2-dmSfc.shape[0]//2:size//2-dmSfc.shape[0]//2+dmSfc.shape[0],
       size//2-dmSfc.shape[1]//2:size//2-dmSfc.shape[1]//2+dmSfc.shape[1]]=(
         dmSfc )
   return thisdmSfc

np.random.seed(18071977)

N=12
subApS=8
fftScl=4
dmSize=[(N+1)*subApS]*2 
dmRot=0
dmSpacing=[N+1]*2

# ---

assert not dmSize is None, "dmSize cannot be None in this code"

size=N*subApS
cds=np.arange(fftScl*subApS)-(fftScl*subApS-1)/2.0
corrCds=np.arange(subApS)-(subApS-1)/2.0
fftCorr=ei(-np.pi*(fftScl*subApS)**-1.0*np.add.outer(corrCds,corrCds)
      ).reshape([1,1]+[subApS]*2) # reuse cds
   # \/ a difference of 1 between the edges of the sub-aperture
   # = 1/subApS*(x) = 2*pi/(fftScl*subApS)*(fftScl)/2/pi*(x)
   # so movement is then fftScl/2/pi of a pixel
expectedGradGain=4 #fftScl/2/np.pi
aperture=Zernike.anyZernike(1,size,size//2)
apMask=( reshaper(aperture).sum(axis=-1).sum(axis=-1) 
            > (0.5*(subApS)**2) ).astype(np.bool)
apIdx=apMask.ravel().nonzero()[0]

gO=abbot.gradientOperator.gradientOperatorType1(apMask)
gM=gO.returnOp()
reconM=np.dot(
   np.linalg.inv( np.dot( gM.T, gM )+1e-3*np.identity(gO.numberPhases) ), gM.T )

dm=abbot.dm.dm(dmSize,dmSpacing,rotation=dmRot,within=0,ifScl=2**-0.5)


dmActIdx=(Zernike.anyZernike(1,N+1,(N+1)/2.0)!=0).ravel().nonzero()[0]

print(">",end="") ; sys.stdout.flush()
pokeM=[]
for i in dmActIdx:
   print(".",end="") ; sys.stdout.flush()
   thisApWf=dmFitter(size,dm.poke( i ),dm)
   pokeM.append(getGrads(thisApWf)[0])

print("<")


