from __future__ import print_function
# What is this?
# Test methods of wavefront reconstruction using a SH spot output from a FT

import numpy as np
np.random.seed(18071977)

N=12
subApS=8
fftScl=4
dmSize=[(N+1)*subApS]*2 # can be None
dmRot=0
dmSpacing=[N+1]*2
#dmSize[0]-=8
#dmSize[1]+=4
#dmRot=5

# ---

import abbot.gradientOperator
import abbot.dicure
import abbot.dm
import pylab
import sys
import Zernike
import kolmogorov


ei=lambda x : np.cos(x)+1.0j*np.sin(x)
smilT2=lambda x :\
      np.roll( np.roll( x, x.shape[-1]/2, axis=-1 ), x.shape[-2]/2, axis=-2 )
reshaper=lambda ip :\
      ip.reshape([N,subApS,N,subApS]).swapaxes(1,2)
flattener=lambda focalP:\
   focalP.reshape([-1]+[subApS*fftScl]*2).take(apMaskIdx,axis=0)

def prepCure(gO):
   chainsNumber,chainsDef,chainsDefChStrts=abbot.dicure.chainsDefine( gO )
   chainsOvlps=abbot.dicure.chainsOverlaps(chainsDef)
   A,B=abbot.dicure.chainsDefMatrices(chainsOvlps, chainsDef, chainsDefChStrts)
   #
   #invchOScovM=np.identity(A.shape[1])
   #offsetEstM=np.dot(
   #         np.dot( np.linalg.inv( np.dot(A.T,A)+invchOScovM*1e-3 ), A.T), -B )
   offsetEstM=np.dot( np.linalg.pinv( A, rcond=0.1 ), -B )
   wO=abbot.gradientOperator.waffleOperatorType1(apMask)
   wM=wO.returnOp()
   #
   return(chainsDef,chainsDefChStrts,offsetEstM,wM)

def doCure(gradsV,chainsDef,gO,offsetEstM,chainsDefChStrts,wM):
   rgradV=abbot.dicure.rotateVectors(gradsV).ravel()
   chains=abbot.dicure.chainsIntegrate(chainsDef,rgradV,gO)
   chainsV,chainsVOffsets=abbot.dicure.chainsVectorize(chains)
   offsetEstV=np.dot( offsetEstM, chainsV )
   #      
   # do one way...
   for x in range(len(chains[0])):
      toeI=x
      for i in range((chainsDef[0][x][1])):
         chainsV[chainsVOffsets[x]+i]+=offsetEstV[toeI]
   # ...then another
   for x in range(len(chains[1])):
      toeI=chainsDefChStrts[1][2][x]
      for i in range((chainsDef[1][x][1])):
         chainsV[chainsVOffsets[x+len(chains[0])]+i]+=offsetEstV[toeI]
   #
   comp=np.zeros([2,gO.n_[0]*gO.n_[1]], np.float64)
   numbers=np.zeros([gO.n_[0]*gO.n_[1]], np.float64)
   for x in range(len(chains[0])):
      comp[0][ chainsDef[0][x][0] ]=\
         chainsV[chainsVOffsets[x]:chainsVOffsets[x]+chainsDef[0][x][1]]
      numbers[ chainsDef[0][x][0] ]+=1
   #
   for x in range(len(chains[1])):
      comp[1][ chainsDef[1][x][0] ]=\
         chainsV[chainsVOffsets[x+len(chains[1])]:\
                 chainsVOffsets[x+len(chains[1])]+chainsDef[1][x][1]]
      numbers[ chainsDef[1][x][0] ]+=1
   #
   cureV=((comp[0]+comp[1])*(numbers+1e-9)**-1.0
            ).ravel()[gO.illuminatedCornersIdx]
   cureV-=np.dot( wM,cureV )*wM # remove waffle
   comp=comp.reshape([2]+gO.n_)
   #
   return comp,cureV,chainsV,chains

def mkFocalP(apWf):
   return smilT2(abs(np.fft.fft2(
         fftCorr*reshaper(aperture*ei(apWf*2*np.pi/1.0))
            ,s=[subApS*fftScl]*2 ))**2.0) * (fftScl*subApS)**-2.0

def getGrads(apWf):
   tfocalP=flattener(mkFocalP(apWf))
   gradsV=np.array([ (tfocalP*cds.reshape(tShape)).sum(axis=-1).sum(axis=-1)
            /(tfocalP.sum(axis=-1).sum(axis=-1))
           for tShape in ([1,-1],[-1,1]) ]).ravel()*expectedGradGain**-1.0
   return gradsV

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

def doPlottingComparisonTheory(reconWfV):
   newapWf=np.zeros([size+subApS,size+subApS])
   newapWf[subApS/2:size+subApS/2,subApS/2:size+subApS/2]=apWf
   apEdgeCds=np.linspace(subApS/2,subApS/2+size,N+1)
   aEC=apEdgeCds

   binnedApWf=np.array([
      [newapWf[aEC[y]-subApS/2:aEC[y]+subApS/2,
               aEC[x]-subApS/2:aEC[x]+subApS/2].mean()
       for x in range(N+1) ] for y in range(N+1) ])
   binnedApWfV=binnedApWf.ravel().take( gO.illuminatedCornersIdx )

   reconWf=np.zeros( [N+1]*2 )
   reconWf.ravel()[ gO.illuminatedCornersIdx ]=reconWfV

   pylab.figure()
   pylab.plot( binnedApWfV, reconWfV, 'k.' )
   pylab.plot([binnedApWfV.min(),binnedApWfV.max()],
         [binnedApWfV.min(),binnedApWfV.max()],'r--')

   return( binnedApWf, reconWf )

def doPlottingComparisonCure(cureV,gO):
   compAvg=gO.illuminatedCorners*0
   compAvg.ravel()[gO.illuminatedCornersIdx]=cureV
   pylab.figure()
   pylab.subplot(1,2,1)
   pylab.imshow( compAvg )
   pylab.title("dicure'd")
   pylab.subplot(1,2,2)
   pylab.imshow( reconWf )
   pylab.title("reconWf")


def doPlottingComparisonPoke(pokeReconWfV,dmActIdx,dm):
   reconWf=np.zeros( [size]*2 )
   tdmActIdx=dmActIdx.tolist()
   while len(tdmActIdx)>0:
      randI=int(np.random.uniform(0,len(tdmActIdx)))
      tActIdx=tdmActIdx.pop(randI)
      tActMap=np.searchsorted(dmActIdx,tActIdx)
      reconWf+=dmFitter(size,dm.poke( tActIdx )*pokeReconWfV[tActMap],dm)
   #
   reconWf*=aperture
   pylab.figure()
   pylab.subplot(1,2,1)
   pylab.imshow( reconWf )
   pylab.subplot(1,2,2)
   pylab.imshow( apWf )
   #G
   return( reconWf )

def doPlottingFocalPlane(tfocalP):
   
   pylab.figure(1)
   pylab.subplot(2,2,2)
   pylab.imshow( flattener(tfocalP)[0] )
   pylab.subplot(2,2,4)
   pylab.imshow( flattener(tfocalP)[-1] )

   pylab.subplot(1,2,1)
   pylab.imshow( ((tfocalP[:,:,
      subApS*(fftScl-1)/2:subApS*(fftScl+1)/2,
      subApS*(fftScl-1)/2:subApS*(fftScl+1)/2]*apMask.reshape([N,N,1,1])
                  ).swapaxes(1,2).reshape([size]*2)) )
   pylab.title("spots")

# ---

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
apertureIdx=aperture.ravel().nonzero()[0]
apMask=( reshaper(aperture).sum(axis=-1).sum(axis=-1) 
            > (0.5*(subApS)**2) ).astype(np.bool)
apMaskIdx=apMask.ravel().nonzero()[0]

gO=abbot.gradientOperator.gradientOperatorType1(apMask)
gM=gO.returnOp()
reconM=np.dot(
   np.linalg.inv( np.dot( gM.T, gM )+1e-3*np.identity(gO.numberPhases) ), gM.T )

if dmSize!=None:
   dm=abbot.dm.dm(dmSize,dmSpacing,rotation=dmRot,within=0,ifScl=2**-0.5)
else:
   dm=None

##zNums=[2,3] ; zAmps=[1,-1]
#zNums=range(2,Zernike.zernDegFreqToNum(12,0))
#zAmps=np.random.normal(0,len(zNums)**-1.0**0.5,size=len(zNums))
#apWf=0
#print(">",end="")
#for i in range(len(zNums)):
#   apWf+=zAmps[i]*Zernike.anyZernike(zNums[i],size,size//2)
#   print(".",end="") ; sys.stdout.flush()
print("kolmog:",end="") ; sys.stdout.flush()
apWf=kolmogorov.TwoScreens(2**(int(np.log(size)/np.log(2))+1+2),
   size//N*4)[0]
apWf=apWf[apWf.shape[0]//2-size//2:apWf.shape[0]//2+size//2,
          apWf.shape[1]//2-size//2:apWf.shape[1]//2+size//2]*aperture
print("(done)") ; sys.stdout.flush()

if type(dm)!=type(None):
   dmActIdx=(Zernike.anyZernike(1,np.max(dmSpacing),np.max(dmSpacing)/2.0
         )!=0).ravel().nonzero()[0]

   print(">",end="") ; sys.stdout.flush()
   pokeM=[]
   for i in dmActIdx:
      print("+",end="") ; sys.stdout.flush()
      thisApWf=dmFitter(size,dm.poke( i ),dm)
      pokeM.append(getGrads(thisApWf))

   pokeM=np.array(pokeM).T
   pokeReconM=np.dot( np.linalg.inv(
        np.dot( pokeM.T, pokeM )+1e-3*np.identity(pokeM.shape[1]) ), pokeM.T )
else:
   pokeM=None

print("<")


# --- off we go! make a set of spots, get the gradients by centre of gravity
# and then reconstruct

##\/test, shift by one in gradient value
#apWf=(1./(fftScl*subApS)
#      *np.add.outer( np.arange(size)-size/2,np.zeros(size) ) )

print("<")

gradsV,focalP=getGrads(apWf),mkFocalP(apWf)
doPlottingFocalPlane(focalP)
   
reconWfV=np.dot( reconM, gradsV )
if not pokeM is None:
   pokeReconWfV=np.dot( pokeReconM, gradsV )

binnedApWf,reconWf=doPlottingComparisonTheory(reconWfV)

if not pokeM is None:
   pokedWf=doPlottingComparisonPoke(pokeReconWfV,dmActIdx,dm)

chainsDef,chainsDefChStrts,offsetEstM,wM=prepCure(gO)

print("{cure}",end="") ; sys.stdout.flush()
comp,cureWfV,chainsV,chains=doCure(
      gradsV,chainsDef,gO,offsetEstM,chainsDefChStrts,wM*0)
print("{done}") ; sys.stdout.flush()

doPlottingComparisonCure(cureWfV,gO)

if not pokeM is None:
   cureIntM=[]
   print("poke->cure") ; sys.stdout.flush()
   for i in range(pokeM.shape[1]):
      comp,cureV,chainsV,chains=doCure(
            pokeM[:,i],chainsDef,gO,offsetEstM,chainsDefChStrts,wM)
      cureIntM.append( cureV )

   cureIntM=np.array(cureIntM).T
   # \/ this converts from a cure w/f to the
   # poke matrix w/f.
   cureToDMM=np.linalg.pinv( cureIntM, rcond=1e-2 )
   cureActV=np.dot( cureToDMM, cureWfV )
   curePokedWf=doPlottingComparisonPoke(cureActV,dmActIdx,dm)
   # overall comparison
   pylab.figure()
   pylab.subplot(3,2,1)
   pylab.imshow( apWf ) ; pylab.title("input w/f")
   pylab.subplot(3,2,2*1+1)
   pylab.imshow( pokedWf ) ; pylab.title("poked wf w/f")
   pylab.subplot(3,2,2*1+2)
   pylab.imshow( pokedWf-apWf ) ; pylab.title("poked wf less i/p w/f")
   pylab.subplot(3,2,2*2+2)
   pylab.imshow( curePokedWf-apWf ) ; pylab.title("cure via FM less i/p w/f")
   pylab.subplot(3,2,2*2+1)
   pylab.imshow( curePokedWf ) ; pylab.title("cure via DM w/f")

   print("i/p-poked w/f rel var={0:5.3f}".format(
         (apWf-pokedWf).ravel()[apertureIdx].var()/
         (apWf).ravel()[apertureIdx].var() ))
   print("i/p-cured-poked w/f rel var={0:5.3f}".format(
         (apWf-curePokedWf).ravel()[apertureIdx].var()/
         (apWf).ravel()[apertureIdx].var() ))
