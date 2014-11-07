from __future__ import print_function
# What is this?
# Test MVM with regularly spaced apertures to see if mean phase can be
# estimated

import time
import matplotlib.pyplot as pg
import numpy
import gradientOperator
#import commonSeed 
import sys

#< timings={}
# test code
r0=[0.5,0.5] # in pixels
nfft=100 # pixel size
L0=[ 30, 30] # outer scale of [phase, covariance regularisation]

import Zernike
# define pupil mask as sub-apertures
#< timings['s:maskC']=time.time()
#> bAp=Zernike.anyZernike(1,nfft,7)
#> oAp=bAp+0.0
#> r=(nfft/2/2.0-8/2.0)*2
#> cds=numpy.array([ [numpy.sin(t)*r,numpy.cos(t)*r]
#>    for t in numpy.arange(6)/6.0*2*numpy.pi ])
#> for i in cds:
#>   oAp+=numpy.roll(numpy.roll(bAp,int(i[0])),int(i[1]),axis=0)
#> subapMask=oAp

pupilMask=numpy.zeros([6,nfft],numpy.int32)
subPupils=[]
#,  # vertical parts
nspan=3.0
print("[")
#, for i in range(int(nspan)):
#,    subPupil=Zernike.anyZernike(1,nfft,3,offset=[-nfft/2+5+i*nfft/float(nspan),-nfft/2+5])
#,    subPupils.append(numpy.flatnonzero(subPupil.ravel()))
#,    pupilMask+=subPupil
 # horizontal parts
print(" : ")
for j in range(1,int(nfft/nspan)):
   subPupil=Zernike.anyZernike(1,nfft,1,
         offset=[0,-nfft/2+2+j*nspan])[nfft/2-3:nfft/2+3]
   subPupils.append(numpy.flatnonzero(subPupil.ravel()))
   pupilMask+=subPupil
   print(".",end="") ; sys.stdout.flush()

print("]")

#pupilCds=numpy.add.outer(
#   (numpy.arange(nfft)-(nfft-1)/2.)**2.0,
#   (numpy.arange(nfft)-(nfft-1)/2.)**2.0 )
#>    # \/ spider
#pupilMask[:nfft/2,nfft/2-0:nfft/2+2]=0
#pupilMask[nfft/2:,nfft/2-2:nfft/2+0]=0
#pupilMask[nfft/2-0:nfft/2+2,:nfft/2]=0
#pupilMask[nfft/2-2:nfft/2+0,nfft/2:]=0
#pupilMask*=(pupilCds>(nfft/8)**2)*(pupilCds<(nfft/2)**2)
#pupilMask=(pupilCds>(nfft/3)**2)
#pupilMask=(pupilCds<(nfft/2)**2)
#< timings['e:maskC']=time.time()

gO=gradientOperator.gradientOperatorType1()
gO.newPupilGiven(pupilMask)
subapMask=numpy.zeros(gO.n)
subapMask.ravel()[ (gO.subapMaskIdx!=-1).ravel().nonzero()[0] ]=1

gM=gO.returnOp()

# define phase at corners of pixels, each of which is a sub-aperture
centreExtract=lambda x,n :\
   x.ravel()[numpy.flatnonzero(
      (abs(numpy.arange(x.shape[0]*x.shape[1])//(x.shape[0])-(x.shape[0]-1)/2.)>n)&\
      (abs(numpy.arange(x.shape[0]*x.shape[1])%(x.shape[1])-(x.shape[1]-1)/2.)>n)
         )].reshape([x.shape[0]-2*n,x.shape[1]-2*n])

#< timings['s:phaseC']=time.time()
import phaseCovariance as pc
singleCov1=pc.covarianceDirectRegular(nfft+1,r0[0],L0[0])
cov=pc.covarianceMatrixFillInMasked(singleCov1,numpy.ones(pupilMask.shape))
choleskyC=pc.choleskyDecomp(cov)
#< timings['e:phaseC']=time.time()

# define phase covariances

# now try solving
# linear least squares is (G^T G+alphaC^-1)^-1 G^T
# alpha!=0 with MVM
# beta!=0 with bi-harmonic approximation
#< timings['s:lOM']=time.time()

singleCov2=pc.covarianceDirectRegular(nfft+2,r0[1],L0[1])
covM=pc.covarianceMatrixFillInMasked(
   centreExtract(singleCov2,1), gO.illuminatedCorners )
   # \/ power method for largest eigenvalue of covariance matrix
#< timings['s:eigV']=time.time()
vecLen=covM.shape[0]
eigVEst=numpy.random.uniform(size=vecLen)
eigEstV=numpy.dot(eigVEst,eigVEst)**0.5
relChangeEigV=1
iterNum=1
while relChangeEigV>0.01:
   eigVEst=numpy.dot( covM, eigVEst ) # iterate
   oldEigEstV=eigEstV
   eigEstV=numpy.dot(eigVEst,eigVEst)**0.5
   eigVEst/=eigEstV
   relChangeEigV=abs(oldEigEstV-eigEstV)/abs(eigEstV)
   iterNum+=1
#< timings['e:eigV']=time.time()

#< timings['s:gtgM']=time.time()
gTg=numpy.dot( gM.transpose(), gM )
#< timings['e:gtgM']=time.time()
#< timings['s:invCovM']=time.time()
RinvCovM=numpy.linalg.inv( covM )
#< timings['e:invCovM']=time.time()

alpha=eigEstV*1e-6 # quash about largest eigenvalue

   # \/ MVM
beta=alpha#0.1
invgTgMVMM=numpy.linalg.inv( gTg + beta*RinvCovM )

# generate phase and gradients
#< timings['s:phasev']=time.time()
nreps=100
meanPhases=[]
print("<",end="")
for n in range(nreps):
   onePhase=numpy.dot( choleskyC, numpy.random.normal(size=pupilMask.shape ).ravel() )
   onePhaseV=onePhase[gO.illuminatedCornersIdx]
   onePhase-=onePhaseV.mean() # normalise
   onePhase.resize(pupilMask.shape)
   #< timings['e:phasev']=time.time()
   gradV=numpy.dot(gM,onePhaseV)

   # reconstructor matrices
   reconPhaseV=numpy.dot( invgTgMVMM, numpy.dot( gM.transpose(), gradV ) )

   # imaging of phases
   reconPhaseD=numpy.zeros(pupilMask.shape,numpy.float64)
   reconPhaseD.ravel()[gO.illuminatedCornersIdx]=reconPhaseV
   #reconPhaseD=numpy.ma.masked_array( reconPhaseD, [gO.illuminatedCorners==0] )

   meanPhases.append([])
   meanPhases[-1].append( [ reconPhaseD.ravel()[z].mean() for z in subPupils ] )
   meanPhases[-1].append( [ onePhase.ravel()[z].mean() for z in subPupils ] )
   print(".",end="") ; sys.stdout.flush()

print(">")
reconPhaseD=numpy.ma.masked_array( reconPhaseD, [gO.illuminatedCorners==0] )
onePhaseD=numpy.ma.masked_array(onePhase,gO.illuminatedCorners==0)
meanPhases=numpy.array(meanPhases)

#, pg.figure(2)
#, pg.subplot(3,1,1)
#, pg.imshow( onePhaseD, interpolation='nearest', origin='lower',
#,    extent=[-0.5,pupilMask.shape[1]+0.5,-0.5,pupilMask.shape[0]+0.5] )
#, pg.title("Fried, MVM: input phase")
#, pg.colorbar()
#, pg.subplot(3,1,2)
#, pg.imshow( reconPhaseD, interpolation='nearest', origin='lower',
#,    extent=[-0.5,pupilMask.shape[1]+0.5,-0.5,pupilMask.shape[0]+0.5] )
#, pg.title("reconstructed phase (MVM reg)")
#, pg.colorbar()
#, pg.subplot(3,1,3)
#, pg.imshow( reconPhaseD-onePhaseD, interpolation='nearest', origin='lower',
#,    extent=[-0.5,pupilMask.shape[1]+0.5,-0.5,pupilMask.shape[0]+0.5] )
#, pg.colorbar()
#} 
#} for dat in (
#}       ("Mask creation","maskC"), ("Phase covariance","phaseC"),
#}       ("Eig Values time","eigV"),("Phase creation","phasev"),
#}       ("gTg","gtgM"), ("Inverse covariance matrix","invCovM"),
#}      ):
#}    print("{1}={0:5.3f}s".format(
#}       timings['e:'+dat[1]]-timings['s:'+dat[1]],dat[0]))

pg.subplot(1,3,1) ; pg.imshow( meanPhases[:,0], vmin=meanPhases[:,1].min(), vmax=meanPhases[:,1].max() )
pg.subplot(1,3,2) ; pg.imshow( meanPhases[:,1], vmin=meanPhases[:,1].min(), vmax=meanPhases[:,1].max() )
pg.subplot(1,3,3) ; pg.imshow( meanPhases[:,1]-meanPhases[:,0], vmin=(meanPhases[:,1]-meanPhases[:,0]).min(), vmax=(meanPhases[:,1]-meanPhases[:,0]).max() )

print("Actual phases, variance", (meanPhases[:,1].var(axis=1).mean(),meanPhases[:,1].var(axis=1).var()**0.5 ))
pd=meanPhases[:,1]-meanPhases[:,0]
print("Actual less estimated difference phase, variance", (
      pd.var(axis=1).mean(), pd.var(axis=1).var()**0.5 ))

print("Second order difference, actual phases, mean and mean of every other", (
      (meanPhases[:,1,2:]+meanPhases[:,1,:-2]-2*meanPhases[:,1,1:-1]).mean(),
      (meanPhases[:,1,2::2]+meanPhases[:,1,:-2:2]-2*meanPhases[:,1,1:-1:2]).mean() ))
print("Second order difference, actual less estimated phases, mean", (
      (pd[:,2:]+pd[:,:-2]-2*pd[:,1:-1]).mean(),
      (pd[:,2::2]+pd[:,:-2:2]-2*pd[:,1:-1:2]).mean() ))
