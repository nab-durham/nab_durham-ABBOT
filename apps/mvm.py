# What is this?
# Test MVM

import time
import matplotlib.pyplot as pg
import numpy
import gradientOperator
import commonSeed 

timings={}
# test code
r0=1 # in pixels
L0=[ 1000, 48  ] # outer scale of [phase, covariance regularisation]
nfft=31 # pixel size

# define pupil mask as sub-apertures
timings['s:maskC']=time.time()
pupilMask=numpy.ones([nfft]*2,numpy.int32)
pupilCds=numpy.add.outer(
   (numpy.arange(nfft)-(nfft-1)/2.)**2.0,
   (numpy.arange(nfft)-(nfft-1)/2.)**2.0 )
pupilMask=(pupilCds>(nfft/6)**2)*(pupilCds<(nfft/2)**2)
timings['e:maskC']=time.time()
gO=gradientOperator.gradientOperatorType1(pupilMask)
gM=gO.returnOp()

# define phase at corners of pixels, each of which is a sub-aperture
centreExtract=lambda x,n :\
   x.ravel()[numpy.flatnonzero(
      (abs(numpy.arange(x.shape[0]*x.shape[1])//(x.shape[0])-(x.shape[0]-1)/2.)>n)&\
      (abs(numpy.arange(x.shape[0]*x.shape[1])%(x.shape[1])-(x.shape[1]-1)/2.)>n)
         )].reshape([x.shape[0]-2*n,x.shape[1]-2*n])

timings['s:phaseC']=time.time()
import phaseCovariance as pc
singleCov1=pc.covarianceDirectRegular(nfft+2,r0,L0[0])
cov=pc.covarianceMatrixFillInRegular(singleCov1)
choleskyC=pc.choleskyDecomp(cov)
timings['e:phaseC']=time.time()

# define phase covariances

# generate phase and gradients
timings['s:phasev']=time.time()
onePhase=numpy.dot( choleskyC, numpy.random.normal(size=(nfft+2)**2 ) )

   # \/ for comparison purposes onePhase is too big, so take a mean
meanO=numpy.zeros([(nfft+1)**2,(nfft+2)**2], numpy.float64)
for i in range((nfft+1)**2):
   meanO[i,i//(nfft+1)*(nfft+2)+i%(nfft+1)]=0.25
   meanO[i,i//(nfft+1)*(nfft+2)+i%(nfft+1)+1]=0.25
   meanO[i,(i//(nfft+1)+1)*(nfft+2)+i%(nfft+1)]=0.25
   meanO[i,(i//(nfft+1)+1)*(nfft+2)+i%(nfft+1)+1]=0.25
onePhase=numpy.dot(meanO,onePhase)

onePhaseV=onePhase[gO.illuminatedCornersIdx]
onePhase-=onePhaseV.mean() # normalise
onePhase.resize([nfft+1]*2)
onePhaseD=numpy.ma.masked_array(onePhase,gO.illuminatedCorners==0)
timings['e:phasev']=time.time()
gradV=numpy.dot(gM,onePhaseV)

# now try solving
# linear least squares is (G^T G+alphaC^-1)^-1 G^T
# alpha!=0 with MVM
# alpha=0 with no regularisation
# or use SVD to quash poorly spanned eigenvectors of phase space
singleCov2=pc.covarianceDirectRegular(nfft+2,r0,L0[1])
covM=pc.covarianceMatrixFillInMasked(
   centreExtract(singleCov2,1), gO.illuminatedCorners )
   # \/ power method for largest eigenvalue
timings['s:eigV']=time.time()
vecLen=covM.shape[0]
eigVEst=numpy.random.uniform(size=vecLen)
eigEstV=numpy.dot(eigVEst,eigVEst)**0.5
relChangeEigV=1
iterNum=1
while relChangeEigV>0.01:
   eigVEst=numpy.dot(covM,eigVEst) # iterate
   oldEigEstV=eigEstV
   eigEstV=numpy.dot(eigVEst,eigVEst)**0.5
   eigVEst/=eigEstV
   relChangeEigV=abs(oldEigEstV-eigEstV)/abs(eigEstV)
   iterNum+=1
timings['e:eigV']=time.time()

timings['s:gtgM']=time.time()
gTg=numpy.dot( gM.transpose(), gM )
timings['e:gtgM']=time.time()
timings['s:invCovM']=time.time()
RinvCovM=numpy.linalg.inv( covM+numpy.identity(gO.numberPhases)*eigEstV*1e-3 )
# sparsify
makeSparse=0
if makeSparse:
   maxRinvCovM=abs(RinvCovM).max()
   RinvCovM=numpy.where( abs(RinvCovM)>(maxRinvCovM*1e-3), RinvCovM, 0 )
   print("** SPARSE cov inv.")
else:
   print("Non-sparse cov inv.")
timings['e:invCovM']=time.time()

timings['e:invCovM']=time.time()

try:
   # \/ no regularisation 
   invgTgM=numpy.linalg.inv(gTg)
except numpy.linalg.LinAlgError as exceptionErr:
   print("FAILURE: Inversion didn't work ,{0:s}".format(exceptionErr))
   invgTgM=None

   # \/ MVM
beta=1
invgTgMVMM=numpy.linalg.inv( gTg + beta*RinvCovM )

try:
   # \/ SVD reconstruction
   invgTgSVDM=numpy.linalg.pinv(gTg,rcond=1e-14)
except numpy.linalg.LinAlgError as exceptionErr:
   print("FAILURE: SVD didn't converge,{0:s}".format(exceptionErr))
   invgTgSVDM=None


# three reconstructor matrices
reconM=[]
for thisInvM in (invgTgMVMM, invgTgM, invgTgSVDM):
   if not thisInvM is None:
      reconM.append( numpy.dot( thisInvM, gM.transpose() ) )
   else: 
      reconM.append( None )
      
reconPhaseV=[]
for thisRM in reconM:
   if thisRM is not None:
      reconPhaseV.append( numpy.dot( thisRM, gradV ) )
   else:      
      reconPhaseV.append(None)

# imaging of phases
reconPhaseD=[] 
for i in range(3):
   if reconM[i] is None: 
      reconPhaseD.append(None)
   else:
      thisPhaseD=numpy.zeros((nfft+1)**2,numpy.float64)
      thisPhaseD[gO.illuminatedCornersIdx]=reconPhaseV[i]
      reconPhaseD.append( numpy.ma.masked_array(
         thisPhaseD.reshape([nfft+1]*2), [gO.illuminatedCorners==0] ) )

pg.figure(2)
pg.subplot(221)
pg.imshow( onePhaseD, interpolation='nearest', origin='lower',
   extent=[-0.5,nfft+0.5,-0.5,nfft+0.5] )
pg.title("Fried, MVM: input phase")
pg.colorbar()
pg.subplot(222)
pg.imshow( reconPhaseD[0], interpolation='nearest', origin='lower',
   extent=[-0.5,nfft+0.5,-0.5,nfft+0.5] )
pg.title("reconstructed phase (MVM reg)")
pg.colorbar()
pg.subplot(234)
pg.imshow( reconPhaseD[0]-onePhaseD, interpolation='nearest', origin='lower',
   extent=[-0.5,nfft+0.5,-0.5,nfft+0.5] )
pg.title("diff (MVM.)")
pg.colorbar()
pg.subplot(235)
if not reconM[1] is None:
   pg.imshow( reconPhaseD[1]-onePhaseD, interpolation='nearest', origin='lower',
      extent=[-1.5,nfft+0.5,-1.5,nfft+0.5] )
   pg.title("diff (No reg.)")
   pg.colorbar()
else:
   pg.text(0.1,0.1,"Inv failed")
pg.subplot(236)
if reconM[2] is not None:
   pg.imshow( reconPhaseD[2]-onePhaseD, interpolation='nearest', origin='lower',
      extent=[-1.5,nfft+0.5,-1.5,nfft+0.5] )
   pg.title("diff (SVD)")
   pg.colorbar()
else:
   pg.text(0.1,0.1,"SVD failed")

# remnant variances
print("input var=",onePhaseD.var())
print("input-recon (MVM.) var=\t",(reconPhaseD[0]-onePhaseD).var())
if reconM[1] is not None:
   print("input-recon (No reg.) var=\t",(reconPhaseD[1]-onePhaseD).var())
if reconM[2] is not None:
   print("input-recon (SVD) var=\t",(reconPhaseD[2]-onePhaseD).var())

# waffle operator
waffleO=gradientOperator.waffleOperatorType1(pupilMask)
waffleV=waffleO.returnOp()
print("waffle input amp=\t",numpy.dot(onePhaseV, waffleV))
print("waffle recon (MVM.) amp=\t",numpy.dot(reconPhaseV[0], waffleV))
if not reconM[1] is None:
   print("waffle recon (No reg.) amp=\t",numpy.dot(reconPhaseV[1], waffleV))
if not reconM[2] is None:
   print("waffle recon (SVD) amp=\t",numpy.dot(reconPhaseV[2], waffleV))

for dat in (
      ("Mask creation","maskC"), ("Phase covariance","phaseC"),
      ("Eig Values time","eigV"),("Phase creation","phasev"),
      ("gTg","gtgM"), ("Inverse covariance matrix","invCovM"),
     ):
   print("{1}={0:5.3f}s".format(
      timings['e:'+dat[1]]-timings['s:'+dat[1]],dat[0]))

pg.waitforbuttonpress()
