# What is this?
# Test regularisation, with a Type 3 geometry

import matplotlib.pyplot as pg
import numpy
import gradientOperator
import commonSeed 

# test code
r0=1 # in pixels
L0=1000
nfft=13 # pixel size

# define pupil mask as sub-apertures
pupilMask=numpy.ones([nfft]*2,numpy.int32)
pupilCds=numpy.add.outer(
   (numpy.arange(nfft)-(nfft-1)/2.)**2.0,
   (numpy.arange(nfft)-(nfft-1)/2.)**2.0 )
#pupilMask=(pupilCds>(nfft/6)**2)*(pupilCds<(nfft/2)**2)
#pupilMask=(pupilCds<(nfft/2)**2)
gO=gradientOperator.gradientOperatorType3Centred(pupilMask)
gM=gO.returnOp()

# define phase at pixels themselves, each of which is a sub-aperture
import phaseCovariance as pc
cov=pc.covarianceMatrixFillInRegular(
   pc.covarianceDirectRegular(nfft+2,r0,L0) )
choleskyC=pc.choleskyDecomp(cov)

# generate phase and gradients
onePhase=numpy.dot( choleskyC, numpy.random.normal(size=(nfft+2)**2 ) )
onePhaseTrueV=onePhase[gO.allCentresIdx]
onePhase-=onePhaseTrueV.mean() # normalise
onePhase.resize([nfft+2]*2)
onePhaseD=numpy.ma.masked_array(onePhase,gO.illuminatedCentres==0)

onePhaseV=onePhase.ravel()[gO.allCentresIdx]
gradV=numpy.dot(gM,onePhaseV)

# now try solving
# linear least squares is (G^T G+alpha I+beta R)^-1 G^T
# alpha!=0 with Tikhonov regularisation
# alpha=0 with no regularisation
# or use SVD to quash poorly spanned eigenvectors of phase space
#lO=gradientOperator.laplacianOperatorType3Centred(pupilMask)
#lM=lO.returnOp() # laplacian operator
gTg=numpy.dot( gM.transpose(), gM )

try:
   # \/ no regularisation 
   invgTgM=numpy.linalg.inv(gTg)
except numpy.linalg.LinAlgError as exceptionErr:
   print("FAILURE: Inversion didn't work ,{0:s}".format(exceptionErr))
   invgTgM=None

   # \/ Tikhonov reconstruction
eigV=numpy.linalg.eigvals(gTg) ; eigV.sort()
alpha=eigV[-1]*1e-3 # quash about largest eigenvalue x factor
#beta=alpha**2.0
invgTgTikM=numpy.linalg.inv(
   gTg + alpha*numpy.identity(gO.numberPhases) ).real # force real
#       + beta*numpy.dot(lM.transpose(),lM) )

try:
   # \/ SVD reconstruction
   invgTgSVDM=numpy.linalg.pinv(gTg,rcond=1e-3*eigV[-1])
except numpy.linalg.LinAlgError as exceptionErr:
   print("FAILURE: SVD didn't converge,{0:s}".format(exceptionErr))
   invgTgSVDM=None

## explicitly
#svd=numpy.linalg.svd(gTg)
#svdMask=svd[1]>(svd[1][0]*1e-6)
#sinv=numpy.dot( (svd[2].transpose()*(svd**-1*svdMask).reshape([1,-1])),svd[0].transpose() )
#reconExpSVDM=numpy.dot(sinv,gM.transpose())

# reconstructor matrices
reconM=[]
for thisInvM in (invgTgTikM, invgTgM, invgTgSVDM):
   if thisInvM!=None:
      reconM.append( numpy.dot( thisInvM, gM.transpose() ) )
   else: 
      reconM.append( None )
      

reconPhaseV=[]
for thisRM in reconM:
   if thisRM!=None:
      reconPhaseV.append( numpy.dot( thisRM, gradV ) )
   else:      
      reconPhaseV.append(None)

# imaging of phases
reconPhaseD=[] 
# reconPhaseD[:,gO.illuminatedCentresIdx]=gO.constrictPhase(reconPhaseV)
for i in range(3):
   if reconM[i]==None: 
      reconPhaseD.append(None)
   else:
      thisPhaseD=numpy.zeros((nfft+2)**2,numpy.float64)
      thisPhaseD[gO.allCentresIdx]=reconPhaseV[i]
      reconPhaseD.append( numpy.ma.masked_array(
         thisPhaseD.reshape([nfft+2]*2), [gO.maskIdx<1] ) )
 
pg.figure(2)
pg.subplot(221)
pg.imshow( onePhaseD, interpolation='nearest', origin='lower',
   extent=[-1.5,nfft+0.5,-1.5,nfft+0.5] )
pg.title("type 3: input phase")
pg.colorbar()
pg.subplot(222)
pg.imshow( reconPhaseD[0], interpolation='nearest', origin='lower',
   extent=[-1.5,nfft+0.5,-1.5,nfft+0.5] )
pg.title("reconstructed phase (Tik reg)")
pg.colorbar()
pg.subplot(234)
pg.imshow( reconPhaseD[0]-onePhaseD, interpolation='nearest', origin='lower',
   extent=[-1.5,nfft+0.5,-1.5,nfft+0.5] )
pg.title("diff (Tik.)")
pg.colorbar()
pg.subplot(235)
if reconM[1]!=None:
   pg.imshow( reconPhaseD[1]-onePhaseD, interpolation='nearest', origin='lower',
      extent=[-1.5,nfft+0.5,-1.5,nfft+0.5] )
   pg.title("diff (No reg.)")
   pg.colorbar()
else:
   pg.text(0.1,0.1,"Inv failed")
pg.subplot(236)
if reconM[2]!=None:
   pg.imshow( reconPhaseD[2]-onePhaseD, interpolation='nearest', origin='lower',
      extent=[-1.5,nfft+0.5,-1.5,nfft+0.5] )
   pg.title("diff (SVD)")
   pg.colorbar()
else:
   pg.text(0.1,0.1,"SVD failed")

# remnant variances
print("input var=",onePhaseD.var())
print("input-recon (Tik.) var=\t",(reconPhaseD[0]-onePhaseD).var())
if reconM[1]!=None:
   print("input-recon (No reg.) var=\t",(reconPhaseD[1]-onePhaseD).var())
if reconM[2]!=None:
   print("input-recon (SVD) var=\t",(reconPhaseD[2]-onePhaseD).var())

#[ # waffle operator
#[ waffleO=gradientOperator.waffleOperatorType1(pupilMask)
#[ waffleV=waffleO.returnOp()
#[ print("waffle input amp=\t",numpy.dot(onePhaseV, waffleV))
#[ print("waffle recon (Tik.) amp=\t",numpy.dot(reconPhaseV[0], waffleV))
#[ print("waffle recon (No reg.) amp=\t",numpy.dot(reconPhaseV[1], waffleV))
#[ print("waffle recon (SVD) amp=\t",numpy.dot(reconPhaseV[2], waffleV))
