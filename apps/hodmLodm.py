from __future__ import print_function
# What is this?
# High-order/low-order separation

import numpy
import Zernike
import dm
import matplotlib.pyplot as pg
import commonSeed
import gradientOperator
import phaseCovariance

nfft=30

mask=(Zernike.anyZernike(1,nfft,nfft/2,ongrid=1)
   -Zernike.anyZernike(1,nfft,nfft/8,ongrid=1))
mask=mask.astype(numpy.int32)
nMask=int(mask.sum())

gO=gradientOperator.gradientOperatorType1( mask )
gM=gO.returnOp()

hodm=dm.dm(gO.n_,[16]*2)
lodm=dm.dm(gO.n_,[4]*2)

hodmPokeM=numpy.zeros([2*gO.numberSubaps,hodm.nacts],numpy.float64)
lodmPokeM=numpy.zeros([2*gO.numberSubaps,lodm.nacts],numpy.float64)
for i in range(hodm.nacts):
   hodmPokeM[:,i]=numpy.dot( gM, hodm.poke(i).take(gO.illuminatedCornersIdx) )
for i in range(lodm.nacts):
   lodmPokeM[:,i]=numpy.dot( gM, lodm.poke(i).take(gO.illuminatedCornersIdx) )

#hodmreconM=numpy.linalg.pinv(hodmPokeM)
#lodmreconM=numpy.linalg.pinv(lodmPokeM)
stackPokeM=numpy.zeros([2*gO.numberSubaps,hodm.nacts+lodm.nacts],numpy.float64)
stackPokeM[:,:hodm.nacts]=hodmPokeM
stackPokeM[:,hodm.nacts:]=lodmPokeM


print("making unregularised reconM...",end="")
sTsM=numpy.dot( stackPokeM.transpose(), stackPokeM )
print(".",end="")
reconM=numpy.linalg.pinv( sTsM )
print(".",end="")
reconM=numpy.dot( reconM, stackPokeM.transpose() )
print("/")

lambd=0.1
selecM=numpy.zeros([hodm.nacts,hodm.nacts+lodm.nacts],numpy.float64)
selecM.ravel()[ numpy.arange(hodm.nacts)*(hodm.nacts+lodm.nacts+1) ]=1
seTseM=numpy.dot( selecM.transpose(), selecM )
print(".",end="")
reconBM=numpy.linalg.pinv( sTsM+lambd*seTseM )
print(".",end="")
reconBM=numpy.dot( reconBM, stackPokeM.transpose() )

thisReconM=reconBM # choose which to use

# generate some test data
phaseMask=(gO.illuminatedCorners>0)
r0=nfft/4
L0=1e3
rdm=numpy.random.normal(size=gO.numberPhases)

directPCOne=phaseCovariance.covarianceDirectRegular( nfft+1, r0, L0 )
directPC=phaseCovariance.covarianceMatrixFillInMasked(
   directPCOne, phaseMask )
directcholesky=phaseCovariance.choleskyDecomp(directPC)
directTestPhase=phaseCovariance.numpy.dot(directcholesky, rdm)
testPhase2dI=numpy.zeros(gO.n_, numpy.float64)
testPhase2dI.ravel()[gO.illuminatedCornersIdx]=directTestPhase
testPhase2dI=numpy.ma.masked_array( testPhase2dI, [phaseMask==0] )

slopes=numpy.dot( gM, directTestPhase )

actuatorV=numpy.dot( thisReconM, slopes ) # this is for both mirrors, so split
horeconPhaseV=numpy.zeros([gO.numberPhases],numpy.float64)
loreconPhaseV=numpy.zeros([gO.numberPhases],numpy.float64)
for i in range(hodm.nacts):
   horeconPhaseV+=( hodm.poke(i)*actuatorV[i] ).take(gO.illuminatedCornersIdx)
for i in range(lodm.nacts):
   loreconPhaseV+=( lodm.poke(i)*actuatorV[hodm.nacts+i] ).take(gO.illuminatedCornersIdx)

horecon2dI=numpy.zeros(gO.n_, numpy.float64)
horecon2dI.ravel()[gO.illuminatedCornersIdx]=horeconPhaseV
horecon2dI=numpy.ma.masked_array( horecon2dI, [phaseMask==0] )
lorecon2dI=numpy.zeros(gO.n_, numpy.float64)
lorecon2dI.ravel()[gO.illuminatedCornersIdx]=loreconPhaseV
lorecon2dI=numpy.ma.masked_array( lorecon2dI, [phaseMask==0] )

testPhase2dI-=testPhase2dI.mean()
horecon2dI-=horecon2dI.mean()
lorecon2dI-=lorecon2dI.mean()
phsRge=[ (testPhase2dI.max()), (testPhase2dI.min()) ]
pg.subplot(2,2,1) ; pg.imshow( testPhase2dI, vmin=phsRge[1],vmax=phsRge[0] ) ; pg.title("orig") ; pg.colorbar()
pg.subplot(2,2,2) ; pg.imshow( horecon2dI, vmin=phsRge[1],vmax=phsRge[0] ) ; pg.title("ho")
pg.subplot(2,2,3) ; pg.imshow( lorecon2dI, vmin=phsRge[1],vmax=phsRge[0] ) ; pg.title("lo")
pg.subplot(2,2,4) ; pg.imshow( lorecon2dI+horecon2dI-testPhase2dI ) ; pg.title("(ho+lo)-orig") ; pg.colorbar()

