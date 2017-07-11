from __future__ import print_function
# What is this?
# Test modal basis filtering and split modal reconstruction

import numpy
import Zernike
import abbot.gradientOperator
import abbot.modalBasis as MB
import matplotlib.pyplot as pyp
import commonSeed
import sys
import kolmogorov

import time
numpy.random.seed(int(time.time()%1234))

baseSize=16
clipModalFunctions=False
#
# naive means Tikhonov
# intermediate means 'intermediate layer restriction'
# laplacian means Laplacian approximation
# SVD means avoid direct inversion
regularizationType=['intermediate','laplacian','SVD','naive'][1]
modalPowers={'r':[1,2],'ang':[1]}


mask=Zernike.anyZernike(1,baseSize,baseSize/2,ongrid=1)\
      -Zernike.anyZernike(1,baseSize,baseSize/2/7.0,ongrid=1)
mask[baseSize/2]=0   # /
mask[:,baseSize/2]=0 # \ crude spider
mask=mask.astype(numpy.int32)
nMask=int(mask.sum())

gradOp=abbot.gradientOperator.gradientOperatorType1(pupilMask=mask)
gradM=gradOp.returnOp()

modalBasis=MB.polySinRadAziBasisType1( mask,
      [],[], orthonormalize=0, verbose=1 )
#mBs=modalBasis.orthomodalFunctions
mBidxs=[(1, 1, 1), (1, 1, 0), (1, 2, 1), (1, 3, 1), (1, 3, 0)]
mBs=numpy.array([
   modalBasis.modalFunction( tmBidx[0],tmBidx[1],tmBidx[2] )
   for tmBidx in mBidxs ])
   
#modalFiltering=[ 
#      thismodalB.reshape([-1,1]).dot(thismodalB.reshape([1,-1]))
#         for thismodalB in modalBasis.modalFunctions ]
#modalFilteringSummed=(
#      numpy.identity(nMask)-numpy.array(modalFiltering).sum(axis=0) )
modalFilterM=mBs.T.dot(mBs)

# (f) fourierModalBasis=MB.FourierModalBasisType1( mask )
# (f) mbOFs=fourierModalBasis.modalFunctions.T
# (f) u,s,v=numpy.linalg.svd(fourierModalBasis.modalFunctions,full_matrices=0)
# (f) clippedmbFs=u[:gradOp.numberPhases].dot(v*s.reshape([-1,1])).T
# (F) fmbinteractM=gradM.dot(fmbOFs)

explicitModalBasis=MB.polySinRadAziBasisType1( mask,
      range(19),range(19), orthonormalize=0, verbose=1 )
embFs=explicitModalBasis.modalFunctions
if clipModalFunctions:
   u,s,v=numpy.linalg.svd( embFs, full_matrices=0 )
   embFs_limit=max( len( numpy.flatnonzero( s ) ),
      gradOp.numberPhases )
   print("explicitModalBasis truncated to: {0:d}/{1:d}".format(
         embFs_limit,u.shape[0]))
   clippedmbFs=u[:embFs_limit].dot(v*s.reshape([-1,1])).T
else:
   clippedmbFs=embFs.T
fmbinteractM=gradM.dot(clippedmbFs)
fmbreconM=clippedmbFs.dot( numpy.linalg.pinv( fmbinteractM, 1e-3 ) )

gradMplus=numpy.linalg.pinv(gradM,1e-6)
gradmM=gradM.dot( modalFilterM )
gradmMplus=numpy.linalg.pinv(gradmM,1e-6)
#gradzM=gradM.dot( numpy.identity(nMask)-modalFilterM )
gradzM=gradM-gradmM
#gradzMplus=numpy.linalg.pinv(gradzM,1e-1)
gradzMplus=gradMplus
gradzMplus=(numpy.identity(nMask)-modalFilterM).dot(gradzMplus)

print("Input data...",end="") ; sys.stdout.flush()

thisData=kolmogorov.TwoScreens(baseSize*2,
            (nMask**0.5)/2.0,
            flattening=2*numpy.pi/baseSize/2.0*1e4)[0][:baseSize+2,:baseSize]
# [tilt] thisData=numpy.add.outer( numpy.arange(baseSize+2), numpy.arange(baseSize)*0
# [tilt]       ).astype('f')
thisData=thisData[:-2] -thisData[1:-1] # represent shifted screen
# thisData=thisData[:-2] # full
thisData-=thisData.mean()
thisData/=(thisData.max()-thisData.min())
# (O)    # \/ add a fake offset to the bottom left quadrant
# (O) thisData[:baseSize/2,:baseSize/2]+=1
inputDataV=thisData.ravel()[gradOp.illuminatedCornersIdx] 
# (m),inputDataV=numpy.array(
# (m),   [ numpy.random.normal()*tb for tb in modalBasis.modalFunctions ]).sum(axis=0)

inputDataA=numpy.ma.masked_array(
   numpy.empty(gradOp.n_), gradOp.illuminatedCorners==0 )
inputDataA.ravel()[gradOp.illuminatedCornersIdx]=inputDataV

print("(done)") ; sys.stdout.flush()

# calculate input vector
gradsV=numpy.dot( gradM, inputDataV )
            
print("All data prepared") ; sys.stdout.flush()

reconFilteredV=gradzMplus.dot(gradsV)
reconModalV=gradmMplus.dot(gradsV)
reconFourierModalV=(fmbreconM).dot(gradsV)
reconFourierModalV-=reconFourierModalV.mean()
reconJointV=(gradzMplus+gradmMplus).dot(gradsV)
reconOrigV=(gradMplus).dot(gradsV)
reconFilteredA=inputDataA.copy()*0
reconModalA=inputDataA.copy()*0
reconFourierModalA=inputDataA.copy()*0
reconJointA=inputDataA.copy()*0
reconOrigA=inputDataA.copy()*0
reconFilteredA.ravel()[gradOp.illuminatedCornersIdx]=reconFilteredV
reconModalA.ravel()[gradOp.illuminatedCornersIdx]=reconModalV
reconFourierModalA.ravel()[gradOp.illuminatedCornersIdx]=reconFourierModalV
reconJointA.ravel()[gradOp.illuminatedCornersIdx]=reconJointV
reconOrigA.ravel()[gradOp.illuminatedCornersIdx]=reconOrigV

pyp.subplot(5,3,1)
pyp.imshow(inputDataA)
pyp.colorbar()
pyp.title("i/p")
#
pyp.subplot(5,3,2)
pyp.imshow(reconFilteredA,vmax=inputDataA.max(),vmin=inputDataA.min())
pyp.title("recon,less Z")
#
pyp.subplot(5,3,3)
pyp.imshow(inputDataA-reconFilteredA,vmax=inputDataA.max()*0.1,vmin=inputDataA.min())
pyp.colorbar()
pyp.title("(delta)")

pyp.title("i/p")
pyp.subplot(5,3,2+3)
pyp.imshow(reconModalA,vmax=inputDataA.max(),vmin=inputDataA.min())
pyp.title("recon,Z only")
pyp.subplot(5,3,3+3)
pyp.imshow(inputDataA-reconModalA,vmax=inputDataA.max(),vmin=inputDataA.min())
pyp.colorbar()
pyp.title("(delta)")

pyp.title("i/p")
pyp.subplot(5,3,2+6)
pyp.imshow(reconJointA,vmax=inputDataA.max(),vmin=inputDataA.min())
pyp.title("recon,joint")
pyp.subplot(5,3,3+6)
pyp.imshow(inputDataA-reconJointA,
   vmax=inputDataA.max(),vmin=inputDataA.min())
pyp.colorbar()
pyp.title("(delta)")

pyp.title("i/p")
pyp.subplot(5,3,2+9)
pyp.imshow(reconFourierModalA,vmax=inputDataA.max(),vmin=inputDataA.min())
pyp.title("recon,Fourier modal")
pyp.subplot(5,3,3+9)
pyp.imshow(inputDataA-reconFourierModalA,
   vmax=inputDataA.max(),vmin=inputDataA.min())
pyp.colorbar()
pyp.title("(delta)")

pyp.title("i/p")
pyp.subplot(5,3,2+12)
pyp.imshow(reconOrigA,vmax=inputDataA.max(),vmin=inputDataA.min())
pyp.title("recon,orig")
pyp.subplot(5,3,3+12)
pyp.imshow(inputDataA-reconOrigA,
   vmax=inputDataA.max(),vmin=inputDataA.min())
pyp.colorbar()
pyp.title("(delta)")

for x in pyp.gcf().get_axes(): x.xaxis.set_visible(0)

print("(input-recon_{{standard/original}}).var()/input.var()={0:5.3f}".format(
  (inputDataV-reconOrigV).var()/inputDataV.var() ))
print("(input-recon_{{filtered}}).var()/input.var()={0:5.3f}".format(
  (inputDataV-reconFilteredV).var()/inputDataV.var() ))
print("(input-recon_{{modal}}).var()/input.var()={0:5.3f}".format(
  (inputDataV-reconModalV).var()/inputDataV.var() ))
print("(input-recon_{{Fourier modal}}).var()/input.var()={0:5.3f}".format(
  (inputDataV-reconFourierModalV).var()/inputDataV.var() ))
print("(input-recon_{{joint}}).var()/input.var()={0:5.3f}".format(
  (inputDataV-reconJointV).var()/inputDataV.var() ))
