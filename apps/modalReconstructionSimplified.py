# -*- coding: utf-8 -*-
# 
from __future__ import print_function
# What is this?
# Test modal reconstruction vs. zonal, minimum code

import numpy
import abbot.gradientOperator
import abbot.modalBasis as MB

cds=lambda N : numpy.arange(N)-(N-1)/2.0
circle=lambda N,r : numpy.where(
   numpy.add.outer( cds(N[0])**2.0, cds(N[1])**2.0 ) <= r**2.0,
      1, 
      0 )

# \/ --------- variables start -----------
baseSize=20
##clipModalFunctions=False
# /\ --------- variables end -------------

numpy.random.seed(18071977)
mask=circle([baseSize,baseSize],baseSize/2)
mask-=circle([baseSize,baseSize],4)
mask[:baseSize/2,baseSize/2-1:baseSize/2+1]=0
mask=mask.astype('i')
gradOp=abbot.gradientOperator.gradientOperatorType1(pupilMask=mask)
gradM=gradOp.returnOp()

### \/ Generate reconstructors
# embFs are the modal basis (*not* Zernikes)
# clippedmbFs are a version of the above...perhaps constricted (hence clipped) but could be == 
# MBinteractM is the modal interaction matrix 
# MBreconM is the modal reconstruction matrix
# ZreconM is the zonal reconstruction matrix
#
explicitModalBasis=MB.polySinRadAziBasisType1(
      mask, range(19),range(19), orthonormalize=0 ) # setup modal basis generator
embFs=explicitModalBasis.modalFunctions # alias to realized modal bases
##if clipModalFunctions:
##   u,s,v=numpy.linalg.svd( embFs, full_matrices=0 )
##   embFs_limit=max( len( numpy.flatnonzero( s ) ),
##      gradOp.numberPhases )
##   print("explicitModalBasis truncated to: {0:d}/{1:d}".format(
##         embFs_limit,u.shape[0]))
##   clippedmbFs=u[:embFs_limit].dot(v*s.reshape([-1,1])).T
##else:
if 1==1: # temporary to allow for the if...else structure that's been commented
         # out above
   clippedmbFs=embFs.T
MBinteractM=gradM.dot(clippedmbFs) # the interaction matrix of the modal basis
MBreconM=clippedmbFs.dot( numpy.linalg.pinv( MBinteractM, 1e-3 ) ) # reconstruction matrix
#
ZreconM=numpy.linalg.pinv(gradM,1e-6) # reconstruction, zonal

### \/ Input data, for calculation and for comparison
# thisData is the normalized (peak-to-peak=1) zero-mean input (phase?)
# inputDataV is the vector version of the input
# inputDataA is the 2D, masked version of the input, for visualization
# gradsV is the gradients of the input vector, and so what we want to reconstruct from
#
print("Input data...",end="")
thisData=numpy.fft.fft2( numpy.random.normal(size=[x*2 for x in gradOp.n_])*
         numpy.fft.fftshift(numpy.add.outer(
            (1e-1+numpy.arange(-gradOp.n_[0],gradOp.n_[0]))**(2.),
            (1e-1+numpy.arange(-gradOp.n_[1],gradOp.n_[1]))**(2.) )**(-0.6) )
      ).real[:gradOp.n_[0],:gradOp.n_[1]]

##thisData=thisData[:-2] -thisData[1:-1] # differential of screen...
##thisData=thisData[:-2] # full screen...
thisData-=thisData.mean() # ...remove mean...
thisData/=(thisData.max()-thisData.min()) # ...and normalize.
inputDataV=thisData.ravel()[gradOp.illuminatedCornersIdx] # for calculations
inputDataA=numpy.ma.masked_array(
   numpy.empty(gradOp.n_), gradOp.illuminatedCorners==0 ) # for visualization
inputDataA.ravel()[gradOp.illuminatedCornersIdx]=inputDataV
print("(done)")
gradsV=numpy.dot( gradM, inputDataV ) # calculate input vector
print("All data prepared")


### \/ Reconstruction
# reconstructionZonalV is the zonally reconstructed vector, to be compared to inputDataV
# reconstructionModalV is the modally reconstructed vector
# 
# \/ Zonal method
reconstructionZonalV=(ZreconM).dot(gradsV) # reconstruction
reconOrigA=inputDataA.copy()*0 # setup masked array for visualization
reconOrigA.ravel()[gradOp.illuminatedCornersIdx]=reconstructionZonalV # copy into masked array

# \/ Modal method
reconstructionModalV=(MBreconM).dot(gradsV) # reconstruction c. piston
reconstructionModalV-=reconstructionModalV.mean() # ...then remove piston
reconModalA=inputDataA.copy()*0
reconModalA.ravel()[gradOp.illuminatedCornersIdx]=reconstructionModalV


### \/ Plot comparisons
# reconstructionZonalV is the zonally reconstructed vector, to be compared to inputDataV
# reconstructionModalV is the modally reconstructed vector
# 
from matplotlib import pyplot 
pyplot.figure(1,figsize=(10,5))
#
pyplot.subplot(2,3,1)
pyplot.imshow(inputDataA)
pyplot.colorbar()
pyplot.title("i/p")
#
pyplot.subplot(2,3,2)
pyplot.imshow(reconModalA,vmax=inputDataA.max(),vmin=inputDataA.min())
pyplot.title("recon,modal")
pyplot.subplot(2,3,3)
pyplot.imshow(inputDataA-reconModalA,
   vmax=inputDataA.max(),vmin=inputDataA.min())
pyplot.colorbar()
pyplot.title("(delta)")

pyplot.subplot(2,3,2+3)
pyplot.imshow(reconOrigA,vmax=inputDataA.max(),vmin=inputDataA.min())
pyplot.title("recon,zonal")
pyplot.subplot(2,3,3+3)
pyplot.imshow(inputDataA-reconOrigA,
   vmax=inputDataA.max(),vmin=inputDataA.min())
pyplot.colorbar()
pyplot.title("(delta)")

[ pyplot.tight_layout() ]*2 # do this twice

# \/ State the fraction of variance unexplained (or, the normalized residual variance)
print("(input-recon_{{zonal}}).var()/input.var()={0:5.3f}".format(
  (inputDataV-reconstructionZonalV).var()/inputDataV.var() ))
print("(input-recon_{{modal}}).var()/input.var()={0:5.3f}".format(
  (inputDataV-reconstructionModalV).var()/inputDataV.var() ))
