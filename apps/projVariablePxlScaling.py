from __future__ import print_function
# What is this?
# Test variable pixel scaling on reconstruction
# Gradients only

import abbot.projection as proj
import abbot.phaseCovariance as pc
import abbot.gradientOperator as gO
import Zernike
import matplotlib.pyplot as pyplot
import sys
import numpy

numpy.random.seed(18071977)

# ------------------------
# PERMISSIBLE CHANGES ARE IN THIS SECTION ONLY
r0,L0=3,100
pixScl=1
fund=16/pixScl
nLayers=2
#
# ------------------------

nAzi=3 # this code fixed to 3 directions
layerNpix=int(fund*pixScl*2)
randomIps=[ numpy.random.normal(size=layerNpix**2) for i in [None]*nLayers ]
mask=Zernike.anyZernike(1,fund,fund/2,ongrid=1)

gradO=gO.gradientOperatorType1(pupilMask=mask)
gradM=gradO.returnOp()
tripleGradM=numpy.zeros(
      [gradM.shape[0]*nAzi, gradM.shape[1]*nAzi], numpy.float64)
for i in range(nAzi):
   tripleGradM[i*gradM.shape[0]:(i+1)*gradM.shape[0],
               i*gradM.shape[1]:(i+1)*gradM.shape[1]]=gradM

print("Using gradient version")

geometry=proj.projection(
   range(nLayers),
   numpy.array([fund*pixScl/2]*nAzi),
   numpy.arange(nAzi)*2*numpy.pi*(nAzi**-1.0), mask,
   None, pixScl, [[layerNpix]*2]*2  )
geometry.define()

   # \/ show the layer masks
pyplot.figure()
pyplot.subplot(2,2,1)
pyplot.title("layer masks")
pyplot.imshow( geometry.layerMasks[0].sum(axis=0), interpolation='nearest' )
pyplot.subplot(2,2,2)
pyplot.imshow( geometry.layerMasks[1].sum(axis=0), interpolation='nearest' )

   # \/ try projection
print("Projection matrix calcs...",end="")
layerExM=geometry.layerExtractionMatrix()
sumPrM=geometry.sumProjectedMatrix()
trimIdx=geometry.trimIdx()
sumLayerExM=numpy.dot(
      tripleGradM, numpy.dot( sumPrM, layerExM.take(trimIdx,axis=1) ) )
print("(done)")

   # \/ realistic input
directPCOne=pc.covarianceDirectRegular( layerNpix, r0, L0 )
directPC=pc.covarianceMatrixFillInRegular( directPCOne )
directcholesky=pc.choleskyDecomp(directPC)
random=[ numpy.dot( directcholesky, randomIps[i] ).reshape([layerNpix]*2)
      for i in range(nLayers) ]
random=[
   ( random[i]-random[i].mean() )/( random[i]-random[i].mean() ).ravel().max()
      for i in range(nLayers) ]
print("Input creation...",end="") ; sys.stdout.flush()
randomV=(1*random[0].ravel()).tolist()+(1*random[1].ravel()).tolist()
randomExV=numpy.take( randomV, trimIdx )
randomProjV=numpy.dot( sumLayerExM, randomExV )
print("(done)"); sys.stdout.flush()
   # \/ just for imaging
randomA=[ numpy.ma.masked_array(random[i],
      geometry.layerMasks[i].sum(axis=0)==0) for i in (0,1) ]

   # \/ create an imagable per-projection array of the random values
sfcMask=numpy.zeros(geometry.layerNpix[0], numpy.float64)
sfcMIdx=[]
idx,fra=geometry.maskLayerCentreIdx(0)
for i in range(len(idx)):
   sfcMIdx+=idx[i].tolist()
   sfcMask.ravel()[idx[i]]+=fra[i]

subapMask=(gradO.subapMask>0)*1.0 # force to numbers
projectedRdmVA=numpy.ma.masked_array(
      numpy.zeros([nAzi]+list(subapMask.shape),numpy.float64),
      (subapMask*numpy.ones([nAzi,1,1]))==0, astype=numpy.float64)
subapMaskIdx=gradO.subapMaskIdx
projectedRdmVA.ravel()[
      (subapMaskIdx+(numpy.arange(0,nAzi)
       *subapMask.shape[0]*subapMask.shape[1]).reshape([-1,1])).ravel() ]\
      =randomProjV*numpy.ones(len(randomProjV))

pyplot.figure()
for i in range(nAzi):
   pyplot.subplot(3,2,i+1)
   pyplot.imshow( projectedRdmVA[i,:,:], interpolation='nearest' )
   pyplot.title("projection #{0:1d}".format(i+1))
pyplot.xlabel("layer values")
pyplot.draw()


# now, try straight inversion onto the illuminated portions of the layers 
# with regularisation and via SVD
layerInsertionIdx=geometry.trimIdx(False)
offset=0
regularisationM=numpy.zeros([layerInsertionIdx[-1][0]]*2)
for i in range(nLayers):
   tlO=gO.laplacianOperatorType1( pupilMask=geometry.layerMasks[i].sum(axis=0) )
   tlO=tlO.returnOp() # replace variable
   regularisationM[ offset:offset+tlO.shape[0],offset:offset+tlO.shape[0]]=tlO
   offset+=tlO.shape[0]
print("Inversion...",end=""); sys.stdout.flush()
sTs=numpy.dot(sumLayerExM.transpose(),sumLayerExM)
sTs_invSVD=numpy.linalg.inv( sTs +regularisationM*1e-4 )
print("...",end="");sys.stdout.flush()
recoveryM=numpy.dot( sTs_invSVD, sumLayerExM.transpose() )


print("Recon...",end="")
recoveredV=numpy.dot( recoveryM, randomProjV )
recoveredLayersA=[
   numpy.ma.masked_array(numpy.zeros(geometry.layerNpix[i], numpy.float64),
      geometry.layerMasks[i].sum(axis=0)==0) for i in range(nLayers) ]
layerInsertionIdx=geometry.trimIdx(False)
print("(done)"); sys.stdout.flush()

for i in range(nLayers):
   recoveredLayersA[i].ravel()[layerInsertionIdx[i][1]]=\
         recoveredV[layerInsertionIdx[i][0]:layerInsertionIdx[i+1][0]]
   pyplot.figure()
   pyplot.title("layer 1")
   pyplot.subplot(2,3,1+i*3)
   pyplot.imshow( recoveredLayersA[i]-recoveredLayersA[i].mean(),
      interpolation='nearest',vmin=-1,vmax=1 )
   pyplot.xlabel("recov'd")
   pyplot.subplot(2,3,2+i*3)
   pyplot.imshow( randomA[i]-randomA[i].mean(),
      interpolation='nearest',vmin=-1,vmax=1 )
   pyplot.xlabel("actual")
   pyplot.subplot(2,3,3+i*3)
   pyplot.imshow( recoveredLayersA[i]-randomA[i]-randomA[i].mean(),
      interpolation='nearest',vmin=-1,vmax=1 )
   pyplot.xlabel("diff")
   print(" Layer#{0:d}".format(i+1))
   print("  Original RMS={0:5.3f}".format(randomA[i].var()))
   print("  Difference RMS={0:5.3f}, {1:4.2f}%".format(
      (randomA[i]-recoveredLayersA[i]).var()**0.5,
      100*(randomA[i]-recoveredLayersA[i]).var()**0.5/randomA[i].var()**0.5))

pyplot.waitforbuttonpress()

