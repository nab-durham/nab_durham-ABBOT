from __future__ import print_function
# What is this?
# Test variable pixel scaling on reconstruction
# Gradients or wavefronts

import abbot.projection as proj
import abbot.phaseCovariance as pc
import abbot.gradientOperator as gO
import Zernike
import matplotlib.pyplot as pg
import sys
import numpy

numpy.random.seed(18071977)
#numpy.random.seed(21111988)
#numpy.random.seed(19690305)
#numpy.random.seed(13111949)
#numpy.random.seed(19491105)

noiseTest=False#True
useGradients=True
r0=3 ; L0=100
recoveryClip=0.0
   # \/ simplified geometry
nAzi=3
pixScl=1
fund=16/pixScl
mask=Zernike.anyZernike(1,fund,fund/2,ongrid=1)\
      -Zernike.anyZernike(1,fund,fund/6,ongrid=1)

gradO=gO.gradientOperatorType1(pupilMask=mask)
gradM=gradO.returnOp()
tripleGradM=numpy.zeros(
      [gradM.shape[0]*nAzi, gradM.shape[1]*nAzi], numpy.float64)
for i in range(nAzi):
   tripleGradM[i*gradM.shape[0]:(i+1)*gradM.shape[0],
               i*gradM.shape[1]:(i+1)*gradM.shape[1]]=gradM
if useGradients:
   print("Using gradient version")
else:
   print("Using wavefront version")

layerNpix=int(fund*pixScl*2)
#gsHeight=None
geometry=proj.geometry(
   numpy.array([0,1]),
   numpy.array([fund*pixScl/2]*nAzi),
   numpy.arange(nAzi)*2*numpy.pi*(nAzi**-1.0), mask,
   None, pixScl, [[layerNpix]*2]*2  )
   # gsHeight )
geometry.define()
okay=geometry.createLayerMasks()
if not okay: raise ValueError("Eek!")

pg.figure()
pg.jet()
pg.subplot(2,2,1)
pg.title("layer masks")
pg.imshow( geometry.layerMasks[0].sum(axis=0), interpolation='nearest' )
pg.subplot(2,2,2)
pg.imshow( geometry.layerMasks[1].sum(axis=0), interpolation='nearest' )

   # \/ try projection
print("Projection matrix calcs...",end="")
layerExM=geometry.layerExtractionMatrix()
sumPrM=geometry.sumProjectedMatrix()
trimIdx=geometry.trimIdx()
if useGradients:
   sumLayerExM=numpy.dot(
         tripleGradM, numpy.dot( sumPrM, layerExM.take(trimIdx,axis=1) ) )
else:
   sumLayerExM=numpy.dot( sumPrM, layerExM.take(trimIdx,axis=1) )

print("(done)")

   # \/ random values as a substitute dataset
#zernidx=[ int(numpy.random.uniform(1,6)) for zn in range(geometry.nLayers) ]
#random=[ Zernike.anyZernike(zernidx[i],geometry.layerNpix[i][0])
#      for i in range(geometry.nLayers) ]
if not noiseTest:
   directPCOne=pc.covarianceDirectRegular( layerNpix, r0, L0 )
   directPC=pc.covarianceMatrixFillInRegular( directPCOne )
   directcholesky=pc.choleskyDecomp(directPC)
   random=[ numpy.dot( directcholesky, numpy.random.normal(size=layerNpix**2)
      ).reshape([layerNpix]*2) for i in range(geometry.nLayers) ]
   random=[
      ( random[i]-random[i].mean() )/( random[i]-random[i].mean() ).ravel().max()
      for i in range(geometry.nLayers) ]
   print("Input creation...",end="") ; sys.stdout.flush()
   randomV=(1*random[0].ravel()).tolist()+(1*random[1].ravel()).tolist()
   randomExV=numpy.take( randomV, trimIdx )
   randomProjV=numpy.dot( sumLayerExM, randomExV )
   if not useGradients:
      for i in range(nAzi):   # remove mean for each
         randomProjV[i*len(geometry.maskIdx):(i+1)*len(geometry.maskIdx)]-=\
               randomProjV[i*len(geometry.maskIdx):(i+1)*len(geometry.maskIdx)
               ].mean()
   print("(done)"); sys.stdout.flush()
else:
   # noisy data only
   import time
   if not useGradients:
      # more complicated, as we want to reconstruct the noise first
      print("Building w/f recon matrix...",end="") ; sys.stdout.flush()
      wfreconM=numpy.dot(
            numpy.linalg.pinv(
               numpy.dot( tripleGradM.transpose(), tripleGradM ) ),
            tripleGradM.transpose() )
      print("(done)") ; sys.stdout.flush()
      randomGradV=numpy.random.normal(size=wfreconM.shape[1])
      randomProjV=numpy.dot( wfreconM, randomGradV )
   else:
      randomProjV=numpy.random.normal(size=sumLayerExM.shape[0])
   random=[] ; offset=0
   for i in range(geometry.nLayers):
      random.append( numpy.zeros([layerNpix]*2, numpy.float64) )
      ti=geometry.maskInLayerIdx(
            i,geometry.layerMasks[i].sum(axis=0))-geometry.layerIdxOffsets()[i]
      random[-1].ravel()[ti]=randomProjV[offset:offset+len(ti)]
      offset+=len(ti)

   print("  >> Input is **JUST NOISE**"); sys.stdout.flush()
   time.sleep(0.5)
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

if useGradients:
#?   subapMask=(gradO.maskIdx>0)*1.0 # force to numbers
   subapMask=(gradO.subapMask>0)*1.0 # force to numbers
   projectedRdmVA=numpy.ma.masked_array(
         numpy.zeros([nAzi]+list(subapMask.shape),numpy.float64),
         (subapMask*numpy.ones([nAzi,1,1]))==0, astype=numpy.float64)
   subapMaskIdx=gradO.subapMaskIdx
#?   subapMaskIdx=numpy.flatnonzero((gradO.maskIdx.ravel()>0))
   projectedRdmVA.ravel()[
         (subapMaskIdx+(numpy.arange(0,nAzi)
          *subapMask.shape[0]*subapMask.shape[1]).reshape([-1,1])).ravel() ]\
         =randomProjV*numpy.ones(len(randomProjV))
else:
   projectedRdmVA=numpy.ma.masked_array(
         numpy.zeros([nAzi]+list(mask.shape),numpy.float64),
         (mask*numpy.ones([nAzi,1,1]))==0, astype=numpy.float64)
   projectedRdmVA.ravel()[
         (geometry.maskIdx+(numpy.arange(0,nAzi)
          *mask.shape[0]*mask.shape[1]).reshape([-1,1])).ravel() ]\
         =randomProjV*numpy.ones(len(randomProjV))

pg.figure()
for i in range(nAzi):
   pg.subplot(3,2,i+1)
   pg.imshow( projectedRdmVA[i,:,:], interpolation='nearest' )
   pg.title("projection #{0:1d}".format(i+1))
pg.xlabel("layer values")
pg.draw()


# now, try straight inversion onto the illuminated portions of the layers 
# with regularisation and via SVD
layerInsertionIdx=geometry.trimIdx(False)
#regularisationM=numpy.identity(layerInsertionIdx[-1][0])*-4
offset=0
regularisationM=numpy.zeros([layerInsertionIdx[-1][0]]*2)
for i in range(geometry.nLayers):
#?   tlO=gO.laplacianOperatorPupil( geometry.layerMasks[i].sum(axis=0) ).op
   tlO=gO.laplacianOperatorType1( pupilMask=geometry.layerMasks[i].sum(axis=0) )
   tlO=tlO.returnOp() # replace variable
   regularisationM[ offset:offset+tlO.shape[0],offset:offset+tlO.shape[0]]=tlO
   offset+=tlO.shape[0]
print("Inversion...",end=""); sys.stdout.flush()
sTs=numpy.dot(sumLayerExM.transpose(),sumLayerExM)
sTs_invSVD=numpy.linalg.inv( sTs +regularisationM*1e-4 )
print("...",end="");sys.stdout.flush()
recoveryM=numpy.dot( sTs_invSVD, sumLayerExM.transpose() )
# now, clip the reconstruction matrix
print("clipping (removing smallest, {0:2.0f}%)...".format(recoveryClip*100),
      end="")
recoveryAbs=abs(recoveryM).ravel()
recoveryM*=(abs(recoveryM)>recoveryClip*recoveryAbs.max())
print("(done)"); sys.stdout.flush()


print("Recon...",end="")
recoveredV=numpy.dot( recoveryM, randomProjV )
recoveredLayersA=[
   numpy.ma.masked_array(numpy.zeros(geometry.layerNpix[i], numpy.float64),
      geometry.layerMasks[i].sum(axis=0)==0) for i in range(geometry.nLayers) ]
layerInsertionIdx=geometry.trimIdx(False)
print("(done)"); sys.stdout.flush()

for i in range(geometry.nLayers):
   recoveredLayersA[i].ravel()[layerInsertionIdx[i][1]]=\
         recoveredV[layerInsertionIdx[i][0]:layerInsertionIdx[i+1][0]]
   if not noiseTest:
      pg.figure()
      pg.title("layer 1")
      pg.subplot(2,3,1+i*3)
      pg.imshow( recoveredLayersA[i]-recoveredLayersA[i].mean(),
         interpolation='nearest',vmin=-1,vmax=1 )
      pg.xlabel("recov'd")
      pg.subplot(2,3,2+i*3)
      pg.imshow( randomA[i]-randomA[i].mean(),
         interpolation='nearest',vmin=-1,vmax=1 )
      pg.xlabel("actual")
      pg.subplot(2,3,3+i*3)
      pg.imshow( recoveredLayersA[i]-randomA[i]-randomA[i].mean(),
         interpolation='nearest',vmin=-1,vmax=1 )
      pg.xlabel("diff")
   print(" Layer#{0:d}".format(i+1))
   print("  Original RMS={0:5.3f}".format(randomA[i].var()))
   print("  Difference RMS={0:5.3f}, {1:4.2f}%".format(
      (randomA[i]-recoveredLayersA[i]).var()**0.5,
      100*(randomA[i]-recoveredLayersA[i]).var()**0.5/randomA[i].var()**0.5))

pg.waitforbuttonpress()


