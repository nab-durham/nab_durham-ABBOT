from __future__ import print_function
# What is this?
# Test projection matrices over N layers, using a different reconstructor.
#
# This version adds regularisation to constrain layer solution variance, to add
# a priori information and modal filtering

import numpy
import Zernike
import projection
import gradientOperator
import matplotlib.pyplot as pyp
import commonSeed
import sys

nAzi=4
baseSize=8
za=15/20.0e3
dH=2e3
Hmax=6e3
#
testZernikes=False
#}sameGeometry=True # always assume True in this code
#
useGrads=True
#
# naive means Tikhonov
# intermediate means 'intermediate layer restriction'
# laplacian means Laplacian approximation
# SVD means avoid direct inversion
simple=['intermediate','laplacian','SVD','naive'][1]
modalPowers={'r':[1,2],'ang':[1]}



mask=Zernike.anyZernike(1,baseSize,baseSize/2,ongrid=1)\
      -Zernike.anyZernike(1,baseSize,baseSize/2/7.0,ongrid=1)
mask=mask.astype(numpy.int32)
nMask=int(mask.sum())

if useGrads:
   gradOp=gradientOperator.gradientOperatorType1(pupilMask=mask)
   gradM=gradOp.returnOp()
   gradAllM=numpy.zeros(
         [gradM.shape[0]*nAzi, gradM.shape[1]*nAzi], numpy.float64)
   for i in range(nAzi): # nAzi directions gradient operator
      gradAllM[gradM.shape[0]*i:gradM.shape[0]*(i+1),
               gradM.shape[1]*i:gradM.shape[1]*(i+1)]=gradM

# angle, 5/20e3 is ~50'' which is ~CANARY 1
# Separation of one sub-ap between regular spacing is at alt,
# |zenith_angle*h*(1,0)-(cos(azi),sin(azi))|=zenith_angle*h*((1-cos)**2.0+sin**2.0)
#=za*h*(2-2cos(azi)), azi=2pi/N
aa=2*numpy.pi*(nAzi**-1.0)
#dH=( za*(2-2*numpy.cos(aa)) )**-1.0*1.0

reconGeometry=projection.projection(
      numpy.ceil( numpy.arange(numpy.ceil((Hmax/dH)))*dH ),
      [za]*nAzi, numpy.arange(nAzi)*aa, mask )
modalBasis=gradientOperator.modalBasis( mask, [1,2],[1,2] )
modalFiltering=[ numpy.identity(nMask)-
      thismodalB.reshape([-1,1]).dot(thismodalB.reshape([1,-1]))
         for thismodalB in modalBasis.orthomodalFunctions ]

if not reconGeometry.createLayerMasks(): raise ValueError("Eek! (1)")

#}if sameGeometry:
actualGeometry=reconGeometry
print("NOTE: Same geometry assumed")
#}else:
#}   actualGeometry=projection.projection(
#}    [0]+numpy.sort(
#}       numpy.random.uniform(1e3,15e3,size=5) ).astype('i').tolist(),
#}    [za]*nAzi, numpy.arange(nAzi)*aa, mask )
#}
#}okay=actualGeometry.createLayerMasks()
#}if not okay:
#}   raise ValueError("Eek! (2)")

# projection matrices
layerExM=reconGeometry.layerExtractionMatrix()
sumPrM=reconGeometry.sumProjectedMatrix()
reconTrimIdx=reconGeometry.trimIdx()
sumLayerExM=numpy.dot( sumPrM, layerExM.take(reconTrimIdx,axis=1) )

if useGrads:
   sumLayerExM=numpy.dot( gradAllM, sumLayerExM )
   print("Including gradient operator") ; sys.stdout.flush()

actualLayerExM,actualSumPrM,actualTrimIdx,actualSumLayerExM=\
      layerExM,sumPrM,reconTrimIdx,sumLayerExM
#}actualLayerExM=actualGeometry.layerExtractionMatrix()
#}actualSumPrM=actualGeometry.sumProjectedMatrix()
#}actualTrimIdx=actualGeometry.trimIdx()
#}actualSumLayerExM=numpy.dot(
#}   actualSumPrM, actualLayerExM.take(actualTrimIdx,axis=1) )


# projection of the actual data
weights=(numpy.random.uniform(0,1,size=100)).tolist()

import kolmogorov
print("Input data...",end="") ; sys.stdout.flush()

inputData=[]
inputDataV=[]
for i in range(actualGeometry.nLayers):
   tS=actualGeometry.layerNpix[i]
   thisData=kolmogorov.TwoScreens(tS.max()*2,
            (nMask**0.5)/2.0)[0][:tS[0],:tS[1]]
   inputData.append(
         2*(thisData-thisData.mean())/(thisData.max()-thisData.min()) )
   inputData[i]*=weights[i]
   inputDataV+=inputData[i].ravel().tolist() # vectorize # vectorize # vectorize # vectorize

inputDataA=[
   numpy.ma.masked_array(inputData[i],
      actualGeometry.layerMasks[i].sum(axis=0)==0)
         for i in range(actualGeometry.nLayers) ]

print("(done)") ; sys.stdout.flush()

# calculate input vector
randomExV=numpy.take( inputDataV, actualTrimIdx )
randomProjV=numpy.dot( actualSumLayerExM, randomExV )
if not useGrads:
   # remove mean from each input, independently
   for i in range(actualGeometry.nAzi):
      randomProjV[i*nMask:(i+1)*nMask]-=randomProjV[i*nMask:(i+1)*nMask].mean()
            
print("All data prepared") ; sys.stdout.flush()

# now, try straight inversion onto the illuminated portions of the layers 
sTs=numpy.dot(sumLayerExM.transpose(),sumLayerExM)

layerInsertionIdx=reconGeometry.trimIdx(False)
regularisationM=numpy.zeros([layerInsertionIdx[-1][0]]*2)

if simple=='naive':
   # \/ Tikhonov regularisation
   layerInsertionIdx=reconGeometry.trimIdx(False)
   for i in range(reconGeometry.nLayers):
      diagIdx=numpy.arange( layerInsertionIdx[i][0],layerInsertionIdx[i+1][0] )\
            *(layerInsertionIdx[-1][0]+1)
      regularisationM.ravel()[diagIdx]=1e-3*weights[i]**-2.0
elif simple=='intermediate':
   # \/ Intermediate layer restriction
   print("Making regularisation matrix...",end="") ; sys.stdout.flush()
   layerMapping=[]
   for actualLh in actualGeometry.layerHeights:
      i=reconGeometry.layerHeights.searchsorted(actualLh)
      if i==reconGeometry.nLayers: continue # can't do this one
      layerMapping.append([actualLh,i])
      layerMapping[-1].append(
         (reconGeometry.layerHeights[i]-actualLh
            )/(reconGeometry.layerHeights[i]-reconGeometry.layerHeights[i-1]))

   regularisationD=numpy.ones(layerInsertionIdx[-1][0])*1e-12
     # \/ fill diagonal roughly 
   for lm in layerMapping:
      diagIdx=\
         numpy.arange(layerInsertionIdx[lm[1]][0],layerInsertionIdx[lm[1]+1][0])
            
      regularisationD[diagIdx]=(1-lm[-1])**2.0*(weights[lm[1]]**2.0) # add weighting
      if lm[-1]!=0: # add to corresponding points in layer below
         diagIdx=\
           numpy.arange(layerInsertionIdx[lm[1]-1][0],layerInsertionIdx[lm[1]][0])
         regularisationD[diagIdx]+=(lm[-1])**2.0*(weights[lm[1]]**2.0)

   regularisationM.ravel()[
         numpy.arange(layerInsertionIdx[-1][0])*(layerInsertionIdx[-1][0]+1)]=\
            1e-3*regularisationD**-0.5
elif simple=='laplacian':
   offset=0
   for i in range(reconGeometry.nLayers):
      tlO=gradientOperator.laplacianOperatorType1(
         pupilMask=reconGeometry.layerMasks[i].sum(axis=0) ).returnOp()
      print(i,offset,tlO.shape[0])
      regularisationM[
         offset:offset+tlO.shape[0],offset:offset+tlO.shape[0]]=\
               1e-3*tlO.dot(tlO.T)*weights[i]**-2.0
      offset+=tlO.shape[0]

if simple=='SVD':
   # \/ SVD approach
   usePinv=True
   print("SVD...",end="") ; sys.stdout.flush()
   sTs_invSVD=numpy.linalg.pinv( sTs, rcond=1e-3 ) # pinv version
   print("pinv, done.") ; sys.stdout.flush()
else:
   if regularisationM.var()==0:
      raise ValueError("Regularisation is zero, was it forgot?")
   print("...inverting...",end="") ; sys.stdout.flush()
   sTs_invR=numpy.linalg.inv(sTs + regularisationM ) 
   print("(done)") ; sys.stdout.flush()

# \/ choose inversion method
if "sTs_invSVD" in dir():
   sTs_inv=sTs_invSVD
elif "sTs_invR" in dir():
   print("Using regularisation")
   sTs_inv=sTs_invR
if 'sTs_inv' not in dir():
   raise ValueError("Did you forget to enable a matrix?")


recoveryM=numpy.dot( sTs_inv, sumLayerExM.transpose() )

recoveredV=numpy.dot( recoveryM, randomProjV )
recoveredLayersA=[
   numpy.ma.masked_array(
      numpy.zeros(reconGeometry.layerNpix[i], numpy.float64),
         reconGeometry.layerMasks[i].sum(axis=0)==0)
            for i in range(reconGeometry.nLayers) ]

for i in range(reconGeometry.nLayers):
   recoveredLayersA[i].ravel()[layerInsertionIdx[i][1]]=\
     recoveredV[layerInsertionIdx[i][0]:layerInsertionIdx[i+1][0]]
   recoveredLayersA[i]-=recoveredLayersA[i].mean()

# now decide if we can do a simple, naive comparison
#}if len(recoveredV)==len(randomExV):
print("\nDirect")
pyp.figure(1)
pyp.jet()
for i in range(reconGeometry.nLayers):
   pyp.subplot(reconGeometry.nLayers,3,1+i*3)
   pyp.imshow( recoveredLayersA[i],
      interpolation='nearest',vmin=-1*weights[i],vmax=1*weights[i] )
   pyp.xlabel("layer={0:1d}".format(i+1))
   pyp.ylabel("recov.")
   pyp.subplot(reconGeometry.nLayers,3,2+i*3)
   if i==0: pyp.title("input vs. reconstruction")
   pyp.imshow( inputDataA[i],
      interpolation='nearest',vmin=-1*weights[i],vmax=1*weights[i] )
   pyp.ylabel("orig.")
   pyp.subplot(reconGeometry.nLayers,3,3+i*3)
   pyp.imshow( inputDataA[i]-recoveredLayersA[i],
      interpolation='nearest',vmin=-1*weights[i],vmax=1*weights[i] )
   pyp.ylabel("diff.")
   print(" Layer#{0:d}".format(i+1))
   print("  Original RMS={0:5.3g}".format(inputDataA[i].var()))
   print("  Difference RMS={0:5.3g}".format(
      (inputDataA[i]-recoveredLayersA[i]).var()))

pyp.waitforbuttonpress()
