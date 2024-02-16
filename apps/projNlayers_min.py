from __future__ import print_function
# What is this?
# Test projection matrices over N layers, using a different reconstructor.

import numpy
import Zernike
import projection
import matplotlib.pyplot as pg

nAzi=5
mask=Zernike.anyZernike(1,8,4,ongrid=1)-Zernike.anyZernike(1,8,1,ongrid=1)
mask=mask.astype(numpy.int32)
nMask=int(mask.sum())
setSeed=False
useRandomIp=False

# ---------

if setSeed:
   import commonSeed

# angle, 5/20e3 is ~50'' which is ~CANARY 1
# [0,1e3,5e3,10e3,15e3],
reconGeometry=projection.projection(
 [0,5e3,15e3],
 numpy.ones(5)*5/20.0e3,
 numpy.arange(nAzi)*2*numpy.pi*(nAzi**-1.0), mask )

okay=reconGeometry.createLayerMasks()
if not okay:
   raise ValueError("Eek!")

actualGeometry=projection.projection(
 [0,5e3,15e3],
# numpy.linspace(0,20e3,12),
 numpy.ones(5)*5/20.0e3,\
 numpy.arange(nAzi)*2*numpy.pi*(nAzi**-1.0), mask )

okay=actualGeometry.createLayerMasks()
if not okay:
   raise ValueError("Eek!")

# projection
layerExM=reconGeometry.layerExtractionMatrix()
sumPrM=reconGeometry.sumProjectedMatrix()
reconTrimIdx=reconGeometry.trimIdx()
sumLayerExM=numpy.dot( sumPrM, layerExM.take(reconTrimIdx,axis=1) )

# projection of the actual data
#weights=[1,0.2,0.2,0.5,0.5,0.2,0.2]
weights=numpy.random.uniform(0,1,size=12)

print("Input data...",end="")
import kolmogorov
inputData=[]
for i in range(actualGeometry.nLayers):
   tS=actualGeometry.layerNpix[i]
   if useRandomIp:
      thisData=numpy.random.uniform(-1,1,size=tS)
   else:
      thisData=kolmogorov.TwoScreens(
            tS.max()*2,(nMask**0.5)/2.0)[0][:tS[0],:tS[1]]
   inputData.append(
      2*(thisData-thisData.mean())/(thisData.max()-thisData.min()) )
#   if i!=0: inputData[-1]*=0 # artificially null layers other than the nth
print("(done)")


for i in range(len(inputData)):
   inputData[i]*=weights[i]
inputDataA=[
   numpy.ma.masked_array(inputData[i],
      actualGeometry.layerMasks[i].sum(axis=0)==0)
         for i in range(actualGeometry.nLayers) ]
inputDataV=[]
for id in inputData: inputDataV+=id.ravel().tolist()

actualLayerExM=actualGeometry.layerExtractionMatrix()
actualSumPrM=actualGeometry.sumProjectedMatrix()
actualTrimIdx=actualGeometry.trimIdx()
actualSumLayerExM=numpy.dot(
   actualSumPrM, actualLayerExM.take(actualTrimIdx,axis=1) )

randomExV=numpy.take( inputDataV, actualTrimIdx )
randomProjV=numpy.dot( actualSumLayerExM, randomExV ) # our input vector

# now, try straight inversion onto the illuminated portions of the layers 
sTs=numpy.dot(sumLayerExM.transpose(),sumLayerExM)
sTs_invR=numpy.linalg.inv( sTs + numpy.identity(len(reconTrimIdx))*1e-1) 
#sTs_invSVD=numpy.linalg.pinv( sTs, rcond=1e-6 )
sTs_inv=None
if "sTs_invSVD" in dir():
   sTs_inv=sTs_invSVD
   print("Using SVD")
if "sTs_invR" in dir():
   if sTs_inv==None:
      print("Using regularisation")
      sTs_inv=sTs_invR
   else:
      raise ValueError("Something funny, two inversion matrices?")
if type(sTs_inv)==type(None):
   raise ValueError("Did you forget to enable a matrix?")

   

recoveryM=numpy.dot( sTs_inv, sumLayerExM.transpose() )

recoveredV=numpy.dot( recoveryM, randomProjV )
recoveredLayersA=[
   numpy.ma.masked_array(
      numpy.zeros(reconGeometry.layerNpix[i], numpy.float64),
         reconGeometry.layerMasks[i].sum(axis=0)==0)
            for i in range(reconGeometry.nLayers) ]
layerInsertionIdx=reconGeometry.trimIdx(False)

# now decide if we can do a simple, naive comparison
if len(recoveredV)==len(randomExV):
   print("\nDirect")
   # /\ a bit naive but ought to do
   pg.figure()
   pg.jet()
   for i in range(reconGeometry.nLayers):
      recoveredLayersA[i].ravel()[layerInsertionIdx[i][1]]=\
        recoveredV[layerInsertionIdx[i][0]:layerInsertionIdx[i+1][0]]
      pg.subplot(reconGeometry.nLayers,3,1+i*3)
      pg.imshow( recoveredLayersA[i],
         interpolation='nearest',vmin=-1*weights[i],vmax=1*weights[i] )
      pg.xlabel("layer={0:1d}".format(i+1))
      pg.ylabel("recov.")
      pg.subplot(reconGeometry.nLayers,3,2+i*3)
      pg.imshow( inputDataA[i],
         interpolation='nearest',vmin=-1*weights[i],vmax=1*weights[i] )
      pg.ylabel("orig.")
      pg.subplot(reconGeometry.nLayers,3,3+i*3)
      pg.imshow( inputDataA[i]-recoveredLayersA[i],
         interpolation='nearest',vmin=-1*weights[i],vmax=1*weights[i] )
      pg.ylabel("diff.")
      print(" Layer#{0:d}".format(i+1))
      print("  Original RMS={0:5.3g}".format(inputDataA[i].var()))
      print("  Difference RMS={0:5.3g}".format(
         (inputDataA[i]-recoveredLayersA[i]).var()))
else:
   print("Cannot compare the directly recovered values, incompatible sizes")

# centre projected values
# first check that the centre projection is valid
import sys
valid=False
try:
   actualCentreProjM=actualGeometry.layerCentreProjectionMatrix().take(
      actualTrimIdx, axis=1 )
   if actualCentreProjM.sum(axis=1).var()!=0\
      and actualCentreProjM.sum(axis=1)[0]!=1:
         raise ValueError("Actual geometry isn't suitable")
   reconCentreProjM=reconGeometry.layerCentreProjectionMatrix().take(
      reconTrimIdx, axis=1 )
   if reconCentreProjM.sum(axis=1).var()!=0\
      and reconCentreProjM.sum(axis=1)[0]!=1:
         raise ValueError("Reconstruction geometry isn't suitable")
   valid=True
except:
   print(sys.exc_info())

if valid:
   centreRecoveredV=numpy.dot( reconCentreProjM, recoveredV )
   inputCentreV=numpy.dot( actualCentreProjM, randomExV )

   centreMaskedA=numpy.ma.masked_array( 
      numpy.zeros([reconGeometry.nLayers,2]+list(mask.shape)),
         [[mask==0]*2]*reconGeometry.nLayers )
   if reconGeometry.nLayers==actualGeometry.nLayers:
      pg.figure()
      print("\nCentre proj")
      for i in range(reconGeometry.nLayers):
         pg.subplot(reconGeometry.nLayers,3,1+i*3)
         centreMaskedA[i,0].ravel()[actualGeometry.maskIdx]=\
            centreRecoveredV[i*nMask:(i+1)*nMask]
         pg.imshow( centreMaskedA[i,0]+0.0, interpolation='nearest',
            vmin=-1*weights[i],vmax=1*weights[i] )
         pg.xlabel("layer={0:1d}".format(i+1))
         pg.ylabel("recov.")
         
         pg.subplot(reconGeometry.nLayers,3,2+i*3)
         centreMaskedA[i,1].ravel()[actualGeometry.maskIdx]=\
            inputCentreV[i*nMask:(i+1)*nMask]
         pg.imshow( centreMaskedA[i,1]+0.0, interpolation='nearest',
            vmin=-1*weights[i],vmax=1*weights[i] )
         pg.ylabel("orig.")
         
         pg.subplot(reconGeometry.nLayers,3,3+i*3)
         pg.imshow( centreMaskedA[i,1]-centreMaskedA[i,0],
            interpolation='nearest',
            vmin=-1*weights[i],vmax=1*weights[i] )
         pg.ylabel("diff.")

         print(" Layer#{0:d}".format(i+1))
         print("  Original RMS={0:5.3g}".format(centreMaskedA[i,1].var()))
         print("  Difference RMS={0:5.3g}".format(
            (centreMaskedA[i,0]-centreMaskedA[i,1]).var()))
   
   centreSumMaskedA=numpy.ma.masked_array( 
      numpy.zeros([2]+list(mask.shape)), [mask==0]*2 )
   # last bit, see how centre projected and summed values differ
   actualProjCentSumM=actualGeometry.sumCentreProjectedMatrix()
   reconProjCentSumM=reconGeometry.sumCentreProjectedMatrix()
   actualCentSumV=numpy.dot( actualProjCentSumM, inputCentreV )
   reconCentSumV=numpy.dot( reconProjCentSumM, centreRecoveredV )
   pg.figure()
   centreSumMaskedA[1].ravel()[actualGeometry.maskIdx]=actualCentSumV
   centreSumMaskedA[0].ravel()[actualGeometry.maskIdx]=reconCentSumV
   pg.subplot(1,3,1)            
   pg.imshow( centreSumMaskedA[0], interpolation='nearest',
            vmin=centreSumMaskedA[0].min(), vmax=centreSumMaskedA[0].max() )
   pg.title("recov.")
   pg.subplot(1,3,2)
   pg.imshow( centreSumMaskedA[1], interpolation='nearest',
            vmin=centreSumMaskedA[0].min(), vmax=centreSumMaskedA[0].max() )
   pg.title("orig.")
   pg.subplot(1,3,3)
   pg.imshow( centreSumMaskedA[1]-centreSumMaskedA[0], interpolation='nearest',
            vmin=centreSumMaskedA[0].min(), vmax=centreSumMaskedA[0].max() )
   pg.title("diff.")

   print("\nCentre summed")
   print(" Original RMS={0:5.3g}".format(actualCentSumV.var()))
   print(" Difference RMS={0:5.3g}".format(
      (actualCentSumV-reconCentSumV).var()))
   print("\n~GLAO")
   print(" Original RMS={0:5.3g}".format(inputCentreV[:nMask].var()))
   print(" Difference RMS={0:5.3g}".format(
      (inputCentreV[:nMask]-centreRecoveredV[:nMask]).var()))
   pg.figure()
   pg.plot( actualCentSumV, reconCentSumV, 'rx' )
   pg.plot( inputCentreV[:nMask], centreRecoveredV[:nMask], 'bx' )
      
   pg.plot( [numpy.array(
      [list(pg.gca().get_xlim())]+[list(pg.gca().get_ylim())]
         ).transpose().min(),numpy.array(
      [list(pg.gca().get_xlim())]+[list(pg.gca().get_ylim())]
         ).transpose().max()],
            [numpy.array(
      [list(pg.gca().get_xlim())]+[list(pg.gca().get_ylim())]
         ).transpose().min(),numpy.array(
      [list(pg.gca().get_xlim())]+[list(pg.gca().get_ylim())]
         ).transpose().max()], 'k--')
   pg.xlabel("Input, centre proj.")
   pg.ylabel("Recovered, centre proj.")
   
pg.waitforbuttonpress()
