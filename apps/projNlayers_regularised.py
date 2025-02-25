from __future__ import print_function
# What is this?
# Test projection matrices over N layers, using a different reconstructor.
#
# This version adds regularisation to constrain layer solution variance, to add
# a priori information.

import numpy
import Zernike
import projection
import gradientOperator
import matplotlib.pyplot as pyp
import commonSeed
import sys

#??# What does this for loop do??
#??# I think it is here to account for the setting of the seed
#??for i in range(3): j=numpy.random.normal()

nAzi=5
baseSize=13
za=15/20.0e3
dH=3e3
Hmax=12e3
snr=10
#
testZernikes=False
nullNth=False
sameGeometry=True
#
useDerivativeOp=[False,'grads','lap','curv'][1]
derivativeOpFirst=False
#
# None means Tikhonov
# True means 'intermediate layer restriction'
# False means Laplacian/explicit covariance-based approximation
# SVD means avoid direct inversion
simple=['interpolated','lap','SVD','tikhonov'][1]
layerExclusion=[]#[0]+range(2,reconGeometry.nLayers)
laplacian=True



mask=numpy.ones([baseSize]*2)
#Zernike.anyZernike(1,baseSize,baseSize/2,ongrid=1)\
#      -Zernike.anyZernike(1,baseSize,baseSize/2/7.0,ongrid=1)
mask=mask.astype(numpy.int32)
nMask=int(mask.sum())

# angle, 5/20e3 is ~50'' which is ~CANARY 1
# Separation of one sub-ap between regular spacing is at alt,
# |zenith_angle*h*(1,0)-(cos(azi),sin(azi))|=zenith_angle*h*((1-cos)**2.0+sin**2.0)
#=za*h*(2-2cos(azi)), azi=2pi/N
aa=2*numpy.pi*(nAzi**-1.0)
#dH=( za*(2-2*numpy.cos(aa)) )**-1.0*1.0

reconGeometry=projection.projection(
      numpy.ceil( numpy.arange(numpy.ceil((Hmax/dH)))*dH ),
      [za]*nAzi, numpy.arange(nAzi)*aa, mask )

okay=reconGeometry.createLayerMasks()
if not okay:
   raise ValueError("Eek! (1)")


if useDerivativeOp!=False:
   if useDerivativeOp!='curv':
      if useDerivativeOp=='grads':
         if derivativeOpFirst:
            raise RuntimeError("Only Laplacian operator and useDerivativeFirst"+
            " is currently supported")
         print("Gradient operator applied")
         derivOp=gradientOperator.gradientOperatorType1( pupilMask=mask )
         derivM=derivOp.returnOp()
      elif useDerivativeOp=='lap':
         print("Laplacian operator applied")
         derivOp=gradientOperator.laplacianOperatorType1(
               pupilMask=mask) #, rebalance=0)
         derivM=derivOp.returnOp()
      else:
         raise ValueError("Unsupported derivative operator")
      if derivativeOpFirst:
         print("Order is : derivative then projection")
         # the derivative is applied to the projected layers 
         # i.e. integration and then tomography of wavefront
      else:
         # the derivative is applied to the projected layers 
         # i.e. integration and then tomography of wavefront
         print("Order is : projection then derivative")
   else:
      if derivativeOpFirst:
         raise RuntimeError(
               "derivativeOpFirst & curvatureOfSlopes not supported")
      derivOps=[ thisOperator(pupilMask=mask) for thisOperator in (
         gradientOperator.gradientOperatorType1,
         gradientOperator.curvatureViaSlopesType1) ]
      derivMs=[ thisOperator.returnOp() for thisOperator in derivOps ]
      derivM=derivMs[1].dot(derivMs[0])
      print("Curvature of slopes and gradient operators applied")
  
   if derivativeOpFirst:
      derivMs=[ gradientOperator.laplacianOperatorType1(
               pupilMask=reconGeometry.layerMasks[i].sum(axis=0), rebalance=1
               ).returnOp() 
            for i in range( reconGeometry.nLayers ) ]
      derivFirstM=numpy.zeros(
            numpy.array([ tdM.shape for tdM in derivMs ]).sum(axis=0),
            numpy.float64 )
      for i in range(reconGeometry.nLayers):
         derivFirstM[
             reconGeometry.trimIdx(0)[i][0]:reconGeometry.trimIdx(0)[i+1][0]
            ,reconGeometry.trimIdx(0)[i][0]:reconGeometry.trimIdx(0)[i+1][0]]=\
               derivMs[i]
   derivAllM=numpy.zeros(
         [derivM.shape[0]*nAzi, derivM.shape[1]*nAzi], numpy.float64)
   for i in range(nAzi): # nAzi directions gradient operator
      derivAllM[derivM.shape[0]*i:derivM.shape[0]*(i+1),
               derivM.shape[1]*i:derivM.shape[1]*(i+1)]=derivM
else:
   print("No derivative operator applied")



if sameGeometry:
   actualGeometry=reconGeometry
   print("NOTE: Same geometry assumed")
else:
   actualGeometry=projection.projection(
    [0]+numpy.sort(
       numpy.random.uniform(1e3,15e3,size=5) ).astype('i').tolist(),
    [za]*nAzi, numpy.arange(nAzi)*aa, mask )

okay=actualGeometry.createLayerMasks()
if not okay:
   raise ValueError("Eek! (2)")

# projection matrices
layerExM=reconGeometry.layerExtractionMatrix()
sumPrM=reconGeometry.sumProjectedMatrix()
reconTrimIdx=reconGeometry.trimIdx()
sumLayerExM=numpy.dot( sumPrM, layerExM.take(reconTrimIdx,axis=1) )
if useDerivativeOp:
   if derivativeOpFirst:
      sumLayerExM=numpy.dot( sumLayerExM, derivFirstM )
      sumLayerExMComparator=numpy.dot( derivAllM, sumLayerExM )
   else:
      sumLayerExM=numpy.dot( derivAllM, sumLayerExM )
   print("Applying derivative operator") ; sys.stdout.flush()

actualLayerExM=actualGeometry.layerExtractionMatrix()
actualSumPrM=actualGeometry.sumProjectedMatrix()
actualTrimIdx=actualGeometry.trimIdx()
actualSumLayerExM=numpy.dot(
   actualSumPrM, actualLayerExM.take(actualTrimIdx,axis=1) )


# projection of the actual data
weights=(numpy.random.uniform(0,1,size=100)).tolist()

import kolmogorov
print("Input data...",end="") ; sys.stdout.flush()

inputData=[]
if testZernikes: print("*** Test Zernikes")
for i in range(actualGeometry.nLayers):
   tS=actualGeometry.layerNpix[i]
   if not testZernikes:
      thisData=kolmogorov.TwoScreens(tS.max()*2,
               (nMask**0.5)/2.0)[0][:tS[0],:tS[1]]
   else:
      thisData=Zernike.anyZernike(i+2,tS.max(),tS.max()/2,clip=False)
   inputData.append(
         2*(thisData-thisData.mean())/(thisData.max()-thisData.min()) )
   if i==nullNth-1: inputData[-1]*=0 # artificially null layer

print("...reforming...", end="") ; sys.stdout.flush()
for i in range(len(inputData)):
   inputData[i]*=weights[i]

inputDataA=[
   numpy.ma.masked_array(inputData[i],
      actualGeometry.layerMasks[i].sum(axis=0)==0)
         for i in range(actualGeometry.nLayers) ]

inputDataV=[]
for id in inputData: inputDataV+=id.ravel().tolist()

print("(done)") ; sys.stdout.flush()

# calculate input vector
randomExV=numpy.take( inputDataV, actualTrimIdx )
randomProjV=numpy.dot( actualSumLayerExM, randomExV )
## now add random tilt, to simulate that from a LGS
#tipTiltV=numpy.array([
#   numpy.add.outer(
#     numpy.arange(actualGeometry.pupilMask.shape[0])*numpy.random.normal()
#     ,numpy.arange(actualGeometry.pupilMask.shape[1])*numpy.random.normal()
#     ).ravel().take( actualGeometry.maskIdx ) for zzz in range(nAzi) ]).ravel()
#tipTiltV/=tipTiltV.max()-tipTiltV.min()
#randomProjV+=tipTiltV
if useDerivativeOp:
   randomProjV=numpy.dot( derivAllM, randomProjV )
else:
   # remove mean from each input, independently
   for i in range(actualGeometry.nAzi):
      randomProjV[i*nMask:(i+1)*nMask]-=randomProjV[i*nMask:(i+1)*nMask].mean()

if snr: randomProjV+=numpy.random.normal(0,snr**-1.0,len(randomProjV))
            
print("All data prepared") ; sys.stdout.flush()

# now, try straight inversion onto the illuminated portions of the layers 
sTs=numpy.dot(sumLayerExM.transpose(),sumLayerExM)

layerInsertionIdx=reconGeometry.trimIdx(False)
regularisationM=numpy.zeros([layerInsertionIdx[-1][0]]*2)

if simple=='tikhonov':
   # \/ Tikhonov regularisation
   layerInsertionIdx=reconGeometry.trimIdx(False)
   for i in range(reconGeometry.nLayers):
      diagIdx=numpy.arange( layerInsertionIdx[i][0],layerInsertionIdx[i+1][0] )\
            *(layerInsertionIdx[-1][0]+1)
      if i in layerExclusion:
         regularisationM.ravel()[diagIdx]=1e3
      else:
         regularisationM.ravel()[diagIdx]=1e-3*weights[i]**-2.0
elif simple=='interpolated':
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
elif simple=='lap':
   offset=0
   if laplacian:
      print("Laplacian covariance approximation")
      import gradientOperator
      for i in range(reconGeometry.nLayers):
         tlO=gradientOperator.laplacianOperatorType1(
            pupilMask=reconGeometry.layerMasks[i].sum(axis=0) ).returnOp()
         
         regularisationM[
            offset:offset+tlO.shape[0],offset:offset+tlO.shape[0]]=\
                  1e-3*tlO.dot(tlO.T)*weights[i]**-2.0
         offset+=tlO.shape[0]
   else:
      print("Full covariance") ; sys.stdout.flush()
      import phaseCovariance
      directPCOne=phaseCovariance.covarianceDirectRegular(
            reconGeometry.layerNpix[-1].max(), nMask**0.5/2.0, 1e6)
      for i in range(reconGeometry.nLayers):
         maskedCov=phaseCovariance.covarianceMatrixFillInMasked(
               directPCOne, (reconGeometry.layerMasks[i].sum(axis=0)!=0) )
         covInvM=numpy.linalg.inv(
               maskedCov+0.001*numpy.identity(maskedCov.shape[0]) )
         regularisationM[
               offset:offset+covInvM.shape[0],offset:offset+covInvM.shape[0]]=\
                  1e-3*covInvM*weights[i]**-2.0
         offset+=covInvM.shape[0]
elif simple=='SVD':
   # \/ SVD approach
   usePinv=True
   print("SVD...",end="") ; sys.stdout.flush()
   if not usePinv:
      sTs_invSVD_components=numpy.linalg.svd( sTs, full_matrices=False )
      inv_s=numpy.zeros( [sTs.shape[0]]*2 )
      inv_s.ravel()[ numpy.arange(sTs.shape[0])*(sTs.shape[0]+1) ]=\
         (sTs_invSVD_components[1]+1e-10)**-1.0
      sTs_invSVD=( numpy.dot( sTs_invSVD_components[2].T,
         numpy.dot( inv_s, sTs_invSVD_components[0].T )))
      print("manually, done.") ; sys.stdout.flush()
   else:
      sTs_invSVD=numpy.linalg.pinv( sTs, rcond=1e-3 ) # pinv version
      print("pinv, done.") ; sys.stdout.flush()

if simple!='SVD':
   if regularisationM.var()==0:
      raise ValueError("Regularisation is zero, was it forgot?")
   print("...inverting...",end="") ; sys.stdout.flush()
   sTs_invR=numpy.linalg.inv(sTs + regularisationM ) 
   print("(done)") ; sys.stdout.flush()

# \/ choose inversion method
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
for i in range(reconGeometry.nLayers):
   recoveredLayersA[i].ravel()[layerInsertionIdx[i][1]]=\
     recoveredV[layerInsertionIdx[i][0]:layerInsertionIdx[i+1][0]]
   recoveredLayersA[i]-=recoveredLayersA[i].mean()

# now decide if we can do a simple, naive comparison
if len(recoveredV)==len(randomExV):
   print("\nDirect")
   # /\ a bit naive but ought to do
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
   for i in range(reconGeometry.nLayers):
      centreRecoveredV[i*nMask:(i+1)*nMask]-=\
            centreRecoveredV[i*nMask:(i+1)*nMask].mean()
      inputCentreV[i*nMask:(i+1)*nMask]-=\
            inputCentreV[i*nMask:(i+1)*nMask].mean()

   centreMaskedA=numpy.ma.masked_array( 
      numpy.zeros([reconGeometry.nLayers,2]+list(mask.shape)),
         [[mask==0]*2]*reconGeometry.nLayers )
   if reconGeometry.nLayers==actualGeometry.nLayers:
      pyp.figure(2)
      print("\nCentre proj")
      for i in range(reconGeometry.nLayers):
         pyp.subplot(reconGeometry.nLayers,3,1+i*3)
         centreMaskedA[i,0].ravel()[actualGeometry.maskIdxs[0]]=\
            centreRecoveredV[i*nMask:(i+1)*nMask]
         centreMaskedA[i,0]-=centreMaskedA[i,0].mean()
         pyp.imshow( centreMaskedA[i,0]+0.0, interpolation='nearest',
            vmin=-1*weights[i],vmax=1*weights[i] )
         pyp.xlabel("layer={0:1d}".format(i+1))
         pyp.ylabel("recov.")
         
         pyp.subplot(reconGeometry.nLayers,3,2+i*3)
         if i==0: pyp.title("centre proj: input vs. reconstruction")
         centreMaskedA[i,1].ravel()[actualGeometry.maskIdxs[0]]=\
            inputCentreV[i*nMask:(i+1)*nMask]
         centreMaskedA[i,1]-=centreMaskedA[i,1].mean()
         pyp.imshow( centreMaskedA[i,1]+0.0, interpolation='nearest',
            vmin=-1*weights[i],vmax=1*weights[i] )
         pyp.ylabel("orig.")
         
         pyp.subplot(reconGeometry.nLayers,3,3+i*3)
         pyp.imshow( centreMaskedA[i,1]-centreMaskedA[i,0],
            interpolation='nearest',
            vmin=-1*weights[i],vmax=1*weights[i] )
         pyp.ylabel("diff.")

         print(" Layer#{0:d}".format(i+1))
         print("  Original RMS={0:5.3g}".format(centreMaskedA[i,1].var()))
         print("  Difference RMS={0:5.3g}".format(
            (centreMaskedA[i,0]-centreMaskedA[i,1]).var()))
   
   # last bit, see how centre projected and summed values differ
   # naive is just the mean of the input vectors
   actualProjCentSumM=actualGeometry.sumCentreProjectedMatrix()
   reconProjCentSumM=reconGeometry.sumCentreProjectedMatrix()
   actualCentSumV=numpy.dot( actualProjCentSumM, inputCentreV )
   reconCentSumV=numpy.dot( reconProjCentSumM, centreRecoveredV )
   naiveMeanV=numpy.zeros( nMask, numpy.float64 )
   for i in range(nAzi): # create arithmetic mean, by slicing
      naiveMeanV+=randomProjV[i*nMask:(i+1)*nMask]
   naiveMeanV/=nAzi+0.0
   naiveMeanV-=naiveMeanV.mean()

   centreSumMaskedA={}
   for i in ("actual","recon","naive"):
      centreSumMaskedA[i]=numpy.ma.masked_array( 
         numpy.zeros(mask.shape), [mask==0] )
   pyp.figure(3)
   centreSumMaskedA['actual'].ravel()[actualGeometry.maskIdxs[0]]=actualCentSumV
   centreSumMaskedA['recon'].ravel()[actualGeometry.maskIdxs[0]]=reconCentSumV
   centreSumMaskedA['naive'].ravel()[actualGeometry.maskIdxs[0]]=naiveMeanV

   minMax=( centreSumMaskedA['actual'].ravel().min()
          , centreSumMaskedA['actual'].ravel().max() )
   pyp.subplot(2,2,1)            
   pyp.title("centreSum: recov.")
   pyp.imshow( centreSumMaskedA['actual'], interpolation='nearest',
     vmin=minMax[0], vmax=minMax[1] )

   pyp.subplot(2,2,2)
   pyp.imshow( centreSumMaskedA['recon'], interpolation='nearest',
     vmin=minMax[0], vmax=minMax[1] )
   pyp.title("CS: orig.")

   pyp.subplot(2,2,3)
   pyp.imshow( centreSumMaskedA['recon']-centreSumMaskedA['actual'],
     interpolation='nearest',
     vmin=minMax[0], vmax=minMax[1] )
   pyp.title("diff.")

   pyp.subplot(2,2,4)
   pyp.imshow( centreSumMaskedA['naive'], interpolation='nearest',
     vmin=minMax[0], vmax=minMax[1] )
   pyp.title("naive.")

   print("\nCentre summed")
   print(" Original RMS={0:5.3g}".format(actualCentSumV.var()))
   print(" Difference RMS={0:5.3g}".format(
      (actualCentSumV-reconCentSumV).var()))
   print(" ( naive difference RMS={0:5.3g} )".format(
      (actualCentSumV-naiveMeanV).var()))
   print("\n~GLAO")
   print(" Original RMS={0:5.3g}".format(inputCentreV[:nMask].var()))
   print(" Difference RMS={0:5.3g}".format(
      (inputCentreV[:nMask]-centreRecoveredV[:nMask]).var()))
   print(" ( naive difference RMS={0:5.3g} )".format(
      (inputCentreV[:nMask]--naiveMeanV).var()))
   pyp.figure(4)
   pyp.plot( actualCentSumV, reconCentSumV, 'rx',
     label='projected centre, summed' )
   pyp.plot( inputCentreV[:nMask], centreRecoveredV[:nMask], 'bx',
     label='surface layers' )
   pyp.legend(loc=0)
      
   pyp.plot( [numpy.array(
      [list(pyp.gca().get_xlim())]+[list(pyp.gca().get_ylim())]
         ).transpose().min(),numpy.array(
      [list(pyp.gca().get_xlim())]+[list(pyp.gca().get_ylim())]
         ).transpose().max()],
            [numpy.array(
      [list(pyp.gca().get_xlim())]+[list(pyp.gca().get_ylim())]
         ).transpose().min(),numpy.array(
      [list(pyp.gca().get_xlim())]+[list(pyp.gca().get_ylim())]
         ).transpose().max()], 'k--')
   pyp.xlabel("Input, centre proj.")
   pyp.ylabel("Recovered, centre proj.")
   
pyp.waitforbuttonpress()
