from __future__ import print_function
# What is this?
# Test projection matrices over N layers, using different reconstructors
# and repetitions.

import numpy
import Zernike
import projection
import gradientOperator
import commonSeed
import sys

#??# What does this for loop do??
#??# I think it is here to account for the setting of the seed
#??for i in range(3): j=numpy.random.normal()

nAzi=3
baseSize=8
za=15/20.0e3
dH=2e3
Hmax=6e3
snr=10
#
testZernikes=False
nullNth=False
sameGeometry=True
#
useDerivativeOp=[False,'wfVS','grads','lapVS','lap'][2]
derivativeOpFirst=False
useContinuity=True # makes no difference, since ultimately the wavefront is reconstructed!
#
# None means Tikhonov
# True means 'intermediate layer restriction'
# False means Laplacian/explicit covariance-based approximation
# SVD means avoid direct inversion
simple=['interpolated','lap','SVD','tikhonov'][1]
layerExclusion=[]#[0]+range(2,reconGeometry.nLayers)
laplacian=True
nLoops=100
continuityPartitionPeriod=None

# ----------------------------------------------------------------------------

weights=(numpy.random.uniform(0,1,size=100)).tolist()

if useContinuity:
   import continuity
if not testZernikes:
   import kolmogorov

mask=Zernike.anyZernike(1,baseSize,baseSize/2,ongrid=1)\
      -Zernike.anyZernike(1,baseSize,baseSize/2/7.0,ongrid=1)
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
   gradOp=gradientOperator.gradientOperatorType1( pupilMask=mask )
   if useContinuity:
      print("Calculating continuity operator")
      loopsDef=continuity.loopsDefine( gradOp, continuityPartitionPeriod )
      loopIntM=continuity.loopsIntegrationMatrix( loopsDef, gradOp )
      neM,nrM=continuity.loopsNoiseMatrices(loopIntM, gradOp)
   gradM=gradOp.returnOp()
   gradMplus=numpy.identity(
         gradOp.numberSubaps*2) #numpy.linalg.pinv( gradM, 1e-6 )
   if useDerivativeOp!='curv':
      if useDerivativeOp=='wfVS':
         if derivativeOpFirst:
            raise RuntimeError("Only Laplacian operator and useDerivativeFirst"+
            " is currently supported")
         print("Wavefront-via-slopes operator applied")
         derivM=numpy.identity(gradOp.numberPhases)
         derivGM=numpy.linalg.pinv(gradM,1e-6)
      elif useDerivativeOp=='grads':
         if derivativeOpFirst:
            raise RuntimeError("Only Laplacian operator and useDerivativeFirst"+
            " is currently supported")
         print("Gradient operator applied")
         derivM=gradOp.returnOp()
         derivGM=numpy.identity(gradOp.numberSubaps*2)
      elif useDerivativeOp=='lapVS':
         print("Laplacian-via-slopes operator applied")
         derivOp=gradientOperator.laplacianOperatorViaSlopesType1(
               pupilMask=mask)
         derivGM=derivOp.returnOp()
         derivM=derivGM.dot(gradM)
      elif useDerivativeOp=='lap':
         print("Laplacian operator applied")
         derivOp=gradientOperator.laplacianOperatorType1( pupilMask=mask )
         derivM=derivOp.returnOp()
         derivGM=derivM.dot( numpy.linalg.pinv(gradM,1e-6) )
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
   derivAllGM=numpy.zeros(
         [derivM.shape[0]*nAzi, gradMplus.shape[1]*nAzi], numpy.float64)
   derivAllM=numpy.zeros(
         [derivM.shape[0]*nAzi, derivM.shape[1]*nAzi], numpy.float64)
   gradAllM=numpy.zeros(
         [gradM.shape[0]*nAzi, gradM.shape[1]*nAzi], numpy.float64)
   if useContinuity:
      nrAllM=numpy.zeros(
            [nrM.shape[0]*nAzi, nrM.shape[1]*nAzi], numpy.float64)
   for i in range(nAzi): # nAzi directions gradient operator
      derivAllGM[derivM.shape[0]*i:derivM.shape[0]*(i+1),
               gradMplus.shape[1]*i:gradMplus.shape[1]*(i+1)]=derivGM
      derivAllM[derivM.shape[0]*i:derivM.shape[0]*(i+1),
               derivM.shape[1]*i:derivM.shape[1]*(i+1)]=derivM
      gradAllM[gradM.shape[0]*i:gradM.shape[0]*(i+1),
               gradM.shape[1]*i:gradM.shape[1]*(i+1)]=gradM
      if useContinuity:
         nrAllM[nrM.shape[0]*i:nrM.shape[0]*(i+1),
                  nrM.shape[1]*i:nrM.shape[1]*(i+1)]=nrM
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
actualApplyLayerExM=actualSumLayerExM
if useDerivativeOp:
   actualApplyLayerExM=gradAllM.dot(actualApplyLayerExM)


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

   
# centre projected values
# first check that the centre projection is valid
centreValid=True
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
except ValueError:
   centreValid=False
except:
   print(sys.exc_info())

recoveryM=numpy.dot( sTs_inv, sumLayerExM.transpose() )

print("All data prepared") ; sys.stdout.flush()


# loop can begin here

data={
   'ip':[],
   'projip':[],
   'projip_cnoise':[],
   'recon_ip':[],
   'res':{}
}

print("<",end="") ; sys.stdout.flush()
for loopNum in range(nLoops):
   if loopNum%10==0: print(".",end="") ; sys.stdout.flush()
   if loopNum%100==0:
      print("{0:6d}".format(loopNum),end="") ; sys.stdout.flush()

   # projection of the actual data
   inputData=[]
   for i in range(actualGeometry.nLayers):
      tS=actualGeometry.layerNpix[i]
      if not testZernikes:
         thisData=kolmogorov.TwoScreens(tS.max()*2,
                  (nMask**0.5)/2.0)[0][:tS[0],:tS[1]]
      else:
         thisData=numpy.random.normal()*Zernike.anyZernike(
               i+2,tS.max(),tS.max()/2,clip=False)
      inputData.append(
            2*(thisData-thisData.mean())/(thisData.max()-thisData.min()) )
      if i==nullNth-1: inputData[-1]*=0 # artificially null layer

   for i in range(len(inputData)):
      inputData[i]*=weights[i]

   inputDataV=[]
   for id in inputData: inputDataV+=id.ravel().tolist()


   # calculate input vector
   randomExV=numpy.take( inputDataV, actualTrimIdx )
   data['ip'].append( randomExV ) # ***
   randomProjV=numpy.dot( actualApplyLayerExM, randomExV )
   if snr:
      randomProjV+=numpy.random.normal(0,snr**-1.0,len(randomProjV))
      if useContinuity:
         randomProjV=nrAllM.dot(randomProjV)
   
   if useDerivativeOp:
      randomProjV=numpy.dot( derivAllGM, randomProjV )
   else:
      # remove mean from each input, independently
      for i in range(actualGeometry.nAzi):
         randomProjV[i*nMask:(i+1)*nMask]-=randomProjV[i*nMask:(i+1)*nMask].mean()

   data['projip'].append( randomProjV ) # ***
   data['projip_cnoise'].append( randomProjV ) # ***
               
   recoveredV=numpy.dot( recoveryM, randomProjV )
   data['recon_ip'].append( recoveredV ) # ***
   layerInsertionIdx=reconGeometry.trimIdx(False)

   # now decide if we can do a simple, naive comparison
   if len(recoveredV)==len(randomExV):
      if 'direct' not in data['res'].keys():
         data['res']['direct']=[]
      tdata=[]
      for i in range(reconGeometry.nLayers):
         tdata+=[ (randomExV
                   )[layerInsertionIdx[i][0]:layerInsertionIdx[i+1][0]].var(),
                  (randomExV-recoveredV
                   )[layerInsertionIdx[i][0]:layerInsertionIdx[i+1][0]].var() ]
      
      data['res']['direct'].append([ loopNum ]+tdata )

   if centreValid:
      centreRecoveredV=numpy.dot( reconCentreProjM, recoveredV )
      inputCentreV=numpy.dot( actualCentreProjM, randomExV )
      for i in range(reconGeometry.nLayers):
         centreRecoveredV[i*nMask:(i+1)*nMask]-=\
               centreRecoveredV[i*nMask:(i+1)*nMask].mean()
         inputCentreV[i*nMask:(i+1)*nMask]-=\
               inputCentreV[i*nMask:(i+1)*nMask].mean()

      # last bit, see how centre projected and summed values differ
      # naive is just the mean of the input vectors
      actualProjCentSumM=actualGeometry.sumCentreProjectedMatrix()
      reconProjCentSumM=reconGeometry.sumCentreProjectedMatrix()
      actualCentSumV=numpy.dot( actualProjCentSumM, inputCentreV )
      reconCentSumV=numpy.dot( reconProjCentSumM, centreRecoveredV )
      
      if 'centre' not in data['res'].keys():
         data['res']['centre']=[]
      if 'surface' not in data['res'].keys():
         data['res']['surface']=[]

      data['res']['centre'].append( [ loopNum,
           actualCentSumV.var(), (actualCentSumV-reconCentSumV).var() ])
      data['res']['surface'].append( [ loopNum,
           inputCentreV[:nMask].var(),
           (inputCentreV[:nMask]-centreRecoveredV[:nMask]).var() ])

print(">",end="\n") ; sys.stdout.flush()

print("Configuration:")
print("\tSNR={0:5.2f}".format(snr))
if len(data['res'].keys())==0:
   raise RuntimeError("No results were stored, couldn't be stored?")
else:
   print("Results stored are:")
   for x in data['res'].keys():
      print("\t{0:s}".format(x))
      if x=='centre':
         centreData=numpy.array(data['res']['centre'])[:,1:]
         print("\t\tCentre projection: relative variance=\n\t\t"+
            "{0:6.3f}+/-{1:6.3f}".format(
               ( centreData[:,1]/centreData[:,0] ).mean(),
               ( centreData[:,1]/centreData[:,0] ).var()**0.5 ))
      if x=='surface':
         sfcData=numpy.array(data['res']['surface'])[:,1:]
         print("\t\tSurface layer: relative variance=\n\t\t"+
            "{0:6.3f}+/-{1:6.3f}".format(
               ( sfcData[:,1]/sfcData[:,0] ).mean(),
               ( sfcData[:,1]/sfcData[:,0] ).var()**0.5 ))
