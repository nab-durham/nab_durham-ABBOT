from __future__ import print_function
# What is this?
# Test projection matrices over N layers, using a different reconstructor.
#
# This version adds regularisation to constrain layer solution variance, to add
# a priori information and modal filtering with mixed NGS/LGS geometries.

import numpy
import Zernike
import projection
import gradientOperator
import modalBasis
import matplotlib.pyplot as pyp
import commonSeed,hurricaneNames
import sys

import time
titleName=hurricaneNames.randomName()+":"
# ------------------------------------------------------------------------
fixSeed=True
nAzis=(3,1)
nAzi=sum(nAzis)
baseSize=8,2
starHeights=[15e3]*nAzis[0]+[None]*nAzis[1]
zenDists=[4]*nAzis[0]+[0]*nAzis[1]
zenAngs=numpy.array(zenDists)*20e3**-1.0
azAngs=[ 2*numpy.pi*(nAzis[0]**-1.0)*i for i in range(nAzis[0]) ]+\
   [ 2*numpy.pi*(nAzis[1]**-1.0)*i for i in range(nAzis[1]) ] 
dH=5e3
Hmax=10e3+1
#}sameGeometry=True # always assume True in this code
#
# naive means Tikhonov
# intermediate means 'intermediate layer restriction'
# laplacian means Laplacian approximation
# SVD means avoid direct inversion
regularizationType=['intermediate','laplacian','SVD','naive'][1]
modalFilterEnabled,removeAllLGStilts=1,1
# ------------------------------------------------------------------------

if not fixSeed: numpy.random.seed(int(time.time()%1234))
heights=numpy.ceil( numpy.arange(numpy.ceil((Hmax/dH)))*dH )
print("Run is '{0:s}'".format(titleName[:-1]))
print("Seed is "+"not "*(not fixSeed)+"fixed")
if modalFilterEnabled:
   print("Mixed NGS/LGS geometry with modal filtering")
   if removeAllLGStilts:
      print("Removing mean of LGS tilts")
   else:
      print("Not altering LGS tilts")
else:
   print("Effective NGS-only geometry")
if not modalFilterEnabled and removeAllLGStilts:
   raise RuntimeError("Clash: no modal filtering, but applied LGS tilt removal")
print("No. LGS={0[0]:d}\nNo. NGS={0[1]:d}".format(nAzis))
if not modalFilterEnabled: print("****",end="\t")
print("Modal filtering is {0:s}abled".format(
      "en" if modalFilterEnabled else "dis"))
print("Heights="+str(heights))
print("Reconstructor="+regularizationType)
print("azAngs="+str([ tA/numpy.pi*180 for tA in azAngs[:nAzis[0]]])
           +"/"+str([ tA/numpy.pi*180 for tA in azAngs[nAzis[0]:]]) )
print("zenAngs="+str(zenAngs[:nAzis[0]])+"/"+str(zenAngs[nAzis[0]:]))
##print("zenDists="+str(zenDists[:nAzis[0]])+"/"+str(zenDists[nAzis[0]:]))
print("LGS/NGS scale:= {0[0]:d}x{0[0]:d}/{0[1]:d}x{0[1]:d}".format(baseSize))

pixelScales=[1]*nAzis[0]+[ baseSize[0]/baseSize[1] ]*nAzis[1]+[1]

masks=(
   Zernike.anyZernike(1,baseSize[0],baseSize[0]/2,ongrid=1),
   numpy.ones([baseSize[1]]*2) )
##mask=mask.astype(numpy.int32)
nMasks=[ int(tM.sum()) for tM in masks ]

gradOps=map( lambda ip : 
   gradientOperator.gradientOperatorType1(pupilMask=ip), masks )
gradMs=map( lambda ip : ip.returnOp(), gradOps )
gradAllM=numpy.zeros(
      [ sum(map( lambda ip : gradMs[ip].shape[dirn]*nAzis[ip], (0,1) ))
        for dirn in (0,1) ],
      numpy.float64 )
for i,tnAzi in enumerate(nAzis):
   tgradM=gradMs[i]
   for j in range(tnAzi):
      offsets=map( lambda ip : gradMs[0].shape[ip]*(i*nAzis[0]), (0,1) )
      sizes=map( lambda ip : gradMs[i].shape[ip], (0,1) )
      gradAllM[
         offsets[0]+sizes[0]*j: offsets[0]+sizes[0]*(j+1),
         offsets[1]+sizes[1]*j: offsets[1]+sizes[1]*(j+1) ]=tgradM

reconGeometry=projection.projection(
      heights,
      zenAngs, azAngs,
      [ masks[0] ]*nAzis[0]+[ masks[1] ]*nAzis[1]+[ masks[0] ],
      starHeights, pixelScales )

   # \/ compute modal basis for mask no. 1 only
ZmodalBasis=modalBasis.modalBasis( masks[0], [1],[1,2], orthonormalize=0 )
modalFiltering=[ 
      thismodalB.reshape([-1,1]).dot(thismodalB.reshape([1,-1]))
         for thismodalB in ZmodalBasis.modalFunctions ]
modalFilteringSummed=(
      numpy.identity(nMasks[0])-numpy.array(modalFiltering).sum(axis=0) )
modalFiltAllM=numpy.identity(
         nMasks[0]*nAzis[0]+nMasks[1]*nAzis[1], dtype=numpy.float64)
for j in range(nAzis[0]):
   modalFiltAllM[ nMasks[0]*j: nMasks[0]*(j+1),
                  nMasks[0]*j: nMasks[0]*(j+1) ]=modalFilteringSummed

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
sumLayerExM=sumPrM.dot(layerExM.take(reconTrimIdx,axis=1))
if modalFilterEnabled:
   print("Modal filtering applied")
   sumLayerExM=modalFiltAllM.dot(sumLayerExM)


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
weights=[1]*100#(numpy.random.uniform(0,1,size=100)).tolist()

import kolmogorov
print("Input data...",end="") ; sys.stdout.flush()

inputData=[]
inputDataV=[]
for i in range(actualGeometry.nLayers):
   tS=actualGeometry.layerNpix[i]
   thisData=kolmogorov.TwoScreens(tS.max()*2,
            (nMasks[0]**0.5)/2.0)[0][:tS[0],:tS[1]]
   inputData.append(
         2*(thisData-thisData.mean())/(thisData.max()-thisData.min()) )
   inputData[i]*=weights[i]
   inputDataV+=inputData[i].ravel().tolist() # vectorize

inputDataA=[
   numpy.ma.masked_array(inputData[i],
      actualGeometry.layerMasks[i].sum(axis=0)==0)
         for i in range(actualGeometry.nLayers) ]

print("(done)") ; sys.stdout.flush()

# calculate input vector
randomExV=numpy.take( inputDataV, actualTrimIdx )
randomProjV=numpy.dot( actualSumLayerExM, randomExV )
if removeAllLGStilts:
   # now remove all tilts from the first set of gradients
   nSubaps=gradMs[0].shape[0]/2.0
   for i in range(nAzis[0]*2):
      randomProjV[nSubaps*i:nSubaps*(i+1)]-=\
         randomProjV[nSubaps*i:nSubaps*(i+1)].mean()
            
print("All data prepared") ; sys.stdout.flush()

# now, try straight inversion onto the illuminated portions of the layers 
sTs=numpy.dot(sumLayerExM.transpose(),sumLayerExM)

layerInsertionIdx=reconGeometry.trimIdx(False)
regularisationM=numpy.zeros([layerInsertionIdx[-1][0]]*2)

if regularizationType=='naive':
   raise RuntimeError("NOT TESTED")
##   # \/ Tikhonov regularisation
##   print("Naive")
##   layerInsertionIdx=reconGeometry.trimIdx(False)
##   for i in range(reconGeometry.nLayers):
##      diagIdx=numpy.arange( layerInsertionIdx[i][0],layerInsertionIdx[i+1][0] )\
##            *(layerInsertionIdx[-1][0]+1)
##      regularisationM.ravel()[diagIdx]=1e-3*weights[i]**-2.0
elif regularizationType=='intermediate':
   raise RuntimeError("NOT TESTED")
##   # \/ Intermediate layer restriction
##   print("Intermediate-layer restriction...",end="") ; sys.stdout.flush()
##   layerMapping=[]
##   for actualLh in actualGeometry.layerHeights:
##      i=reconGeometry.layerHeights.searchsorted(actualLh)
##      if i==reconGeometry.nLayers: continue # can't do this one
##      layerMapping.append([actualLh,i])
##      layerMapping[-1].append(
##         (reconGeometry.layerHeights[i]-actualLh
##            )/(reconGeometry.layerHeights[i]-reconGeometry.layerHeights[i-1]))
##
##   regularisationD=numpy.ones(layerInsertionIdx[-1][0])*1e-12
##     # \/ fill diagonal roughly 
##   for lm in layerMapping:
##      diagIdx=\
##         numpy.arange(layerInsertionIdx[lm[1]][0],layerInsertionIdx[lm[1]+1][0])
##            
##      regularisationD[diagIdx]=(1-lm[-1])**2.0*(weights[lm[1]]**2.0) # add weighting
##      if lm[-1]!=0: # add to corresponding points in layer below
##         diagIdx=\
##           numpy.arange(layerInsertionIdx[lm[1]-1][0],layerInsertionIdx[lm[1]][0])
##         regularisationD[diagIdx]+=(lm[-1])**2.0*(weights[lm[1]]**2.0)
##
##   regularisationM.ravel()[
##         numpy.arange(layerInsertionIdx[-1][0])*(layerInsertionIdx[-1][0]+1)]=\
##            1e-3*regularisationD**-0.5
elif regularizationType=='laplacian':
   print("Bi-harmonic approximation...(")
   offset=0
   for i in range(reconGeometry.nLayers):
      tlO=gradientOperator.laplacianOperatorType1(
         pupilMask=reconGeometry.layerMasks[i].sum(axis=0) ).returnOp()
      print("\t",i,offset,tlO.shape[0])
      regularisationM[
         offset:offset+tlO.shape[0],offset:offset+tlO.shape[0]]=\
               1e-5*tlO.dot(tlO.T)*weights[i]**-2.0
      offset+=tlO.shape[0]
   print(")",end="")
   #
elif regularizationType=='SVD':
   print("SVD",end="") 


print("...inverting...",end="") ; sys.stdout.flush()
if regularizationType=='SVD':
   # \/ SVD approach
   sTs_invSVD=numpy.linalg.pinv( sTs, rcond=1e-8 ) # pinv version
   print("pinv, done.") ; sys.stdout.flush()
else:
   if regularisationM.var()==0:
      raise ValueError("Regularisation is zero, was it forgot?")
   sTs_invR=numpy.linalg.inv(sTs + regularisationM ) 
   print("linalg.inv, (done)") ; sys.stdout.flush()

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
   if i==0: pyp.title(titleName+"input vs. reconstruction")
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

# centre projected values
reconCentreProjM=reconGeometry.layerCentreProjectionMatrix().take(
   reconTrimIdx, axis=1 )
actualCentreProjM=actualGeometry.layerCentreProjectionMatrix().take(
   actualTrimIdx, axis=1 )
centreRecoveredV=numpy.dot( reconCentreProjM, recoveredV )
inputCentreV=numpy.dot( actualCentreProjM, randomExV )
for i in range(reconGeometry.nLayers):
   centreRecoveredV[i*nMasks[0]:(i+1)*nMasks[0]]-=\
         centreRecoveredV[i*nMasks[0]:(i+1)*nMasks[0]].mean()
   inputCentreV[i*nMasks[0]:(i+1)*nMasks[0]]-=\
         inputCentreV[i*nMasks[0]:(i+1)*nMasks[0]].mean()

centreMaskedA=numpy.ma.masked_array( 
   numpy.zeros([reconGeometry.nLayers,2]+list(masks[0].shape)),
      [[masks[0]==0]*2]*reconGeometry.nLayers )
if reconGeometry.nLayers==actualGeometry.nLayers:
   pyp.figure(2)
   print("\nCentre proj")
   for i in range(reconGeometry.nLayers):
      pyp.subplot(reconGeometry.nLayers,3,1+i*3)
      centreMaskedA[i,0].ravel()[actualGeometry.maskIdxs[0]]=\
         centreRecoveredV[i*nMasks[0]:(i+1)*nMasks[0]]
      centreMaskedA[i,0]-=centreMaskedA[i,0].mean()
      pyp.imshow( centreMaskedA[i,0]+0.0, interpolation='nearest',
         vmin=-1*weights[i],vmax=1*weights[i] )
      pyp.xlabel("layer={0:1d}".format(i+1))
      pyp.ylabel("recov.")
      
      pyp.subplot(reconGeometry.nLayers,3,2+i*3)
      if i==0: pyp.title(titleName+"centre proj: input vs. reconstruction")
      centreMaskedA[i,1].ravel()[actualGeometry.maskIdxs[0]]=\
         inputCentreV[i*nMasks[0]:(i+1)*nMasks[0]]
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
naiveMeanV=numpy.zeros( nMasks[0], numpy.float64 )
for i in range(nAzi): # create arithmetic mean, by slicing
   naiveMeanV+=randomProjV[i*nMasks[0]:(i+1)*nMasks[0]]
naiveMeanV/=nAzi+0.0
naiveMeanV-=naiveMeanV.mean()

centreSumMaskedA={}
for i in ("actual","recon","naive"):
   centreSumMaskedA[i]=numpy.ma.masked_array( 
      numpy.zeros(masks[0].shape), [masks[0]==0] )
pyp.figure(3)
centreSumMaskedA['actual'].ravel()[actualGeometry.maskIdxs[0]]=actualCentSumV
centreSumMaskedA['recon'].ravel()[actualGeometry.maskIdxs[0]]=reconCentSumV
centreSumMaskedA['naive'].ravel()[actualGeometry.maskIdxs[0]]=naiveMeanV

minMax=( centreSumMaskedA['actual'].ravel().min()
       , centreSumMaskedA['actual'].ravel().max() )
pyp.subplot(2,2,1)            
pyp.title(titleName+"centreSum: recov.")
pyp.imshow( centreSumMaskedA['actual'], interpolation='nearest',
  vmin=minMax[0], vmax=minMax[1] )

pyp.subplot(2,2,2)
pyp.imshow( centreSumMaskedA['recon'], interpolation='nearest',
  vmin=minMax[0], vmax=minMax[1] )
pyp.title(titleName+"CS: orig.")

pyp.subplot(2,2,3)
pyp.imshow( centreSumMaskedA['recon']-centreSumMaskedA['actual'],
  interpolation='nearest',
  vmin=minMax[0], vmax=minMax[1] )
pyp.title(titleName+"diff.")

pyp.subplot(2,2,4)
pyp.imshow( centreSumMaskedA['naive'], interpolation='nearest',
  vmin=minMax[0], vmax=minMax[1] )
pyp.title(titleName+"naive.")

print("\nCentre summed")
print(" Original RMS={0:5.3g}".format(actualCentSumV.var()))
print(" Difference RMS={0:5.3g}".format(
   (actualCentSumV-reconCentSumV).var()))
print(" ( naive difference RMS={0:5.3g} )".format(
   (actualCentSumV-naiveMeanV).var()))
print("\n~GLAO")
print(" Original RMS={0:5.3g}".format(inputCentreV[:nMasks[0]].var()))
print(" Difference RMS={0:5.3g}".format(
   (inputCentreV[:nMasks[0]]-centreRecoveredV[:nMasks[0]]).var()))
print(" ( naive difference RMS={0:5.3g} )".format(
   (inputCentreV[:nMasks[0]]--naiveMeanV).var()))


pyp.figure(4)
pyp.plot( actualCentSumV, reconCentSumV, 'rx',
  label='projected centre, summed' )
pyp.plot( inputCentreV[:nMasks[0]], centreRecoveredV[:nMasks[0]], 'bx',
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
pyp.title(titleName+"Comparison of centre point accuracy/precision")

pyp.waitforbuttonpress()
