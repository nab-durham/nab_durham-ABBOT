from __future__ import print_function
# What is this?
# Projection over N layers, with reconstruction on M using a std reconstructor.
#
# Setup to represent CDRAGON LTAO mode, with a MC investigation 

def _plotFractionalBar(frac,char='#',length=70):
   print(
      "[ "+
      char*int(frac*length)+
      "-"*(length-int(frac*length))+
      " ] {0:3d}%\r".format(int(frac*100)), end="" )
   sys.stdout.flush()

import numpy
import Zernike
import projection
import gradientOperator
import modalBasis
import matplotlib.pyplot as pyp
import commonSeed,hurricaneNames
import sys
import kolmogorov
import argparse

import time
titleName=hurricaneNames.randomName()+":"

parser=argparse.ArgumentParser(
      description='investigate CDRAGON-type LTAO LGS asterism diameters' )
parser.add_argument('-f', help='fix seed', action='store_true')
parser.add_argument('-l', help='LGS no. subaps', type=float, default=30)
parser.add_argument('-n', help='NGS no. subaps', type=float, default=2)
parser.add_argument('--angrangebottom', help='bottom of angle range [arcsec]',
      type=float, default=1)
parser.add_argument('--angrangetop', help='top of angle range [arcsec]',
      type=float, default=8)
parser.add_argument('--angrangeno', help='number of angles to consider',
      type=int, default=8)
parser.add_argument('-N', help='no. of modelled (recovered) layers',
      type=int, default=3)
parser.add_argument('-M', help='no. of input layers',
      type=int, default=3)
parser.add_argument('--heightmax', help='maximum height [m]',
      type=float, default=20e3)
parser.add_argument('-m', help='modal filter', type=float, default=1)
parser.add_argument('--iters', help='number of iterations per angle',
      type=int, default=100)
args=parser.parse_args(sys.argv[1:])
# ------------------------------------------------------------------------
fixSeed=(args.f!=None)
nAzis=(3,1)
nAzi=sum(nAzis)
baseSize=args.l,args.n # then
scalingFactor=4.2*baseSize[0]**-1.0
starHeights=[90e3/scalingFactor]*nAzis[0]+[None]*nAzis[1]
zenAngsRange=(4.64e-6)*numpy.linspace(
      args.angrangebottom,args.angrangetop,args.angrangeno)
##N,M=args.N,args.M
azAngs=[ 2*numpy.pi*(nAzis[0]**-1.0)*i for i in range(nAzis[0]) ]+\
   [ 0 for i in range(nAzis[1]) ] 
heights=[ numpy.linspace(0,args.heightmax/scalingFactor,tNum)
               for tNum in (args.N,args.M) ]
modalFilterEnabled=args.m
numIters=args.iters
# ------------------------------------------------------------------------

if not fixSeed: numpy.random.seed(int(time.time()%1234))
weights=(numpy.random.uniform(0,1,size=100)).tolist()

print("Run is '{0:s}'".format(titleName[:-1]))
print("Seed is "+"not "*(not fixSeed)+"fixed")
if modalFilterEnabled:
   print("Mixed NGS/LGS geometry with modal filtering")
else:
   print("Effective NGS-only geometry")
print("Heights="+str(heights[0])+"/"+str(heights[1]))
print("No. of layers (modelled/input)="+str(args.N)+"/"+str(args.M))
print("zenAngsRange="+str(zenAngsRange))
print("LGS/NGS scale:= {0[0]:d}x{0[0]:d}/{0[1]:d}x{0[1]:d}".format(baseSize))
print("Number of iterations="+str(numIters))
print("---")

def _doGeneralPreparation(baseSize,nAzis):
   _plotFractionalBar(0)
   pixelScales=[1]*nAzis[0]+[ baseSize[0]/baseSize[1] ]*nAzis[1]+[1]

   masks=(
      Zernike.anyZernike(1,baseSize[0],baseSize[0]/2,ongrid=1),
      numpy.ones([baseSize[1]]*2) )
   nMasks=[ int(tM.sum()) for tM in masks ]
   _plotFractionalBar(0.2)

   gradOps=map( lambda ip : 
      gradientOperator.gradientOperatorType1(pupilMask=ip), masks )
   gradMs=map( lambda ip : ip.returnOp(), gradOps )
   nSubaps=gradMs[0].shape[0]/2.0 # for mean tilt removal
   gradAllM=numpy.zeros(
         [ sum(map( lambda ip : gradMs[ip].shape[dirn]*nAzis[ip], (0,1) ))
           for dirn in (0,1) ],
         numpy.float64 )
   _plotFractionalBar(0.4)
   for i,tnAzi in enumerate(nAzis):
      tgradM=gradMs[i]
      for j in range(tnAzi):
         offsets=map( lambda ip : gradMs[0].shape[ip]*(i*nAzis[0]), (0,1) )
         sizes=map( lambda ip : gradMs[i].shape[ip], (0,1) )
         gradAllM[
            offsets[0]+sizes[0]*j: offsets[0]+sizes[0]*(j+1),
            offsets[1]+sizes[1]*j: offsets[1]+sizes[1]*(j+1) ]=tgradM
   _plotFractionalBar(0.6)

   # \/ compute modal bases 
   ZmodalBasis=modalBasis.polySinRadAziBasisType1(
         masks[0], [1],[1,2], orthonormalize=0 )
   modalFiltering=[ 
         thismodalB.reshape([-1,1]).dot(thismodalB.reshape([1,-1]))
            for thismodalB in ZmodalBasis.modalFunctions ]
      # for mask 2 only, for mask 1 equiv to identity (nothing)
   modalFilteringSummed=numpy.array(modalFiltering).sum(axis=0)
   modalFiltAllM=numpy.identity(
            nMasks[0]*nAzis[0]+nMasks[1]*nAzis[1], dtype=numpy.float64)
   _plotFractionalBar(0.8)
   for j in range(nAzis[0]):
      modalFiltAllM[ nMasks[0]*j: nMasks[0]*(j+1),
                     nMasks[0]*j: nMasks[0]*(j+1) ]-=modalFilteringSummed
   _plotFractionalBar(1.0)
   print()
   return( pixelScales,masks,nMasks,gradAllM,modalFiltAllM,nSubaps )

def _doThisPreparation(heights,zenAngs,azAngs,masks,nAzis,starHeights,
      pixelScales,modalFilterEnabled,gradAllM):
   global recoveryM,sumLayerExM,regularisationM,actualSumLayerExM,sumLayerExM,\
      actualProjCentSumM,actualCentreProjM,reconProjCentSumM,reconCentreProjM,\
      actualGeometry,reconGeometry
   _plotFractionalBar(0)
   reconGeometry=projection.projection(
         heights[0],
         zenAngs, azAngs,
         [ masks[0] ]*nAzis[0]+[ masks[1] ]*nAzis[1]+[ masks[0] ],
         starHeights, pixelScales )
   _plotFractionalBar(0.1)
   actualGeometry=projection.projection(
         heights[1],
         zenAngs, azAngs,
         [ masks[0] ]*nAzis[0]+[ masks[1] ]*nAzis[1]+[ masks[0] ],
         starHeights, pixelScales )
   _plotFractionalBar(0.2)

   if not reconGeometry.createLayerMasks(): raise ValueError("Eek! (1)")
   _plotFractionalBar(0.23)
   if not actualGeometry.createLayerMasks(): raise ValueError("Eek! (2)")

   # \/ projection matrices
   # first the layer illumination definitions,
   # then the projection definitions (->trimmed),
   # which are applied to each other,
   # followed by modal filtering and finally
   # gradient calculation.
   _plotFractionalBar(0.25)
   layerExM=reconGeometry.layerExtractionMatrix()
   sumPrM=reconGeometry.sumProjectedMatrix()
   _plotFractionalBar(0.30)
   reconTrimIdx=reconGeometry.trimIdx()
   sumLayerExM=sumPrM.dot(layerExM.take(reconTrimIdx,axis=1))
   _plotFractionalBar(0.35)
   reconCentreProjM=reconGeometry.layerCentreProjectionMatrix().take(
      reconTrimIdx, axis=1 )
   reconProjCentSumM=reconGeometry.sumCentreProjectedMatrix()
      # repeat for the actual geometry of the modelled atmosphere
   _plotFractionalBar(0.40)
   actualLayerExM=actualGeometry.layerExtractionMatrix()
   actualSumPrM=actualGeometry.sumProjectedMatrix()
   _plotFractionalBar(0.45)
   actualTrimIdx=actualGeometry.trimIdx()
   actualSumLayerExM=numpy.dot(
      actualSumPrM, actualLayerExM.take(actualTrimIdx,axis=1) )
   _plotFractionalBar(0.50)
   actualCentreProjM=actualGeometry.layerCentreProjectionMatrix().take(
      actualTrimIdx, axis=1 )
   actualProjCentSumM=actualGeometry.sumCentreProjectedMatrix()
   _plotFractionalBar(0.55)
   if modalFilterEnabled:
      sumLayerExM=modalFiltAllM.dot(sumLayerExM)
      actualSumLayerExM=modalFiltAllM.dot(actualSumLayerExM)
      # apply gradient operator
   sumLayerExM=numpy.dot( gradAllM, sumLayerExM )
   actualSumLayerExM=numpy.dot( gradAllM, actualSumLayerExM )
   _plotFractionalBar(0.60)


               

   # now, try straight inversion onto the illuminated portions of the layers 
   sTs=numpy.dot(sumLayerExM.transpose(),sumLayerExM)

   layerInsertionIdx=reconGeometry.trimIdx(False)
   regularisationM=numpy.zeros([layerInsertionIdx[-1][0]]*2)

   offset=0
   _plotFractionalBar(0.70)
   for i in range(reconGeometry.nLayers):
      tlO=gradientOperator.laplacianOperatorType1(
         pupilMask=reconGeometry.layerMasks[i].sum(axis=0) ).returnOp()
      regularisationM[
         offset:offset+tlO.shape[0],offset:offset+tlO.shape[0]]=\
               1e-3*tlO.dot(tlO.T)*weights[i]**-2.0
      offset+=tlO.shape[0]
      _plotFractionalBar(0.70+(i*reconGeometry.nLayers**-1.0)*0.2)


   _plotFractionalBar(0.9)
   sTs_inv=numpy.linalg.inv(sTs + regularisationM ) 

   recoveryM=numpy.dot( sTs_inv, sumLayerExM.transpose() )
   _plotFractionalBar(1.0)
   print()
   
   return(recoveryM,sumLayerExM,regularisationM,actualSumLayerExM,sumLayerExM,\
      actualProjCentSumM,actualCentreProjM,reconProjCentSumM,reconCentreProjM,\
      actualGeometry,reconGeometry,actualTrimIdx)

def _doThisIter(data,thisIter,numIters,actualGeometry,weights,actualTrimIdx,actualSumLayerExM,reconCentreProjM,reconProjCentSumM,actualCentreProjM,actualProjCentSumM,reconGeometry,nMasks,nAzis,nSubaps):
   global inputDataV,recoveredV,centreRecoveredV,reconCentSumV,\
         inputCentreV,actualCentSumV
   _plotFractionalBar( thisIter*numIters**-1.0, char='O' )
   inputDataV=[]
   for i in range(actualGeometry.nLayers):
      tS=actualGeometry.layerNpix[i]
      thisData=kolmogorov.TwoScreens(tS.max()*2,
               (nMasks[0]**0.5)/2.0)[0][:tS[0],:tS[1]]
      inputData=2*(thisData-thisData.mean())/(thisData.max()-thisData.min())
      inputData[i]*=weights[i]
      inputDataV+=inputData.ravel().tolist() # vectorize

   ##inputDataA=[
   ##   numpy.ma.masked_array(inputData[i],
   ##      actualGeometry.layerMasks[i].sum(axis=0)==0)
   ##         for i in range(actualGeometry.nLayers) ]
   randomExV=numpy.take( inputDataV, actualTrimIdx )
   # calculate input vector
   randomProjV=numpy.dot( actualSumLayerExM, randomExV )
   for i in range(nAzis[0]*2):# remove mean tilt from each dirn for each LGS WFS
      randomProjV[nSubaps*i:nSubaps*(i+1)]-=\
         randomProjV[nSubaps*i:nSubaps*(i+1)].mean()
   recoveredV=numpy.dot( recoveryM, randomProjV )
   ##recoveredLayersA=[
   ##   numpy.ma.masked_array(
   ##      numpy.zeros(reconGeometry.layerNpix[i], numpy.float64),
   ##      reconGeometry.layerMasks[i].sum(axis=0)==0)
   ##      for i in range(reconGeometry.nLayers) ]
   ##
   ##for i in range(reconGeometry.nLayers):
   ##   recoveredLayersA[i].ravel()[layerInsertionIdx[i][1]]=\
   ##     recoveredV[layerInsertionIdx[i][0]:layerInsertionIdx[i+1][0]]
   ##   recoveredLayersA[i]-=recoveredLayersA[i].mean()

   # centre projected values
   centreRecoveredV=numpy.dot( reconCentreProjM, recoveredV )
   reconCentSumV=numpy.dot( reconProjCentSumM, centreRecoveredV )
   #
   inputCentreV=numpy.dot( actualCentreProjM, randomExV )
   actualCentSumV=numpy.dot( actualProjCentSumM, inputCentreV )
   for i in range(reconGeometry.nLayers):
      centreRecoveredV[i*nMasks[0]:(i+1)*nMasks[0]]-=\
            centreRecoveredV[i*nMasks[0]:(i+1)*nMasks[0]].mean()
      inputCentreV[i*nMasks[0]:(i+1)*nMasks[0]]-=\
            inputCentreV[i*nMasks[0]:(i+1)*nMasks[0]].mean()

   data['centreIpVar'].append(
         actualCentSumV.var() )
   data['centreReconVar'].append(
         reconCentSumV.var() )
   data['centreDiffVar'].append(
         (actualCentSumV-reconCentSumV).var() )
   data['glaoIpVar'].append(
         inputCentreV[:nMasks[0]].var() )
   data['glaoReconVar'].append(
         centreRecoveredV[:nMasks[0]].var() )
   data['glaoDiffVar'].append(
         (inputCentreV[:nMasks[0]]-centreRecoveredV[:nMasks[0]]).var() )

   return( data,inputDataV,recoveredV,centreRecoveredV,reconCentSumV,\
         inputCentreV,actualCentSumV )

# === logic sits here ===

(pixelScales,masks,nMasks,gradAllM,modalFiltAllM,nSubaps
      )=_doGeneralPreparation(baseSize,nAzis)

datas=[]
for zangIdx,thisZang in enumerate(zenAngsRange):
   zenAngs=[thisZang]*nAzis[0]+[0]*nAzis[1]
   (recoveryM,sumLayerExM,regularisationM,actualSumLayerExM,sumLayerExM,\
       actualProjCentSumM,actualCentreProjM,reconProjCentSumM,reconCentreProjM,\
       actualGeometry,reconGeometry,actualTrimIdx
     )=_doThisPreparation(heights,zenAngs,azAngs,masks,nAzis,starHeights,
            pixelScales,modalFilterEnabled,gradAllM)
   #
   data={
      'centreIpVar': [],
      'centreReconVar': [],
      'centreDiffVar': [],
      'glaoIpVar': [],
      'glaoReconVar': [],
      'glaoDiffVar': [],
   }
   for thisIter in range(numIters):
      (data,inputDataV,recoveredV,centreRecoveredV,reconCentSumV,\
            inputCentreV,actualCentSumV)=_doThisIter(
         data,thisIter,numIters,
         actualGeometry,weights,actualTrimIdx,actualSumLayerExM,
         reconCentreProjM,reconProjCentSumM,actualCentreProjM,
         actualProjCentSumM,reconGeometry,nMasks,nAzis,nSubaps)
   #
   print("\t>> done {0:d}/{1:d}\n".format(zangIdx+1,args.angrangeno))
   datas.append((zangIdx,thisZang,data))
