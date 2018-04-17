from __future__ import print_function
# Exposition of continuity.py

from continuity import loopsNoiseMatrices
import pylab
import sys
import time
import numpy
import gradientOperator
   
def doSubapMask(roundAp,nfft):
   if roundAp:
      subapCds=numpy.add.outer(
            (numpy.arange(nfft-1)-(nfft-2)/2.)**2.0, 
            (numpy.arange(nfft-1)-(nfft-2)/2.)**2.0 )
      return (subapCds<=(nfft/2-0.5)**2)*\
             (subapCds>(((nfft*6)//39.0)/2-0.5)**2)
   else:
      return numpy.ones([nfft-1]*2,numpy.int32) # square

# -- config begins -----------------------------------
if len(sys.argv)>1:
   nfft=int(sys.argv[1])
else:
   nfft=16
roundAp=True
doSparse=False # True does *not* work at the moment
partitionPeriod=None#[2,2]
partitionPeriodOffset=[0,0]
sparsifyFrac=0#.01 # fraction to eliminate
nReps=100
# -- config ends -------------------------------------
   
subapMask=doSubapMask(roundAp,nfft)
ts=time.time()
gO=gradientOperator.gradientOperatorType1(
    subapMask=subapMask, sparse=doSparse )
tdelta=time.time()-ts ; print("(gO:took {0:3.1f}s)".format(tdelta))
ts=time.time()

# 
if partitionPeriod!=None:
   print("Partitioning\t= ({0[0]:d},{0[1]:d})".format(partitionPeriod))
loopsNoiseReduction=loopsNoiseMatrices(
    subapMask=subapMask, pupilMask=None,
    partitionPeriod=partitionPeriod,
    partitionPeriodOffset=partitionPeriodOffset, rotated=False,
    loopTemplates=([1,2,-1,-2]),
    sparse=doSparse, verbose=False )
tdelta=time.time()-ts ; print("(took {0:3.1f}s)".format(tdelta))
print( "Number subaps/corners={0.numberSubaps:d}/{0.numberPhases:d}".format(
      loopsNoiseReduction))
Nloops=len(loopsNoiseReduction.loopsDef)
Ngradients=loopsNoiseReduction.numberSubaps*2
print("Ngradients,Nloops={0:3d},{1:3d} =>".format(Ngradients, Nloops),end="")
if Ngradients>Nloops:
   print("Under-determined")
elif Ngradients==Nloops:
   print("Well-determined")
elif Ngradients<Nloops:
   print("Over-determined")

#(redundant)   corners=gO.illuminatedCorners!=0
#(old)   print("Loops definition...",end="") ; sys.stdout.flush()
#(old)   print("...loop integration...") ; sys.stdout.flush()
#(old)   loopIntM=loopsIntegrationMatrix( loopsDef, gO, sparse=True )
print("Matrix creation...",end="") ; sys.stdout.flush()
ts=time.time()
noiseExtM,noiseReductionM=loopsNoiseReduction.returnOp()
tdelta=time.time()-ts ; print("(took {0:3.1f}s)".format(tdelta))
print("(done)") ; sys.stdout.flush()
if doSparse:
   print("noiseReductionM!=0 fraction = {0:5.3f}".format(
         noiseReductionM.getnnz()*Ngradients**-2.0 ))
loopIntM=loopsNoiseReduction.loopIntM

if sparsifyFrac!=0:
   def doForceSparsify(sparsifyFrac,ipM):
      # \/ sparsify
      ipM.ravel()[numpy.arange(Ngradients)*(Ngradients+1)]-=1
      maxInM=abs(ipM).max()
      ipM=numpy.where( abs(ipM)>(maxInM*sparsifyFrac),
            ipM, 0 )
      ipM.ravel()[numpy.arange(Ngradients)*(Ngradients+1)]+=1
      return ipM
   noiseReductionM = doForceSparsify(sparsifyFrac,noiseReductionM)
   print("Sparsified by {0:f}".format(sparsifyFrac)) 

ts=time.time()
gM=gO.returnOp()
if doSparse: gM=numpy.array( gM.todense() )
tdelta=time.time()-ts ; print("(took {0:3.1f}s)".format(tdelta))
ts=time.time()
reconM=numpy.dot(
    numpy.linalg.inv( numpy.dot( gM.T,gM )+1e-4*numpy.identity(gO.numberPhases) ), 
    gM.T )
tdelta=time.time()-ts ; print("(took {0:3.1f}s)".format(tdelta))

# input
# \/
#   rdmV=numpy.random.normal(0,1,size=gO.numberPhases)
#   import phaseCovariance as abbotPC
#   directPCOne=abbotPC.covarianceDirectRegular( N, N/4.0, N*10 )
#   directPC=abbotPC.covarianceMatrixFillInMasked( directPCOne, corners )
#   directcholesky=abbotPC.choleskyDecomp(directPC)
#   testipV=numpy.dot(directcholesky, rdmV)
testipV = numpy.zeros( gO.numberPhases )
gradV = numpy.dot( gM, testipV )

# ensemble statistics
# \/
ngradV=[]
avars={
   'grads':gradV.var(),
   'ip_wf_var':testipV.var()
}
nvars={
   'noise':[],
   'left':[],
   'noisy_recon_var':[],
   'less_noisy_recon_var':[],
   'delta_noisy_recon_var':[],
   'delta_less_noisy_recon_var':[],
}
nReconvars=[],[],[]
def _plotFractionalBar(frac,char='#',length=70):
   if frac==1:
      opstr=" "*(length+9)
   else:
      opstr=("[ "+
         char*int(frac*length)+
         "-"*(length-int(frac*length))+
         " ] {0:3d}%".format(int(frac*100)) )
   print( opstr+"\r", end="" )
   sys.stdout.flush()

for i in range(nReps):
   _plotFractionalBar((i+1)*(nReps**-1.0))
   if (i%100)==0: print(".",end="") ; sys.stdout.flush()
   ngradV.append( gradV+numpy.random.normal(0,1,size=Ngradients) )
   loopV=numpy.dot( loopIntM, ngradV[-1] )
   lessngradV=numpy.dot( noiseReductionM, ngradV[-1] )
   nvars['noise'].append(
         (ngradV[-1]-gradV).var() )
   nvars['left'].append(
         (lessngradV-gradV).var() )
   nvars['noisy_recon_var'].append(
         (numpy.dot(reconM,ngradV[-1])).var() )
   nvars['less_noisy_recon_var'].append(
         (numpy.dot(reconM,lessngradV)).var() )
   nvars['delta_noisy_recon_var'].append(
         (numpy.dot(reconM,ngradV[-1])-testipV).var() )
   nvars['delta_less_noisy_recon_var'].append(
         (numpy.dot(reconM,lessngradV)-testipV).var() )

for k in avars.keys(): print("<{0:s}>={1:5.3f}".format(k,numpy.mean(avars[k])))
print("--")
for k in nvars.keys(): print("<{0:s}>={1:5.3f}".format(k,numpy.mean(nvars[k])))

print("remnant gradient noise={0:5.3f}+/-{1:5.3f}".format(
      numpy.mean(nvars['left'])*numpy.mean(nvars['noise'])**-1.0,
      numpy.var(numpy.array(nvars['left'])*numpy.array(nvars['noise'])**-1.0)**0.5
#         (numpy.var(nvars['left'])+
#            (numpy.mean(nvars['left'])**2.0*numpy.mean(nvars['noise'])**-4.0)*
#               numpy.var(nvars['noise']) )**0.5 )
      ))
