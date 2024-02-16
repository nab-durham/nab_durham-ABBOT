from __future__ import print_function
# What is this?
# diagonal cure over arbritrary apertures

assert 1==0, "THIS CODE DOES NOT CURRENTLY FUNCTION"

import abbot.dicure
import abbot.phaseCovariance
import gradNoise_Fried
import time
import pylab,numpy.ma as ma

# \/ configuration here_______________________ 
N=[30,-1] ; N[1]=N[0]//3 # size
r0=2 ; L0=4*r0#N[0]/3.0
noDirectInv=False # if true, don't attempt MMSE
doSparse=True
nrLoopBoundaries=4 # optional 
gradNoiseVarScal=0.5 # multiplier of gradient noise
chainOvlpType=0 # 0=direct x-over, 1=intermediate x-over
nloops=10
   # /\ (0.15 -> best with VK , 0 -> best c. random
   #     WITH NO NOISE)
# /\ _________________________________________ 


print("config:")
np.random.seed( 18071977 )
if gradNoiseVarScal>0:
   noiseReduction=True
   if "nrLoopBoundaries" not in dir():
      nrLoopBoundaries=N[0]+1 # a default value
#      nrSparsifyFrac=0#.25 # fraction to eliminate
else:
   noiseReduction=False
   if "nrLoopBoundaries" not in dir():
      nrLoopBoundaries=N[0]+1
print(" gradNoiseVarScal={0:3.1f}".format(gradNoiseVarScal)) 
print(" noiseReduction={0:d}".format(noiseReduction>0)) 
print(" nrLoopBoundaries={0:s}".format(
      str(np.where(nrLoopBoundaries!=None,nrLoopBoundaries,"")) ))
print(" noiseReduction={0:d}".format(noiseReduction>0)) 
print(" chainOvlpType={0:5.3f}".format(chainOvlpType)) 
print(" N={0:d}".format(N[0])) 
print(" r0={0:d}, L0={1:d}".format(int(r0),int(L0)))
print(" doSparse={0:d}".format(doSparse>0)) 
print(" noDirectInv={0:d}".format(noDirectInv>0)) 

pupAp=Zernike.anyZernike(1,N[0],N[0]//2)

print("gInst...",end="") ; sys.stdout.flush()
gInst=abbot.gradientOperator.gradientOperatorType1(
   pupilMask=pupAp, sparse=doSparse )
pupAp=(gInst.illuminatedCorners>0)*1 # force as integers
gO=gInst.returnOp() # gradient operator matrix
print("(done)") ; sys.stdout.flush()


rdm=np.random.normal(size=gInst.numberPhases) # what to reconstruct
#rdm=np.add.outer(np.arange(N[0]),(N[0]-1)*0+(0)*np.arange(N[0])).ravel().take(
#   gInst.illuminatedCornersIdx ) # 45deg slope
#>    print("phscov.",end="") ; sys.stdout.flush()
#>    directPCOne=abbot.phaseCovariance.covarianceDirectRegular( N[0], r0, L0 )
#>    print(".",end="") ; sys.stdout.flush()
#>    directPC=abbot.phaseCovariance.covarianceMatrixFillInRegular( directPCOne ) 
#>    print(".",end="") ; sys.stdout.flush()
#>    directcholesky=abbot.phaseCovariance.choleskyDecomp(directPC)
#>    print(".",end="") ; sys.stdout.flush()
#>    rdm=np.random.normal(size=N[0]**2) # what to reconstruct
#>    directTestPhase=np.dot(directcholesky, rdm)
#>    print(".",end="") ; sys.stdout.flush()
#>    del directPC,directcholesky,rdm
#>    print(".",end="") ; sys.stdout.flush()
#>    rdm=directTestPhase.ravel().take(gInst.illuminatedCornersIdx) # redo vector
#>    print("(done)") ; sys.stdout.flush()

# ----> begins ---->

print("chainsDefine...",end="") ; sys.stdout.flush()
chainsNumber,chainsDef,chainsDefChStrts=chainsDefine(\
      gInst, boundary=[nrLoopBoundaries]*2, maxLen=None )
print("(done)") ; sys.stdout.flush()

if not doSparse:
   gradV=np.dot(gO, rdm)
else:
   gradV=gO.dot(rdm)

if gradNoiseVarScal>0:
   gradV+=np.random.normal(
         0,gradV.var()**0.5*gradNoiseVarScal,size=len(gradV))
rgradV=rotateVectors(gradV).ravel()
print("(done)") ; sys.stdout.flush()

print("mapping & integrate & vectorize...",end="") ; sys.stdout.flush()
chainsMap=chainsMapping(chainsDef,gInst)
print("...",end="") ; sys.stdout.flush()
chains=chainsIntegrate(chainsDef,rgradV,chainsMap)
print("...",end="") ; sys.stdout.flush()
chainsV,chainsVOffsets=chainsVectorize(chains)
print("(done)") ; sys.stdout.flush()

print("overlaps...",end="") ; sys.stdout.flush()
chainsOvlps=chainsOverlaps(chainsDef,chainOvlpType)
print("(done)") ; sys.stdout.flush()

print("matrices...",end="") ; sys.stdout.flush()
A,B=chainsDefMatrices(chainsOvlps, chainsDef, chainsDefChStrts,
      sparse=doSparse)
print("(done)") ; sys.stdout.flush()

if not doSparse:
   print("inversion.",end="") ; sys.stdout.flush()
   invchOScovM=np.identity(A.shape[1])
   offsetEstM=np.dot( np.dot(
      np.linalg.inv( np.dot( A.T,A )+invchOScovM*1e-2 ), A.T ), -B )
   print(" & discovery",end="") ; sys.stdout.flush()
   offsetEstV=np.dot( offsetEstM, chainsV )
   print("(done)") ; sys.stdout.flush()
else:
   print("offset discovery (sparse CG)...",end="") ; sys.stdout.flush()
   import scipy.sparse
   import scipy.sparse.linalg
   ATA=A.transpose().tocsr().dot(A) ; ATB=A.transpose().tocsr().dot(B)
   ATA=ATA+scipy.sparse.csr_matrix(
         (1e-2*np.ones(chainsDefChStrts[2]),
          range(chainsDefChStrts[2]), range(chainsDefChStrts[2]+1) ) )
   _A=ATA
   _b=ATB.dot( chainsV )
   linalgRes=scipy.sparse.linalg.cg( _A, _b )
   if linalgRes[1]!=0:
      raise ValueError("CG failed, returned {0:d} rather than 0."+
            "Stopping.".format(linalgRes[0]))
   else:
      offsetEstV=-linalgRes[0] # this must be negated
print("(done)") ; sys.stdout.flush()
comp=np.zeros(2*N[0]**2, np.float64)
updates=np.zeros(2*N[0]**2, np.float64)

print("recon...",end="") ; sys.stdout.flush()
# do one way...
for x in range(len(chains[0])):
   toeI=x
   for i in range((chainsDef[0][x][1])):
      comp[ chainsDef[0][x][0][i] ]+=(chains[0][x][i]+offsetEstV[toeI])
      updates[ chainsDef[0][x][0][i] ]+=1
      pass
# ...then another
for x in range(len(chains[1])):
   toeI=chainsDefChStrts[1][2][x]
   for i in range((chainsDef[1][x][1])):
      comp[ N[0]**2+chainsDef[1][x][0][i] ]+=\
            (chains[1][x][i]+offsetEstV[toeI])
      updates[ N[0]**2+chainsDef[1][x][0][i] ]+=1
      pass
print("(done)") ; sys.stdout.flush()

comp.resize([2]+[N[0]]*2)
updates.resize([2]+[N[0]]*2)
updatesViz=ma.masked_array(updates,[pupAp==0]*2)
compViz=[ ma.masked_array(comp[i], updatesViz[i]==0) for i in (0,1) ]
compBothViz=ma.masked_array(
      (1e-10+updatesViz[0]+updatesViz[1])**-1.0*(comp[0]+comp[1]), pupAp==0 )
rdmViz=ma.masked_array(np.zeros(comp[0].shape),pupAp==0)
rdmViz.ravel()[ gInst.illuminatedCornersIdx ]=rdm

# try mmse inversion
print("mmse start...",end="") ; sys.stdout.flush()
lO=abbot.gradientOperator.laplacianOperatorType1(
      pupilMask=pupAp*1,sparse=doSparse )
lM=lO.returnOp()
print("(done)") ; sys.stdout.flush()

if not doSparse:
   print(" inversion...",end="") ; sys.stdout.flush()
   invO=np.dot(np.linalg.inv(np.dot(gO.T,gO)+np.dot(lM.T,lM)*1e-1),gO.T)
   invSol=np.dot(invO,gradV)
   print("(done)") ; sys.stdout.flush()
else:
   print(" sparse CG...",end="") ; sys.stdout.flush()
   _A=gO.transpose().tocsr().dot(gO)
   _A=_A+1e-1*lM.tocsr().dot(lM)
   _b=gO.transpose().tocsr().dot( gradV )
   linalgRes=scipy.sparse.linalg.cg( _A, _b )
   if linalgRes[1]!=0:
      raise ValueError("CG failed, returned {0:d} rather than 0."+
            "Stopping.".format(linalgRes[0]))
   else:
      invSol=linalgRes[0]
   print("(done)") ; sys.stdout.flush()

rdmViz-=rdmViz.mean()

invViz=rdmViz.copy() ; invViz.ravel()[gInst.illuminatedCornersIdx]=invSol
print("(rdm-mmse).var={0:5.3f}".format((rdmViz-invViz).var()))

print("rdm.var={0:5.3f}".format(rdmViz.var()))
print("comp.var={0:5.3f}".format(compBothViz.var()))
print("(rdm-comp).var={0:5.3f}".format((rdmViz-compBothViz).var()))


