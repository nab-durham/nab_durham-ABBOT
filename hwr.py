# What is this?
# HWR over arbitrary apertures

from __future__ import print_function
import numpy as numpy
import abbot.gradientOperator
import sys
import pdb
# from (x,y) to (+x/-y,+x/+y)
rotateVectors=lambda grads : numpy.array(
   [ grads[:grads.shape[0]/2]*2**-0.5+grads[grads.shape[0]/2:]*2**-0.5, 
     grads[:grads.shape[0]/2]*-2**-0.5+grads[grads.shape[0]/2:]*2**-0.5 ])

class counter(object):
  n=0
  def cb(self,ip):
    self.n+=1

def chainsDefine( gInst, maxLen=None, boundary=None, shortChains=False ):
   chains=[]
      # \/ define the chains
      # Let the 1st method be to zip through the subaperture index into the
      # pupil mask and find which sub-aperture corners are in the
      # mask2allcorners and that this element is the bottom-left (thus there
      # should also be four partners).
      # Let the 2nd method be similar but mask2allcorners+n_ (top left).
   N=gInst.n_ # alias
   blank=numpy.zeros([2]+N)
   chainsNumber=0
   chaninsStrtNum=0
   chainsDef=[],[]
   chainsStarts=[ ([],[],[]), ([],[],[]), 0]
   insubap=False
   for i in range(N[1]+N[0]-1):
      # top left, upwards right
      y=int(numpy.where( N[0]-i-1<0, 0, N[0]-i-1 )) # starting y, then x
      x=int(numpy.where( i-N[0]+1<0, 0, i-N[0]+1 ))
      end=int(numpy.where( N[0]-y<N[1]-x, N[0]-y, N[1]-x ))
      if insubap:
         # terminate chain
         chainsDef[0].append( newChain )
      insubap=False
      for j in range(end):
         thisIdx=y*N[1]+x+j*(N[0]+1)
         if insubap:
            forceTerminate=False
            if type(None)!=type(maxLen):  # terminate if too long
               if maxLen==newChain[1]:
                  forceTerminate=True
            if type(None)!=type(boundary) and type(None)!=type(boundary[0]):
               # terminate if reached a boundary
#                  # \/ horizontal, vertical boundaries, offset
#               if ((thisIdx%N[0])%boundary[0]==int(boundary[0]/2-1)) or\
#                     ((thisIdx//N[0])%boundary[0]==int(boundary[0]/2-1)):
                  # \/ horizontal, vertical boundaries, common
               if ((thisIdx%N[0])%boundary[0]==0) or\
                     ((thisIdx//N[0])%boundary[0]==0):
                  # \/ diagonal boundaries
#               if ((thisIdx%N[0])%boundary[0]==((N[0]-thisIdx//N[0])%boundary[0])):
                  forceTerminate=True
            if ( thisIdx in gInst.illuminatedCornersIdx and
                  thisIdx+N[1] in gInst.illuminatedCornersIdx and
                  thisIdx+N[1]+1 in gInst.illuminatedCornersIdx and
                  thisIdx+1 in gInst.illuminatedCornersIdx and
                  thisIdx%N[0]<N[1]-1 and
                  gInst.subapMask[thisIdx//N[1],thisIdx%N[0]]!=0
                  ) and not forceTerminate:
               # continue chain
               newChain[0].append( thisIdx ) ; newChain[1]+=1
            elif thisIdx in gInst.illuminatedCornersIdx:
               # must terminate chain but include this point
               if not (forceTerminate and shortChains):
                  newChain[0].append( thisIdx ) ; newChain[1]+=1
               insubap=False
               chainsDef[0].append( newChain )
            else:
               # terminate chain
               insubap=False
               chainsDef[0].append( newChain )
         if not insubap:
            if ( thisIdx in gInst.illuminatedCornersIdx and
                  thisIdx+N[1] in gInst.illuminatedCornersIdx and
                  thisIdx+N[1]+1 in gInst.illuminatedCornersIdx and
                  thisIdx+1 in gInst.illuminatedCornersIdx and
                  thisIdx%N[0]<N[1]-1 and
                  gInst.subapMask[thisIdx//N[1],thisIdx%N[0]]!=0
                  ):
               # start chain
               chainsNumber+=1
               insubap=True
               newChain=[[thisIdx],1, int(chainsNumber)]
         
               # \/ will always be a new entry
               if thisIdx in chainsStarts[0][1]:
                  # so this should never happen
                  raise ValueError("Gone bonkers")
               chainsStarts[0][0].append(chainsNumber)
               chainsStarts[0][1].append(thisIdx)
               chainsStarts[0][2].append(chaninsStrtNum)
               chaninsStrtNum+=1

   for i in range(N[1]+N[0]-1):
      # top right, downwards left
      y=int(numpy.where( N[0]-i-1<0, 0, N[0]-i-1 )) # starting y, then x
      x=int(numpy.where( N[0]-i-1>0, N[1]-1, N[1]-1+N[0]-i-1 ))
      end=int(numpy.where( N[0]-y<=x, N[0]-y, x+1 ))
      if insubap:
         # terminate chain
         chainsDef[1].append( newChain )
      insubap=False
      for j in range(end):
         thisIdx=y*N[1]+x+j*(N[0]-1)
         if insubap:
            forceTerminate=False
            if type(None)!=type(maxLen):
               if maxLen==newChain[1]:
                  forceTerminate=True
            if type(None)!=type(boundary) and type(None)!=type(boundary[1]):
               # terminate if reached a boundary
#                  # \/ horizontal/vertical boundaries, offset
#               if ((thisIdx%N[0])%boundary[1]==1) or\
#                     ((thisIdx//N[0])%boundary[1]==(boundary[1]-1)):
                  # \/ horizontal/vertical boundaries, common
               if ((thisIdx%N[0])%boundary[1]==(0)) or\
                     ((thisIdx//N[0])%boundary[1]==0):
                  # \/ diagonal boundaries
#               if ((thisIdx%N[0])%boundary[1]==((thisIdx//N[0])%boundary[1])):
                  forceTerminate=True
            if ( thisIdx in gInst.illuminatedCornersIdx and
                  thisIdx+N[1] in gInst.illuminatedCornersIdx and
                  thisIdx+N[1]-1 in gInst.illuminatedCornersIdx and
                  thisIdx-1 in gInst.illuminatedCornersIdx and
                  ((thisIdx%N[0]<(N[0]-1) and
                   gInst.subapMask[thisIdx//N[1],thisIdx%N[0]-1]!=0))
                  ) and not forceTerminate:
               # continue chain
               newChain[0].append( thisIdx ) ; newChain[1]+=1
            elif thisIdx in gInst.illuminatedCornersIdx:
               # must terminate chain but include this point
               if not (forceTerminate and shortChains):
                  newChain[0].append( thisIdx ) ; newChain[1]+=1
               insubap=False
               chainsDef[1].append( newChain )
            else:
               # terminate chain
               insubap=False
               chainsDef[1].append( newChain )
         if not insubap:
            if ( thisIdx in gInst.illuminatedCornersIdx and
                  thisIdx+N[1] in gInst.illuminatedCornersIdx and
                  thisIdx+N[1]-1 in gInst.illuminatedCornersIdx and
                  thisIdx-1 in gInst.illuminatedCornersIdx and
                  thisIdx%N[0]>0
                  and ((thisIdx%N[0]>(0) and
                   gInst.subapMask[thisIdx//N[1],(thisIdx%N[0])-1]!=0))
                  ):
               # start chain
               chainsNumber+=1
               insubap=True
               newChain=[[thisIdx],1,int(chainsNumber)]
              
               # \/ principle code to check overlaps of chains
               if thisIdx in chainsStarts[0][1]:
                  # have an overlap so direct there
                  chainsStarts[1][2].append(
                        chainsStarts[0][2][
                           chainsStarts[0][1].index(thisIdx)
                        ])
               elif thisIdx in chainsStarts[1][1]:
                  # should never happen
                  raise ValueError("Gone bonkers")
               else:
                  chainsStarts[1][2].append(chaninsStrtNum)
                  chaninsStrtNum+=1
                  
               # \/ and store which chain we are talking about
               chainsStarts[1][0].append(chainsNumber)
               chainsStarts[1][1].append(thisIdx)

   chainsStarts[-1]=chaninsStrtNum
   return chainsNumber, chainsDef, chainsStarts

def chainsMapping( chainsDef, gInst ):
   chainsMap=([],[])
   for i in range(len(chainsDef[0])):
      tcM=[]
      for j in range(chainsDef[0][i][1]-1): # never use first element, defined as zero 
         tC=chainsDef[0][i][0][j] # chain index
         tgC=(gInst.mask2allcorners==tC).nonzero()[0]  # index into gradient vector
         if type(tgC)==None or len(tgC)==0:
            raise ValueError("Failed to index gradient")
         if len(tgC)>1:
            raise ValueError("Conflict!")
         tcM.append(int(tgC))
      chainsMap[0].append( tcM )
   for i in range(len(chainsDef[1])):
      tcM=[]
      for j in range(chainsDef[1][i][1]-1): # never use first element, defined as zero 
         tC=chainsDef[1][i][0][j] # chain index
         tgC=gInst.numberSubaps+(gInst.mask2allcorners==(tC-1)).nonzero()[0]
         if type(tgC)==None:
            raise ValueError("Failed to index gradient")
         if len(tgC)>1:
            raise ValueError("Conflict!")
         tcM.append(int(tgC))
      chainsMap[1].append( tcM )
   return chainsMap

def chainsIntegrate( chainsDef, rgradV, chainsMap ):
   chains=([],[])
   for j in (0,1):
      for i in range(len(chainsDef[j])):
         chains[j].append(numpy.array(
               [0]+list(numpy.cumsum( rgradV.take(chainsMap[j][i])*2**0.5 )) ))
   return chains

def chainsVectorize( chains ):
   # \/ produce the chain vector and indexing into it
   chainsV=[]
   chainsVOffsets=[]
   for dirn in (0,1):
      for tchain in chains[dirn]:
         chainsVOffsets.append( len(chainsV) )
         chainsV+=tchain.tolist()
   return numpy.array(chainsV), numpy.array(chainsVOffsets)

def chainsOverlaps( chainsDef, intermediate=1 ):
   # Solution approach is: each chain has an offset, as does the orthogonal
   # chain at the edge where they meet. The system can be pinned down by
   # defining the chain values plus their offsets to be equal, or equivalently,
   # their difference to be zero. Note that some pairs of overlapping 
   # chains share offsets.  This latter fact constrains the reconstruction. 
   #
   # An alternative approach which can quash high-frequencies is to also
   # overlap at chain intermediate points. This results in only one grid, not
   # the usual 2 nested grids. The argument 'intermediate' adds this grid with
   # the value being the scaling relative to the chain-offset grids.
   # Thus very small values weight the chain-offset grids solution, whereas
   # very large values weight the chain intermediate-grid solution.
   chainsMatching=[]
   overlap=0
   for i in range(len(chainsDef[0])): # index the chains we want to align
      for k in range(0, chainsDef[0][i][1], 1):
            # /\ can choose which points along a chain to try and match
         if ((chainsDef[0][i][0][k]//20)>10): # NEW
            print("X",end=":")
            continue
         for j in range(len(chainsDef[1])): # index the orthogonal chains
            for l in range(0, chainsDef[1][j][1], 1):
               print((chainsDef[0][i][0][k]//20),end=",")
#               if not intermediate and\
               if intermediate>0 and k<(chainsDef[0][i][1]-1) and\
                     l<(chainsDef[1][j][1]-1) and\
                     (chainsDef[0][i][0][k]==(chainsDef[1][j][0][l]-1) and
                      chainsDef[0][i][0][k+1]==(chainsDef[1][j][0][l+1]+1)):
         # \/ for each consecutive pair of elements in one set of chains, define
         #   an intermediate point and find the matching intermediate points
         #   from the in the other set of chains.
#                  chainsMatching.append( (i,j,k,l,intermediate) )
#                  chainsMatching.append( (i,j,k+1,l+1,intermediate) )
                  chainsMatching.append([(i,j,k,l,intermediate)
                                        ,(i,j,k+1,l+1,intermediate)] )
               if chainsDef[0][i][0][k]==chainsDef[1][j][0][l]:
         # \/ for each point in one set of chains, find matching in
         # the other
                  chainsMatching.append( [(i,j,k,l,1)] )
   # \/ nb There is an inefficiency here in so much that if i,j are already
   #  in one overlap i.e. intermediate>0, then it should be combined rather
   #  than added as a separate row.
   return chainsMatching

def chainsDefMatrices(chainsOvlps, chainsDef, chainsDefChStrts, gO=None,
      sparse=False):
   # If the problem is Ao+Bc=0, where o is the vector of offsets for the
   # chains, which are essentially the currently undefined start value for each
   # chain, and c is the vector of chain values (chainsV here), then A is the
   # matrix which selects the appropriate offsets and B selects the appropriate
   # chain differences which ought to be zero thus the equation.  ->
   # o=A^{+}B(-c) where A^{+} is a pseudo-inverse. Or you can use the
   # least-squares approach, o=(A^{T}A+{\alpha}I)^{-1}A^{T}B(-c) and standard
   # matrix algebra works, as long as \alpha!=0 i.e. regularisation is applied.

   # only need to consider unique offsets; some chains will start at the same
   # point so that would overconstrain the matrix. 
   avoidedRows=[] # record overlaps of chains with the same offset 
   chainsLen=0
   chainsVOffsets=[]
   for dirn in (0,1):
      for tcD in chainsDef[dirn]:
         chainsVOffsets.append(chainsLen)
         chainsLen+=tcD[1]
   if not sparse: # define matrices first
      A=numpy.zeros([ len(chainsOvlps), chainsDefChStrts[2] ], numpy.float32)
      B=numpy.zeros([ len(chainsOvlps), chainsLen ], numpy.float32 )
   else:
      Aidx={'data':[], 'col':[] }
      Bidx={'data':[], 'col':[] }
      rowIdx=[]
   for i in range(len(chainsOvlps)):
         # *** the method assumes that every row is defined by each element
         # *** of the overlaps 
      #
      for j in range(len(chainsOvlps[i])):
         tcO=chainsOvlps[i][j]
#      if tcO[1]==0 and tcO[0]==0: continue
         tcCN=[ chainsDef[dirn][tcO[dirn]][2] for dirn in (0,1) ]
         coI=[ chainsDefChStrts[dirn][0].index( tcCN[dirn] ) for dirn in (0,1) ]
         if chainsDefChStrts[0][2][coI[0]]==chainsDefChStrts[1][2][coI[1]]:
            avoidedRows.append([i,len(chainsOvlps[i])])
#            print("Avoiding row {0:d}".format(i))
            continue
         if not sparse: # fill in here
            A[ i, chainsDefChStrts[0][2][coI[0]]]+=tcO[-1]
            A[ i, chainsDefChStrts[1][2][coI[1]]]+=-tcO[-1]
            #
            B[ i, chainsVOffsets[chainsDef[0][tcO[0]][-1]-1]+tcO[2] ]+=tcO[-1]
            B[ i, chainsVOffsets[chainsDef[1][tcO[1]][-1]-1]+tcO[3] ]+=-tcO[-1]
         else:
            Aidx['data']+=[tcO[-1],-tcO[-1]]
            Aidx['col']+=[ chainsDefChStrts[j][2][coI[j]] for j in (0,1) ]
            rowIdx+=[i-len(avoidedRows)]*2
            Bidx['data']+=[tcO[-1],-tcO[-1]]
            Bidx['col']+=[ chainsVOffsets[chainsDef[j][tcO[j]][-1]-1]+tcO[2+j]
               for j in (0,1) ]

   if sparse:
      import scipy.sparse
      A=scipy.sparse.csr_matrix( (Aidx['data'],(rowIdx,Aidx['col'])),
            [ (len(chainsOvlps)-len(avoidedRows)), chainsDefChStrts[2] ], numpy.float32 )
      B=scipy.sparse.csr_matrix( (Bidx['data'],(rowIdx,Bidx['col'])),
            [ (len(chainsOvlps)-len(avoidedRows)), chainsLen ], numpy.float32)
   else:
      rowIndex=range(len(chainsOvlps))
      for i in avoidedRows[::-1]: # in reverse order
         rowIndex.pop(i[0])
      A=A[rowIndex] ; B=B[rowIndex]

   return A,B

# \/ below follow two helper functions, which make it a bit easier to just
# do hwr, although keep in mind that they may not be right for your use.

from scipy.sparse.linalg import cg as spcg
from scipy.sparse import identity as spidentity

def prepHWR(gO,maxLen=None,boundary=None,overlapType=1,sparse=False):
   chainsNumber,chainsDef,chainsDefChStrts=\
         chainsDefine( gO, maxLen=maxLen, boundary=boundary )
   chainsOvlps=chainsOverlaps(chainsDef,overlapType)
   chainsMap=chainsMapping(chainsDef,gO)
   A,B=chainsDefMatrices(chainsOvlps, chainsDef, chainsDefChStrts, gO, sparse)
#   geometry=numpy.zeros(gO.n_,numpy.float32)
#   for tcDCS in chainsDefChStrts[:-1]:
#       for x in range(len(tcDCS[1])):
#         geometry.ravel()[tcDCS[1][x]]=1   
#   mappingM=numpy.zeros([geometry.sum()]*2, numpy.float32)
#   for tcDCS in chainsDefChStrts[:-1]:
#       for x in range(len(tcDCS[1])):
#         mappingM[ geometry.ravel().nonzero()[0].searchsorted(tcDCS[1][x]),
#                  tcDCS[2][x] ]=1
#   lO=abbot.gradientOperator.laplacianOperatorPupil(geometry)
#   lM=numpy.dot(lO.op,mappingM)
#      # /\ the map converts from offset position to physical position
#      #  increasing in the 1D sense, within the array.
#   lTlM=numpy.dot( lM.T,lM )
   if not sparse:
      zeroRows=(abs(A).sum(axis=1)==0).nonzero()[0]
      if len(zeroRows)>0:
         nonzeroRows=(abs(A).sum(axis=1)!=0).nonzero()[0]
         A=A.take(nonzeroRows,axis=0)
         B=B.take(nonzeroRows,axis=0)
         print("** REMOVING ROWS FROM A,B:"+str(zeroRows))
      #offsetEstM=numpy.dot( numpy.linalg.pinv( A, rcond=0.1 ), -B )
      #offsetEstM=numpy.dot( numpy.dot(
      #      numpy.linalg.inv( numpy.dot( A.T,A )+lTlM*1e-2 ), A.T ), -B )
      offsetEstM=numpy.dot( numpy.dot(
            numpy.linalg.inv( numpy.dot( A.T,A )
           +numpy.identity(A.shape[1])*1e-2 ), A.T ), -B )
   else:
      ATA=A.transpose().tocsr().dot(A) ; ATB=A.transpose().tocsr().dot(B)
      ATA=ATA+spidentity(chainsDefChStrts[2],'d','csr')*1e-2
# >      ATA=ATA+scipy.sparse.csr_matrix(
# >            (1e-2*numpy.ones(len(chainsOvlps)),
# >             range(len(chainsOvlps)), range(len(chainsOvlps)+1) ) )
      offsetEstM=( ATA, ATB ) # return as a tuple for CG algorithm to use
   #
   return(chainsDef,chainsDefChStrts,chainsMap,offsetEstM)

def localWaffle(x,gO):
  mask=numpy.ones(gO.numberPhases)
  for row in (-1,0,1):
    for col in (-1,0,1):
      if col+x%gO.n_[0]>=gO.n_[0]: continue
      if col+x%gO.n_[0]<0: continue
      if row+x//gO.n_[0]>=gO.n_[1]: continue
      if row+x%gO.n_[0]<0: continue
      thisact=row*gO.n_[0]+col+x
      if thisact in gO.illuminatedCornersIdx:
         mask[ gO.illuminatedCornersIdx.searchsorted(thisact) ]=0
 
  maskI=mask.nonzero()[0]
  waffle=numpy.zeros([2,gO.numberPhases],numpy.float64)
  waffle[0,maskI]=(gO.illuminatedCornersIdx[maskI]%2
      +gO.illuminatedCornersIdx[maskI]//gO.n_[0])%2
  waffle[1]=(gO.illuminatedCornersIdx%2+gO.illuminatedCornersIdx//gO.n_[0])%2
  waffle[0,maskI]-=numpy.mean(waffle[0,maskI]**2.0)
  waffle[1]-=numpy.mean(waffle[1]**2.0)
  norms=numpy.array([ numpy.sum(waffle[0,maskI]**2.0),numpy.sum(waffle[1]**2.0)]
      ).reshape([2,1])

  return waffle*norms**-0.5

def localPiston(x,gO):
  mask=numpy.ones(gO.numberPhases)
  for row in (-1,0,1):
     for col in (-1,0,1):
       if col+x%gO.n_[0]>=gO.n_[0]: continue
       if col+x%gO.n_[0]<0: continue
       if row+x//gO.n_[0]>=gO.n_[1]: continue
       if row+x%gO.n_[0]<0: continue
       thisact=row*gO.n_[0]+col+x
       if thisact in gO.illuminatedCornersIdx:
          mask[ gO.illuminatedCornersIdx.searchsorted(thisact) ]=0

  maskI=mask.nonzero()[0]
  localPiston=numpy.zeros([2,gO.numberPhases],numpy.float64)
  localPiston[0,maskI]=len(maskI)**-0.5
  localPiston[1]=len(maskI)**-0.5
  return localPiston

def doHWROnePoke(gradsV,chainsDef,gO,offsetEstM,chainsDefChStrts,chainsMap,
      thisAct,doWaffleReduction=1,doPistonReduction=1):
#   comp,numbers,ts=doHWRIntegration(
   comp,numbers=doHWRIntegration(
         gradsV, chainsDef, gO, offsetEstM, chainsDefChStrts,chainsMap )
   #
   hwrV=((comp[0]+comp[1])*(numbers+1e-9)**-1.0
            ).ravel()[gO.illuminatedCornersIdx]
   if type(thisAct)!=type(None):
      if doWaffleReduction:
         localWV=localWaffle(thisAct,gO)
         hwrV-=numpy.dot( localWV[0], hwrV )*localWV[1]
      if doPistonReduction:
         localPV=localPiston(thisAct,gO)
         hwrV-=numpy.dot( localPV[0], hwrV )*localPV[1]
   #
   return comp,hwrV#,numpy.array(ts)[1:]-ts[0]

#import time
def doHWRIntegration(gradsV,chainsDef,gO,offsetEstM,chainsDefChStrts,
      chainsMap,sparse=False):
   rgradV=rotateVectors(gradsV).ravel()
   chains=chainsIntegrate(chainsDef,rgradV,chainsMap)
   chainsV,chainsVOffsets=chainsVectorize(chains)
#*   print("** len[chainsV,chainsVoffsets]=[{0:d},{1:d}]".format(
#*         len(chainsV),len(chainsVOffsets)))
#*   print("** len[chains[0],chains[1]]=[{0:d},{1:d}]".format(
#*         len(chains[0]),len(chains[1])))
   if not sparse:
      offsetEstV=numpy.dot( offsetEstM, chainsV )
   else:
      offsetEstV=spcg( offsetEstM[0], offsetEstM[1].dot(chainsV) )
      if offsetEstV[1]==0:
         offsetEstV=offsetEstV[0]
      else:
         raise ValueError("Sparse CG did not converge")
   
   chainsV=doHWRChainsAddOffsets(
         chains,chainsV,chainsVOffsets,chainsDef,chainsDefChStrts,offsetEstV)
   
   return doHWRChainsFormatToConventional(
         gO,chains,chainsDef,chainsV,chainsVOffsets)

def doHWRChainsAddOffsets(chains,chainsV,chainsVOffsets,chainsDef,
      chainsDefChStrts,offsetEstV):
   #      
   # do one way...
   for x in range(len(chains[0])):
      toeI=x
      chainsV[chainsVOffsets[x]:chainsVOffsets[x]+chainsDef[0][x][1]]+=\
            offsetEstV[toeI]
   # ...then the other
   for x in range(len(chains[1])):
      toeI=chainsDefChStrts[1][2][x]
      chainsV[chainsVOffsets[x+len(chains[0])]:
              chainsVOffsets[x+len(chains[0])]+chainsDef[1][x][1]]+=\
            offsetEstV[toeI]
   return chainsV

def doHWRChainsFormatToConventional(
      gO,chains,chainsDef,chainsV,chainsVOffsets):
   comp=numpy.zeros([2,gO.n_[0]*gO.n_[1]], numpy.float64)
   numbers=numpy.zeros([gO.n_[0]*gO.n_[1]], numpy.float64)
   for x in range(len(chains[0])):
      comp[0][ chainsDef[0][x][0] ]=\
         chainsV[chainsVOffsets[x]:chainsVOffsets[x]+chainsDef[0][x][1]]
      numbers[ chainsDef[0][x][0] ]+=1
   for x in range(len(chains[1])):
      comp[1][ chainsDef[1][x][0] ]=\
         chainsV[chainsVOffsets[x+len(chains[0])]:\
                 chainsVOffsets[x+len(chains[0])]+chainsDef[1][x][1]]
      numbers[ chainsDef[1][x][0] ]+=1
   return comp,numbers

def doHWRGeneral(gradsV,chainsDef,gO,offsetEstM,chainsDefChStrts,chainsMap,
      doWaffleReduction=1,doPistonReduction=1,sparse=False):
   comp,numbers=doHWRIntegration( gradsV, chainsDef, gO, offsetEstM,
         chainsDefChStrts, chainsMap, sparse )
   #
   hwrV=((comp[0]+comp[1])*(numbers+1e-9)**-1.0
            ).ravel()[gO.illuminatedCornersIdx]
   if doWaffleReduction:
      globalWV=localWaffle(-1,gO)
      hwrV-=numpy.dot( globalWV[1], hwrV )*globalWV[1]
   if doPistonReduction:
      globalPV=localPiston(-1,gO)
      hwrV-=numpy.dot( globalPV[1], hwrV )*globalPV[1]
   #
   return comp,hwrV
# --begin--

if __name__=="__main__":

   import pdb # Python debugger
   import abbot.phaseCovariance
   import gradNoise_Fried
   import time
   import matplotlib.pyplot as pyp, numpy.ma as ma

# \/ configuration here_______________________ 
   N=[20,0] ; N[1]=(N[0]*6)//39. # size
   r0=2 ; L0=4*r0#N[0]/3.0
   noDirectInv=False # if true, don't attempt MMSE
   doSparse=0
   chainPeriodicBound=[8,N[0]+1][0]# optional 
   chainMaxLength=None # better to use periodic-boundaries than fixed lengths
   gradNoiseVarScal=5e-1 # multiplier of gradient noise
   chainOvlpType=0.01     # 0=direct x-over, 1=intermediate x-over
      # /\ (0.15 -> best with VK , 0 -> best c. random
      #     WITH NO NOISE)
   dopinv=True#False 
   doShortChains=False # True means do not include chain bounday truncation overlaps
   disableNoiseReduction=True#False
   contLoopBoundaries=[ N[0]+1, chainPeriodicBound ][0]
   laplacianSmoother=1e-9
   fractionalZeroingoeM=0 # always keep this as zero 
      # (fraction of max{offsetEstM} below which to zero)
# /\ _________________________________________ 


   print("config:")
   numpy.random.seed( 18071977 )
   if gradNoiseVarScal>0 and not disableNoiseReduction:
      noiseReduction=True
      if "contLoopBoundaries" not in dir():
         contLoopBoundaries=N[0]+1 # a default value
#      nrSparsifyFrac=0#.25 # fraction to eliminate
   else:
      noiseReduction=False
      if "contLoopBoundaries" not in dir():
         contLoopBoundariesicBound=N[0]+1
   print(" gradNoiseVarScal={0:3.1f}".format(gradNoiseVarScal)) 
   print(" chainPeriodicBound={0:s}".format(
         str(numpy.where(chainPeriodicBound!=None,chainPeriodicBound,"")) ))
   print(" contLoopBoundaries={0:s}".format(
         str(numpy.where(contLoopBoundaries!=None,contLoopBoundaries,"")) ))
   print(" chainMaxLength={0:s}".format(
         str(numpy.where(chainMaxLength!=None,chainMaxLength,"")) ))
   print(" noiseReduction={0:d}".format(noiseReduction>0)) 
   print(" chainOvlpType={0:5.3f}".format(chainOvlpType)) 
   print(" N={0:d}".format(N[0])) 
   print(" r0={0:d}, L0={1:d}".format(int(r0),int(L0)))
   print(" doSparse={0:d}".format(doSparse>0)) 
   print(" noDirectInv={0:d}".format(noDirectInv>0)) 
   print(" dopinv={0:d}".format(dopinv>0)) 

   cds=numpy.add.outer(
         (numpy.arange(N[0])-(N[0]-1)/2.)**2.0, 
         (numpy.arange(N[0])-(N[0]-1)/2.)**2.0 )
   pupAp=((cds<=(N[0]/2)**2)*(cds>(N[1]/2)**2))*1
#   pupAp=numpy.ones([N[0]]*2) # filled in pupil
   #   # \/ spider
   #pupAp[ pupAp.shape[0]//2-1:pupAp.shape[0]//2+2 ]=0
   #pupAp[ :, pupAp.shape[1]//2-1:pupAp.shape[1]//2+2 ]=0

   #   # \/ STFC logo
   #stfc=pyp.imread(
   #      "/cfai/elite/exchange/STFC_logos/STFC/STFC.png").sum(axis=-1)
   #stfc=stfc[:,:stfc.shape[0]]<2
   #pupAp=stfc+0.0 ; N=[pupAp.shape[0]]
      # /\ N here is not N as in the functions! Different meaning so don't
      # overinterpret.

   print("gInst...",end="") ; sys.stdout.flush()
   ts=time.time()
   gInst=abbot.gradientOperator.gradientOperatorType1(
      pupilMask=pupAp, sparse=doSparse )
#?? # \/ CANARY sub-aperture mask
#??    print("**NB** Using CANARY sub-aperture mask")
#??    N=[8,8]
#??    thisAp=numpy.array([[0,0,1,1,1,0,0], [0,1,1,1,1,1,0], [1,1,1,1,1,1,1],
#??          [1,1,1,0,1,1,1], [1,1,1,1,1,1,1], [0,1,1,1,1,1,0], [0,0,1,1,1,0,0]])
#??    gInst=abbot.gradientOperator.gradientOperatorType1( thisAp,sparse=doSparse )
   pupAp=(gInst.illuminatedCorners>0)*1 # force as integers
   print("({0:3.1f}s)...".format(time.time()-ts),end="") ; sys.stdout.flush()
   gO=gInst.returnOp() # gradient operator matrix
   ts=time.time()
   print("({0:3.1f}s, done)".format(time.time()-ts)) ; sys.stdout.flush()

   if noiseReduction:
      ts=time.time()
      print("noise reduction: defn...",end="") ; sys.stdout.flush()
      loopsDef=gradNoise_Fried.loopsDefine( gInst, contLoopBoundaries ) 
      loopIntM=gradNoise_Fried.loopsIntegrationMatrix(
            loopsDef, gInst, doSparse) 
      print("({0:3.1f}s)...inversion...".format(time.time()-ts),end="")
      ts=time.time()
      sys.stdout.flush()
      noiseExtM,noiseReductionM=\
            gradNoise_Fried.loopsNoiseMatrices( loopIntM, gInst )
#      # \/ sparsify
#      if nrSparsifyFrac!=0:
#         maxInM=abs(noiseExtM).max()
#         noiseReductionM=numpy.identity(gInst.numberSubaps*2)-\
#               numpy.where( abs(noiseExtM)>(maxInM*nrSparsifyFrac), noiseExtM, 0 )
#         print("sparsified",end="") 
#      else:
#         print("(non-sparse)",end="")

      print("({0:3.1f}s, done)".format(time.time()-ts),end="")
      sys.stdout.flush()
   
   rdm=numpy.random.normal(size=gInst.numberPhases) # what to reconstruct
   #rdm=numpy.add.outer(numpy.arange(N[0]),(N[0]-1)*0+(0)*numpy.arange(N[0])).ravel().take(
   #   gInst.illuminatedCornersIdx ) # 45deg slope
#>    print("phscov.",end="") ; sys.stdout.flush()
#>    directPCOne=abbot.phaseCovariance.covarianceDirectRegular( N[0], r0, L0 )
#>    print(".",end="") ; sys.stdout.flush()
#>    directPC=abbot.phaseCovariance.covarianceMatrixFillInRegular( directPCOne ) 
#>    print(".",end="") ; sys.stdout.flush()
#>    directcholesky=abbot.phaseCovariance.choleskyDecomp(directPC)
#>    print(".",end="") ; sys.stdout.flush()
#>    rdm=numpy.random.normal(size=N[0]**2) # what to reconstruct
#>    directTestPhase=numpy.dot(directcholesky, rdm)
#>    print(".",end="") ; sys.stdout.flush()
#>    del directPC,directcholesky,rdm
#>    print(".",end="") ; sys.stdout.flush()
#>    rdm=directTestPhase.ravel().take(gInst.illuminatedCornersIdx) # redo vector
#>    print("(done)") ; sys.stdout.flush()

   # ----> begins ---->

   print("chainsDefine...",end="") ; sys.stdout.flush()
   ts=time.time()
   chainsNumber,chainsDef,chainsDefChStrts=chainsDefine(\
         gInst, boundary=[chainPeriodicBound]*2, maxLen=chainMaxLength,
         shortChains=doShortChains )
   print("({0:3.1f}s, done)".format(time.time()-ts)) ; sys.stdout.flush()

   if not doSparse:
      gradV=numpy.dot(gO, rdm)
   else:
      gradV=gO.dot(rdm)
   print("...add noise",end="") ; sys.stdout.flush()
   if gradNoiseVarScal>0:
      gradV+=numpy.random.normal(
            0,gradV.var()**0.5*gradNoiseVarScal,size=len(gradV))
   if noiseReduction:
      print("...denoise (!) grads...",end="") ; sys.stdout.flush()
      gradV=noiseReductionM.dot( gradV )
   print("...",end="") ; sys.stdout.flush()
   print("rot grads...",end="") ; sys.stdout.flush()
   rgradV=rotateVectors(gradV).ravel()
   print("(done)") ; sys.stdout.flush()
#   raise RuntimeError("Stop just after gradV is setup")

   print("mapping...",end="") ; sys.stdout.flush()
   ts=time.time()
   chainsMap=chainsMapping(chainsDef,gInst)
   print("(done, {0:3.1f})".format(time.time()-ts)) ; sys.stdout.flush()
   print("integrate...",end="") ; sys.stdout.flush()
   ts=time.time()
   chains=chainsIntegrate(chainsDef,rgradV,chainsMap)
   print("(done, {0:3.1f})".format(time.time()-ts)) ; sys.stdout.flush()
   print("vectorize...",end="") ; sys.stdout.flush()
   chainsV,chainsVOffsets=chainsVectorize(chains)
   print("(done)") ; sys.stdout.flush()

   print("overlaps...",end="") ; sys.stdout.flush()
   ts=time.time()
   chainsOvlps=chainsOverlaps(chainsDef,chainOvlpType)
   print("({0:3.1f}s, done)".format(time.time()-ts)) ; sys.stdout.flush()

   print("matrices...",end="") ; sys.stdout.flush()
   A,B=chainsDefMatrices(chainsOvlps, chainsDef, chainsDefChStrts,
         sparse=doSparse)
   print("(done)") ; sys.stdout.flush()

   if not doSparse:
      print("inversion.",end="") ; sys.stdout.flush()
#      zeroRows=(abs(A).sum(axis=1)==0).nonzero()[0]
#      if len(zeroRows)>0:
#         nonzeroRows=(abs(A).sum(axis=1)!=0).nonzero()[0]
#         A=A.take(nonzeroRows,axis=0)
#         B=B.take(nonzeroRows,axis=0)
#         print("** REMOVING ROWS FROM A,B:"+str(zeroRows))
      #offsetEstM=numpy.dot( numpy.linalg.pinv( A, rcond=0.1 ), -B )
#      pupApStartsOnly=numpy.zeros(pupAp.shape)
#      pupApStartsOnly.ravel()[chainsDefChStrts[0][1]+chainsDefChStrts[1][1]]=1
      print(".",end="") ; sys.stdout.flush()
#      chOScovM=abbot.phaseCovariance.covarianceMatrixFillInMasked(
#         directPCOne, pupApStartsOnly )
      print(".",end="") ; sys.stdout.flush()
#<<<>>>      pdb.set_trace() # <<<>>>
#      invchOScovM=numpy.linalg.pinv(chOScovM)
      invchOScovM=numpy.identity(A.shape[1])
      offsetEstM=numpy.dot( numpy.dot(
         numpy.linalg.inv( numpy.dot( A.T,A )+invchOScovM*1e-2 ), A.T ), -B )
      if fractionalZeroingoeM>0:
         maxoeM=max(abs(offsetEstM).ravel())
         offsetEstM*=abs(offsetEstM)>=(maxoeM*fractionalZeroingoeM)
         print("oeM modification, fraction filled={0:5.3f}".format(
           (offsetEstM!=0).sum()*(offsetEstM.shape[0]*offsetEstM.shape[1])**-1.)
            ,end="")
      print(".",end="") ; sys.stdout.flush()
   else:
      import scipy.sparse
      ATA=A.transpose().tocsr().dot(A) ; ATB=A.transpose().tocsr().dot(B)
      ATA=ATA+scipy.sparse.csr_matrix(
            (1e-2*numpy.ones(chainsDefChStrts[2]),
             range(chainsDefChStrts[2]), range(chainsDefChStrts[2]+1) ) )
      offsetEstM=None

   print("(done)") ; sys.stdout.flush()


   if not doSparse:
      print("offset discovery (MVM)...",end="") ; sys.stdout.flush()
      offsetEstV=numpy.dot( offsetEstM, chainsV )
   else:
      print("offset discovery (sparse CG)...",end="") ; sys.stdout.flush()
      import scipy.sparse.linalg
      _A=ATA
      _b=ATB.dot( chainsV )
      thiscounter=counter()
      linalgRes=scipy.sparse.linalg.cg(
            _A, _b, callback=thiscounter.cb,tol=1e-3 )
      print("(number of loops={0:3d})".format(thiscounter.n),end="")
      if linalgRes[1]!=0:
         raise ValueError("CG failed, returned {0:d} rather than 0."+
               "Stopping.".format(linalgRes[0]))
      else:
         offsetEstV=-linalgRes[0] # this must be negated
   print("(done)") ; sys.stdout.flush()
   comp=numpy.zeros(2*N[0]**2, numpy.float64)
   updates=numpy.zeros(2*N[0]**2, numpy.float64)

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
   rdmViz=ma.masked_array(numpy.zeros(comp[0].shape),pupAp==0)
   rdmViz.ravel()[ gInst.illuminatedCornersIdx ]=rdm
   rdmViz-=rdmViz.mean()
   for dirn in (0,1): compViz[dirn]-=compViz[dirn].mean()

   print("rdm.var={0:5.3f}".format(rdmViz.var()))
   print("comp.var={0:5.3f}".format(compBothViz.var()))
   print("(rdm-comp).var={0:7.5f}".format((rdmViz-compBothViz).var()))


#"""    # try removing highest frequencies
#"""    freqLimits= gInst.n_[0]/2-3, gInst.n_[0]/2
#""" 
#"""    cds=(gInst.illuminatedCornersIdx%gInst.n_[0],
#"""         gInst.illuminatedCornersIdx//gInst.n_[0])
#"""    eiBasis=lambda x,y :\
#"""          numpy.exp(1j*2*numpy.pi*(x*gInst.n_[0]**-1.0*cds[0]
#"""             +y*gInst.n_[1]**-1.0*cds[1]))
#"""    modesToRemove=[]
#"""    for i in numpy.arange( freqLimits[0], freqLimits[1]+1, 1 ):
#"""       for j in numpy.arange( freqLimits[0], freqLimits[1]+1, 1 ):
#"""          modesToRemove.append( eiBasis(i,j).real )
#"""          modesToRemove.append( eiBasis(i,j).imag )
#"""    modesToRemove=numpy.array(modesToRemove)[:-1] # last should be zero
#"""    modeNormV=numpy.dot( modesToRemove, modesToRemove.T ).diagonal()
#"""    modesToRemove*=numpy.where(
#"""          abs(modeNormV)<1, 0, (modeNormV+1e-10)**-0.5).reshape([-1,1])
#"""    modeV=numpy.dot( modesToRemove,
#"""          compBothViz.ravel()[ gInst.illuminatedCornersIdx ].data )
#"""    compBoth_HFremoved_Viz=compBothViz.copy()
#"""    compBoth_HFremoved_Viz.ravel()[gInst.illuminatedCornersIdx]-=(
#"""          modesToRemove*modeV.reshape([-1,1])).sum(axis=0)
#"""    print("(rdm-comp_hfremoved).var={0:5.3f}".format(
#"""          (rdmViz-compBoth_HFremoved_Viz).var()))
#"""    pyp.figure()
#"""    pyp.subplot(3,2,1)
#"""    pyp.imshow( compBothViz, vmin=(rdmViz.ravel().min()),
#"""          vmax=(rdmViz.ravel().max())) ; pyp.colorbar()
#"""    pyp.title("Before HF removal")
#"""    pyp.subplot(3,2,2+1)
#"""    pyp.imshow( rdmViz-compBothViz ) ; pyp.colorbar()
#"""    pyp.title("residual")
#"""    pyp.subplot(3,2,4+1)
#"""    pyp.imshow( abs(numpy.fft.fft2(rdmViz-compBothViz)) )
#"""    pyp.title("PS(orig-hwr,no nf)")
#"""    pyp.colorbar()
#"""    #
#"""    pyp.subplot(3,2,2)
#"""    pyp.imshow( compBoth_HFremoved_Viz, vmin=(rdmViz.ravel().min()),
#"""          vmax=(rdmViz.ravel().max()) ) ; pyp.colorbar() 
#"""    pyp.subplot(3,2,2+2)
#"""    pyp.imshow( rdmViz-compBoth_HFremoved_Viz ) ; pyp.colorbar()
#"""    pyp.title("residual")
#"""    pyp.title("After HF removal")
#"""    pyp.subplot(3,2,4+2)
#"""    pyp.imshow( abs(numpy.fft.fft2(rdmViz-compBoth_HFremoved_Viz)) )
#"""    pyp.title("PS(orig-hwr,no nf)")
#"""    pyp.colorbar()

   if noDirectInv:
      raise RunTimeError("Aborting here, not doing inversion")
   # try mmse inversion
   print("mmse start...",end="") ; sys.stdout.flush()
   lO=abbot.gradientOperator.laplacianOperatorType1(
         pupilMask=pupAp*1,sparse=doSparse )
   lM=lO.returnOp()
   print("(done)") ; sys.stdout.flush()

   if not doSparse:
      print(" inversion...",end="") ; sys.stdout.flush()
      if dopinv:
         invO=numpy.linalg.pinv(gO,1e-6)
      else:
         invO=numpy.dot(
               numpy.linalg.inv(numpy.dot(gO.T,gO)
              +numpy.dot(lM.T,lM)*laplacianSmoother), gO.T)
      print("...",end="") ; sys.stdout.flush()
      invSol=numpy.dot(invO,gradV)
      print("(done)") ; sys.stdout.flush()
   else:
      print(" sparse CG...",end="") ; sys.stdout.flush()
      _A=gO.transpose().tocsr().dot(gO)
      _A=_A+laplacianSmoother*lM.tocsr().dot(lM)
      _b=gO.transpose().tocsr().dot( gradV )
      #linalgRes=scipy.sparse.linalg.cg( _A, _b )
      thiscounter=counter()
      linalgRes=scipy.sparse.linalg.cg(
            _A, _b, callback=thiscounter.cb )
      print("(number of loops={0:3d})".format(thiscounter.n),end="")
      if linalgRes[1]!=0:
         raise ValueError("CG failed, returned {0:d} rather than 0."+
               "Stopping.".format(linalgRes[0]))
      else:
         invSol=linalgRes[0]
      print("(done)") ; sys.stdout.flush()
   
   invViz=rdmViz.copy() ; invViz.ravel()[gInst.illuminatedCornersIdx]=invSol
   print("(rdm-mmse).var={0:7.5f}".format((rdmViz-invViz).var()))

   if len(raw_input("plot? (blank=don't)"))>0:
      pyp.spectral()
      blank=numpy.zeros([2]+gInst.n_)
      for i in chainsDef[0]: blank[0].ravel()[ i[0] ]=i[-1]%4
      for i in chainsDef[1]: blank[1].ravel()[ i[0] ]=i[-1]%4

      for c in (0,1):
         pyp.subplot(3,4,c+1)
         pyp.title("def:chains{0:1d}".format(c+1))
         pyp.imshow( ma.masked_array(blank[c],pupAp==0),
               origin='bottom',vmin=0 )
      blank2=numpy.zeros([2]+gInst.n_)-10
      for dirn in (0,1):
         for i in range(len(chainsDef[dirn])):
            blank2[dirn].ravel()[ chainsDef[dirn][i][0] ]=chains[dirn][i]
            pass

      for c in (0,1):
         pyp.subplot(3,4,c+1+2)
         pyp.title("chains{0:1d}".format(c+1))
         pyp.imshow( ma.masked_array(blank2[c],pupAp==0),
               origin='bottom' )

      pyp.subplot(3,4,5)
      pyp.imshow( rdmViz, origin='bottom',
            vmin=rdmViz.ravel().min(), vmax=rdmViz.ravel().max())
      pyp.xlabel("orig")
      pyp.colorbar()
      pyp.subplot(3,4,6)
      pyp.imshow( compViz[0], origin='bottom',
            vmin=rdmViz.ravel().min(), vmax=rdmViz.ravel().max())
      pyp.xlabel("chains1")
      pyp.colorbar()
      pyp.subplot(3,4,7)
      pyp.imshow( compViz[1], origin='bottom',
            vmin=rdmViz.ravel().min(), vmax=rdmViz.ravel().max())
      pyp.xlabel("chains2")
      pyp.subplot(3,4,8)
      pyp.imshow( compBothViz, origin='bottom',
            vmin=rdmViz.ravel().min(), vmax=rdmViz.ravel().max())
      pyp.xlabel("chains1&2")
      pyp.colorbar()

      pyp.subplot(3,4,9)
      pyp.imshow( ((rdmViz-compBothViz)),
            origin='bottom',
            vmin=(rdmViz.ravel().min()),
            vmax=(rdmViz.ravel().max()))
      pyp.xlabel("orig-chains1&2")
      pyp.colorbar()
      pyp.gcf().get_axes()[-1].set_ylabel("nb actual scale")

      pyp.subplot(3,4,10)
      pyp.imshow( numpy.log10(abs(rdmViz-compViz[0])), origin='bottom',
            vmin=-2+numpy.log10(rdmViz.ravel().max()),
            vmax=numpy.log10(rdmViz.ravel().max()))
      pyp.xlabel("orig-chains1")
      pyp.subplot(3,4,11)
      pyp.imshow( numpy.log10(abs(rdmViz-compViz[1])), origin='bottom',
            vmin=-2+numpy.log10(rdmViz.ravel().max()),
            vmax=numpy.log10(rdmViz.ravel().max()))
      pyp.xlabel("orig-chains2")
      pyp.subplot(3,4,12)
      pyp.imshow( numpy.log10(abs(rdmViz-compBothViz)),
            origin='bottom',
            vmin=-2+numpy.log10(rdmViz.ravel().max()),
            vmax=numpy.log10(rdmViz.ravel().max()))
      pyp.xlabel("orig-chains1&2")
      pyp.colorbar()
      pyp.gcf().get_axes()[-1].set_ylabel("nb log10 scale")

      pyp.figure()
      pyp.title("Chain starts, blue circles & red dots")
      pyp.imshow( rdmViz.mask, vmin=-1, vmax=0.5, origin='bottom', cmap='gray' )
      pyp.plot( [ x[0][0]%N[0] for x in chainsDef[1] ],
                  [ x[0][0]//N[0] for x in chainsDef[1] ], 'o',
                  markerfacecolor='none', markeredgecolor='b',
                  markeredgewidth=1 )
      pyp.plot( [ x[0][0]%N[0] for x in chainsDef[0] ],
                  [ x[0][0]//N[0] for x in chainsDef[0] ], 'r.' )
      pyp.axis([-0.5,N[0]-0.5]*2)
      pyp.figure()
      pyp.subplot(3,2,1)
      pyp.imshow( compBothViz, vmin=(rdmViz.ravel().min()),
            vmax=(rdmViz.ravel().max())) ; pyp.colorbar()
      pyp.title("hwr")
      pyp.subplot(3,2,2+1)
      pyp.imshow( rdmViz-compBothViz ) ; pyp.colorbar()
      pyp.title("residual")
      pyp.subplot(3,2,4+1)
      ps=abs(numpy.fft.fft2(rdmViz-compBothViz))
      pyp.imshow( ps, vmax=ps.max(),vmin=0 )
      pyp.colorbar()
      pyp.title("PS(orig-hwr)")
      #
      pyp.subplot(3,2,2)
      pyp.imshow( invViz, vmin=(rdmViz.ravel().min()),
            vmax=(rdmViz.ravel().max()) ) ; pyp.colorbar() 
      pyp.title("mmse")
      pyp.subplot(3,2,2+2)
      pyp.imshow( rdmViz-invViz ) ; pyp.colorbar()
      pyp.title("residual")
      pyp.subplot(3,2,4+2)
      pyp.imshow( abs(numpy.fft.fft2(rdmViz-invViz)) , vmax=ps.max(),vmin=0 )
      pyp.colorbar()
      pyp.title("PS(orig-mmse)")

      pyp.waitforbuttonpress()
