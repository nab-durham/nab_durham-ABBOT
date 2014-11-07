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

def smmtnsDefine( gInst, maxLen=None, boundary=None, shortSmmtns=False ):
   smmtns=[]
      # \/ define the summations
      # Let the 1st method be to zip through the subaperture index into the
      # pupil mask and find which sub-aperture corners are in the
      # mask2allcorners and that this element is the bottom-left (thus there
      # should also be four partners).
      # Let the 2nd method be similar but mask2allcorners+n_ (top left).
   N=gInst.n_ # alias
   blank=numpy.zeros([2]+N)
   smmtnsNumber=0
   smmtnsStrtNum=0
   smmtnsDef=[],[]
   smmtnsStarts=[ ([],[],[]), ([],[],[]), 0]
   insubap=False
   for i in range(N[1]+N[0]-1):
      # top left, upwards right
      y=int(numpy.where( N[0]-i-1<0, 0, N[0]-i-1 )) # starting y, then x
      x=int(numpy.where( i-N[0]+1<0, 0, i-N[0]+1 ))
      end=int(numpy.where( N[0]-y<N[1]-x, N[0]-y, N[1]-x ))
      if insubap:
         # terminate summation
         smmtnsDef[0].append( newSummation )
      insubap=False
      for j in range(end):
         thisIdx=y*N[1]+x+j*(N[0]+1)
#         startNewSmmtn=True
         if insubap:
            forceTerminate=False
            if type(None)!=type(maxLen):  # terminate if too long
               if maxLen==newSummation[1]:
                  forceTerminate=True
            if type(None)!=type(boundary) and type(None)!=type(boundary[0]):
               # terminate if reached a boundary
#                  # \/ horizontal, vertical boundaries, offset
#               if ((thisIdx%N[0])%boundary[0]==int(boundary[0]/2-1)) or\
#                     ((thisIdx//N[0])%boundary[0]==int(boundary[0]/2-1)):
                  # \/ horizontal, vertical boundaries, common
               if (((thisIdx%N[0])%boundary[0]==0) or\
                     ((thisIdx//N[0])%boundary[1]==0))\
                  and\
                  ( ((thisIdx//N[0])<(N[0]-1)) and
                    ((thisIdx%N[0])<(N[1]-1)) ):
                  # \/ diagonal boundaries
#               if ((thisIdx%N[0])%boundary[0]==((N[0]-thisIdx//N[0])%boundary[0])):
                  forceTerminate=True
            if forceTerminate:
               # check that forcing termination won't result in a grid
               # point that cannot then be included
###(xxx)               if shortSmmtns:
###(xxx)                  if thisIdx in gInst.illuminatedCornersIdx and not (
###(xxx)                        thisIdx+N[1] in gInst.illuminatedCornersIdx and
###(xxx)                        thisIdx+N[1]+1 in gInst.illuminatedCornersIdx and
###(xxx)                        thisIdx+1 in gInst.illuminatedCornersIdx):
###(xxx)   #                  print("<<<",thisIdx,thisIdx%N[0],thisIdx//N[0]) 
###(xxx)                     forceTerminate=False
###(xxx)               else:
###(xxx)                  nextThisIdx=thisIdx+(N[0]+1)
###(xxx)                  if nextThisIdx in gInst.illuminatedCornersIdx and not (
###(xxx)                        nextThisIdx+N[1] in gInst.illuminatedCornersIdx and
###(xxx)                        nextThisIdx+N[1]+1 in gInst.illuminatedCornersIdx and
###(xxx)                        nextThisIdx+1 in gInst.illuminatedCornersIdx):
###(xxx)   #                  print("<<<",thisIdx,thisIdx%N[0],thisIdx//N[0]) 
###(xxx)                     forceTerminate=False
               # check that forcing termination won't result in a zero-length
               # grid point
               if newSummation[1]==1:
                  forceTerminate=False
            if ( thisIdx in gInst.illuminatedCornersIdx and
                  thisIdx+N[1] in gInst.illuminatedCornersIdx and
                  thisIdx+N[1]+1 in gInst.illuminatedCornersIdx and
                  thisIdx+1 in gInst.illuminatedCornersIdx and
                  thisIdx%N[0]<(N[1]-1) and
                  gInst.subapMask[thisIdx//N[1],thisIdx%N[0]]!=0
                  ) and not forceTerminate:
               # continue summation 
               newSummation[0].append( thisIdx ) ; newSummation[1]+=1
            elif thisIdx in gInst.illuminatedCornersIdx:
               if not (shortSmmtns and forceTerminate):
                  # must terminate summation but include this point
                  newSummation[0].append( thisIdx ) ; newSummation[1]+=1
               insubap=False
               smmtnsDef[0].append( newSummation )
            else:
               # terminate summation 
               insubap=False
               smmtnsDef[0].append( newSummation )
         if not insubap:# and startNewSmmtn:
            if ( thisIdx in gInst.illuminatedCornersIdx and
                  thisIdx+N[1] in gInst.illuminatedCornersIdx and
                  thisIdx+N[1]+1 in gInst.illuminatedCornersIdx and
                  thisIdx+1 in gInst.illuminatedCornersIdx and
                  thisIdx%N[0]<N[1]-1 and
                  gInst.subapMask[thisIdx//N[1],thisIdx%N[0]]!=0
                  ):
               # start summation 
               smmtnsNumber+=1
               insubap=True
               newSummation=[[thisIdx],1, int(smmtnsNumber)]
         
               # \/ will always be a new entry
               if thisIdx in smmtnsStarts[0][1]:
                  # so this should never happen
                  raise ValueError("Gone bonkers")
               smmtnsStarts[0][0].append(smmtnsNumber)
               smmtnsStarts[0][1].append(thisIdx)
               smmtnsStarts[0][2].append(smmtnsStrtNum)
               smmtnsStrtNum+=1

   for i in range(N[1]+N[0]-1):
      # top right, downwards left
      y=int(numpy.where( N[0]-i-1<0, 0, N[0]-i-1 )) # starting y, then x
      x=int(numpy.where( N[0]-i-1>0, N[1]-1, N[1]-1+N[0]-i-1 ))
      end=int(numpy.where( N[0]-y<=x, N[0]-y, x+1 ))
      if insubap:
         # terminate summation 
         smmtnsDef[1].append( newSummation )
      insubap=False
      for j in range(end):
         thisIdx=y*N[1]+x+j*(N[0]-1)
#         startNewSmmtn=True
         if insubap:
            forceTerminate=False
            if type(None)!=type(maxLen):
               if maxLen==newSummation[1]:
                  forceTerminate=True
            if type(None)!=type(boundary) and type(None)!=type(boundary[1]):
               # terminate if reached a boundary
#                  # \/ horizontal/vertical boundaries, offset
#               if ((thisIdx%N[0])%boundary[1]==1) or\
#                     ((thisIdx//N[0])%boundary[1]==(boundary[1]-1)):
                  # \/ horizontal/vertical boundaries, common
# ?%^                if ((thisIdx%N[0])%boundary[0]==0) or\
# ?%^                      ((thisIdx//N[0])%boundary[1]==0):
               if (((thisIdx%N[0])%boundary[0]==0) or\
                     ((thisIdx//N[0])%boundary[1]==0))\
                  and\
                  ( ((thisIdx//N[0])<(N[0]-1)) and
                    ((thisIdx%N[0])>0) ):
                  # \/ diagonal boundaries
#               if ((thisIdx%N[0])%boundary[1]==((thisIdx//N[0])%boundary[1])):
                  forceTerminate=True
            if forceTerminate:
               # check that forcing termination won't result in a grid
               # point that cannot then be included
###(xxx)               if shortSmmtns:
###(xxx)                  if thisIdx in gInst.illuminatedCornersIdx and not (
###(xxx)                        thisIdx+N[1] in gInst.illuminatedCornersIdx and
###(xxx)                        thisIdx+N[1]-1 in gInst.illuminatedCornersIdx and
###(xxx)                        thisIdx-1 in gInst.illuminatedCornersIdx):
###(xxx)   #                  print(">>>",thisIdx,thisIdx%N[0],thisIdx//N[0]) 
###(xxx)                     forceTerminate=False
###(xxx)               else: 
###(xxx)                  nextThisIdx=thisIdx+(N[0]-1)
###(xxx)                  if nextThisIdx in gInst.illuminatedCornersIdx and not (
###(xxx)                        nextThisIdx+N[1] in gInst.illuminatedCornersIdx and
###(xxx)                        nextThisIdx+N[1]-1 in gInst.illuminatedCornersIdx and
###(xxx)                        nextThisIdx-1 in gInst.illuminatedCornersIdx):
###(xxx)   #                  print(">>>",thisIdx,thisIdx%N[0],thisIdx//N[0]) 
###(xxx)                     forceTerminate=False
               # check that forcing termination won't result in a zero-length
               # grid point
               if newSummation[1]==1:
                  forceTerminate=False
# ?%^             if ( thisIdx in gInst.illuminatedCornersIdx and
# ?%^                   thisIdx+N[1] in gInst.illuminatedCornersIdx and
# ?%^                   thisIdx+N[1]-1 in gInst.illuminatedCornersIdx and
# ?%^                   thisIdx-1 in gInst.illuminatedCornersIdx and
# ?%^                   ((thisIdx%N[0]<(N[0]-1) and
# ?%^                    gInst.subapMask[thisIdx//N[1],thisIdx%N[0]-1]!=0))
# ?%^                   ) and not forceTerminate:
            if ( thisIdx in gInst.illuminatedCornersIdx and
                  thisIdx+N[1] in gInst.illuminatedCornersIdx and
                  thisIdx+N[1]-1 in gInst.illuminatedCornersIdx and
                  thisIdx-1 in gInst.illuminatedCornersIdx and
                  ((thisIdx%N[0]>0 and
                   gInst.subapMask[thisIdx//N[1],thisIdx%N[0]-1]!=0))
                  ) and not forceTerminate:
               # continue summation 
               newSummation[0].append( thisIdx ) ; newSummation[1]+=1
            elif thisIdx in gInst.illuminatedCornersIdx:
               if not (shortSmmtns and forceTerminate):
                  # must terminate summation but include this point
                  newSummation[0].append( thisIdx ) ; newSummation[1]+=1
               insubap=False
               smmtnsDef[1].append( newSummation )
            else:
               # terminate summation 
               insubap=False
               smmtnsDef[1].append( newSummation )
         if not insubap:# and startNewSmmtn:
            if ( thisIdx in gInst.illuminatedCornersIdx and
                  thisIdx+N[1] in gInst.illuminatedCornersIdx and
                  thisIdx+N[1]-1 in gInst.illuminatedCornersIdx and
                  thisIdx-1 in gInst.illuminatedCornersIdx and
                  thisIdx%N[0]>0
                  and ((thisIdx%N[0]>(0) and
                   gInst.subapMask[thisIdx//N[1],(thisIdx%N[0])-1]!=0))
                  ):
               # start summation 
               smmtnsNumber+=1
               insubap=True
               newSummation=[[thisIdx],1,int(smmtnsNumber)]
              
               # \/ principle code to check overlaps of summations
               if thisIdx in smmtnsStarts[0][1]:
                  # have an overlap so direct there
                  smmtnsStarts[1][2].append(
                        smmtnsStarts[0][2][
                           smmtnsStarts[0][1].index(thisIdx)
                        ])
               elif thisIdx in smmtnsStarts[1][1]:
                  # should never happen
                  raise ValueError("Gone bonkers")
               else:
                  smmtnsStarts[1][2].append(smmtnsStrtNum)
                  smmtnsStrtNum+=1
                  
               # \/ and store which summation we are talking about
               smmtnsStarts[1][0].append(smmtnsNumber)
               smmtnsStarts[1][1].append(thisIdx)

   smmtnsStarts[-1]=smmtnsStrtNum
   return smmtnsNumber, smmtnsDef, smmtnsStarts

def smmtnsMapping( smmtnsDef, gInst ):
   smmtnsMap=([],[])
   for i in range(len(smmtnsDef[0])):
      tcM=[]
      for j in range(smmtnsDef[0][i][1]-1): # never use first element, defined as zero 
         tC=smmtnsDef[0][i][0][j] # summation index
         tgC=(gInst.mask2allcorners==tC).nonzero()[0]  # index into gradient vector
         if type(tgC)==None or len(tgC)==0:
            raise ValueError("Failed to index gradient")
         if len(tgC)>1:
            raise ValueError("Conflict!")
         tcM.append(int(tgC))
      smmtnsMap[0].append( tcM )
   for i in range(len(smmtnsDef[1])):
      tcM=[]
      for j in range(smmtnsDef[1][i][1]-1): # never use first element, defined as zero 
         tC=smmtnsDef[1][i][0][j] # summation index
         tgC=gInst.numberSubaps+(gInst.mask2allcorners==(tC-1)).nonzero()[0]
         if type(tgC)==None:
            raise ValueError("Failed to index gradient")
         if len(tgC)>1:
            raise ValueError("Conflict!")
         tcM.append(int(tgC))
      smmtnsMap[1].append( tcM )
   return smmtnsMap

def smmtnsIntegrate( smmtnsDef, rgradV, smmtnsMap ):
   smmtns=([],[])
   for j in (0,1):
      for i in range(len(smmtnsDef[j])):
         smmtns[j].append(numpy.array(
               [0]+list(numpy.cumsum( rgradV.take(smmtnsMap[j][i])*2**0.5 )) ))
   return smmtns

def smmtnsVectorize( smmtns ):
   # \/ produce the summation vector and indexing into it
   smmtnsV=[]
   smmtnsVOffsets=[]
   for dirn in (0,1):
      for tsmmtn in smmtns[dirn]:
         smmtnsVOffsets.append( len(smmtnsV) )
         smmtnsV+=tsmmtn.tolist()
   return numpy.array(smmtnsV), numpy.array(smmtnsVOffsets)

def smmtnsOverlaps( smmtnsDef, intermediate=1 ):
   # Solution approach is: each summation has an offset, as does the orthogonal
   # summation at the edge where they meet. The system can be pinned down by
   # defining the summation values plus their offsets to be equal, or
   # equivalently, their difference to be zero. Note that some pairs of
   # overlapping summations share offsets.  This latter fact constrains the
   # reconstruction. 
   #
   # An alternative approach which can quash high-frequencies is to also
   # overlap at summation intermediate points. This results in only one grid,
   # not the usual 2 nested grids. The argument 'intermediate' adds this grid
   # with the value being the scaling relative to the summation-offset grids.
   # Thus very small values weight the summation-offset grids solution, whereas
   # very large values weight the summation intermediate-grid solution.

   # NB There is a missing piece of information which is that for bounded
   #  but non-short summations, there can be an overlap between summations in
   #  the same direction: the end of one will touch the other. If this was
   #  actually implemented, then it would be equivalent to continuing the
   #  summation over the boundary -> noise propagation. It is unlikely to
   #  actually affect matters if implemented as the noise is homogenous, but
   #  not implementing it also eases the algorithm a bit and only removes one
   #  step.

   smmtnsMatching=[]
   overlap=0
   for i in range(len(smmtnsDef[0])): # index the summations we want to align
      for k in range(0, smmtnsDef[0][i][1], 1):
            # /\ can choose which points along a summation to try and match
#(???) Not clear what the following lines try to do but they do introduce
#(???) a significant bug
#(???)         if ((smmtnsDef[0][i][0][k]//20)>10): # NEW
#(???)#            print("X",end=":")
#(???)            continue
         for j in range(len(smmtnsDef[1])): # index the orthogonal summations 
            for l in range(0, smmtnsDef[1][j][1], 1):
#               print((smmtnsDef[0][i][0][k]//20),end=",")
#               if not intermediate and\
               if intermediate>0 and k<(smmtnsDef[0][i][1]-1) and\
                     l<(smmtnsDef[1][j][1]-1) and\
                     (smmtnsDef[0][i][0][k]==(smmtnsDef[1][j][0][l]-1) and
                      smmtnsDef[0][i][0][k+1]==(smmtnsDef[1][j][0][l+1]+1)):
         # \/ for each consecutive pair of elements in one set of summations,
         #   define an intermediate point and find the matching intermediate
         #   points from the in the other set of summations.
#                  smmtnsMatching.append( (i,j,k,l,intermediate) )
#                  smmtnsMatching.append( (i,j,k+1,l+1,intermediate) )
                  smmtnsMatching.append([(i,j,k,l,intermediate)
                                        ,(i,j,k+1,l+1,intermediate)] )
               if smmtnsDef[0][i][0][k]==smmtnsDef[1][j][0][l]:
         # \/ for each point in one set of summations, find matching in the
         #   other
                  smmtnsMatching.append( [(i,j,k,l,1)] )
   # \/ nb There is an inefficiency here in so much that if i,j are already
   #  in one overlap i.e. intermediate>0, then it should be combined rather
   #  than added as a separate row.
   return smmtnsMatching

def smmtnsDefMatrices(smmtnsOvlps, smmtnsDef, smmtnsDefChStrts, gO=None,
      sparse=False, boundary=None):
   # If the problem is Ao+Bc=0, where o is the vector of offsets for the
   # summations, which are essentially the currently undefined start value for
   # each summation, and c is the vector of summationvalues (smmtnsV here),
   # then A is the matrix which selects the appropriate offsets and B selects
   # the appropriate summations differences which ought to be zero thus the
   # equation.  -> o=A^{+}B(-c) where A^{+} is a pseudo-inverse. Or you can use
   # the least-squares approach, o=(A^{T}A+{\alpha}I)^{-1}A^{T}B(-c) and
   # standard matrix algebra works, as long as \alpha!=0 i.e. regularisation is
   # applied.

   # Only need to consider unique offsets; some summations will start at the
   # same point so that would overconstrain the matrix. 
   # Also, avoid summation overlaps that correspond to crossing boundary
   # divisions.
   avoidedRows=[] # record overlaps of summations with the same offset 
   smmtnsLen=0
   smmtnsVOffsets=[]
   for dirn in (0,1):
      for tcD in smmtnsDef[dirn]:
         smmtnsVOffsets.append(smmtnsLen)
         smmtnsLen+=tcD[1]
   if not sparse: # define matrices first
      A=numpy.zeros([ len(smmtnsOvlps), smmtnsDefChStrts[2] ], numpy.float32)
      B=numpy.zeros([ len(smmtnsOvlps), smmtnsLen ], numpy.float32 )
   else:
      Aidx={'data':[], 'col':[] }
      Bidx={'data':[], 'col':[] }
      rowIdx=[]
   for i in range(len(smmtnsOvlps)):
         # *** the method assumes that every row is defined by each element
         # *** of the overlaps 
      #
      for j in range(len(smmtnsOvlps[i])):
         tcO=smmtnsOvlps[i][j]
#      if tcO[1]==0 and tcO[0]==0: continue
         tcCN=[ smmtnsDef[dirn][tcO[dirn]][2] for dirn in (0,1) ]
         coI=[ smmtnsDefChStrts[dirn][0].index( tcCN[dirn] ) for dirn in (0,1) ]
         if smmtnsDefChStrts[0][2][coI[0]]==smmtnsDefChStrts[1][2][coI[1]]:
            avoidedRows.append([i,len(smmtnsOvlps[i])])
#            print("Avoiding row {0:d}".format(i))
            continue
         if type(boundary)==type(1) and (
		smmtnsDef[0][tcO[0]][0][tcO[2]]//boundary
                !=smmtnsDef[1][tcO[1]][0][tcO[3]]//boundary):
            avoidedRows.append([i,len(smmtnsOvlps[i])])
            print("Vertical boundary crossing avoided {0:d}".format(i))
            continue
#         if type(boundary)==type(1) and (
#		smmtnsDef[0][tcO[0]][0][tcO[2]]%boundary
#                !=smmtnsDef[1][tcO[1]][0][tcO[3]]%boundary):
#            avoidedRows.append([i,len(smmtnsOvlps[i])])
#            print("Horizontal boundary crossing avoided {0:d}".format(i))
#            continue
         if not sparse: # fill in here
            A[ i, smmtnsDefChStrts[0][2][coI[0]]]+=tcO[-1]
            A[ i, smmtnsDefChStrts[1][2][coI[1]]]+=-tcO[-1]
            #
            B[ i, smmtnsVOffsets[smmtnsDef[0][tcO[0]][-1]-1]+tcO[2] ]+=tcO[-1]
            B[ i, smmtnsVOffsets[smmtnsDef[1][tcO[1]][-1]-1]+tcO[3] ]+=-tcO[-1]
         else:
            Aidx['data']+=[tcO[-1],-tcO[-1]]
            Aidx['col']+=[ smmtnsDefChStrts[j][2][coI[j]] for j in (0,1) ]
            rowIdx+=[i-len(avoidedRows)]*2
            Bidx['data']+=[tcO[-1],-tcO[-1]]
            Bidx['col']+=[ smmtnsVOffsets[smmtnsDef[j][tcO[j]][-1]-1]+tcO[2+j]
               for j in (0,1) ]

   if sparse:
      import scipy.sparse
      A=scipy.sparse.csr_matrix( (Aidx['data'],(rowIdx,Aidx['col'])),
            [ (len(smmtnsOvlps)-len(avoidedRows)), smmtnsDefChStrts[2] ], numpy.float32 )
      B=scipy.sparse.csr_matrix( (Bidx['data'],(rowIdx,Bidx['col'])),
            [ (len(smmtnsOvlps)-len(avoidedRows)), smmtnsLen ], numpy.float32)
   else:
      rowIndex=range(len(smmtnsOvlps))
      for i in avoidedRows[::-1]: # in reverse order
         rowIndex.pop(i[0])
      A=A[rowIndex] ; B=B[rowIndex]

   return A,B

# \/ below follow two helper functions, which make it a bit easier to just
# do hwr, although keep in mind that they may not be right for your use.

def prepHWR(gO,maxLen=None,boundary=None,overlapType=1,sparse=False,
      matrices=False,reglnVal=1e-2,laplacianRegln=0):
   smmtnsNumber,smmtnsDef,smmtnsDefChStrts=\
         smmtnsDefine( gO, maxLen=maxLen, boundary=boundary )
   smmtnsOvlps=smmtnsOverlaps(smmtnsDef,overlapType)
   smmtnsMap=smmtnsMapping(smmtnsDef,gO)
   A,B=smmtnsDefMatrices(smmtnsOvlps, smmtnsDef, smmtnsDefChStrts, gO, sparse)
#(obs):#   geometry=numpy.zeros(gO.n_,numpy.float32)
#(obs):#   for tcDCS in smmtnsDefChStrts[:-1]:
#(obs):#       for x in range(len(tcDCS[1])):
#(obs):#         geometry.ravel()[tcDCS[1][x]]=1   
#(obs):#   mappingM=numpy.zeros([geometry.sum()]*2, numpy.float32)
#(obs):#   for tcDCS in smmtnsDefChStrts[:-1]:
#(obs):#       for x in range(len(tcDCS[1])):
#(obs):#         mappingM[ geometry.ravel().nonzero()[0].searchsorted(tcDCS[1][x]),
#(obs):#                  tcDCS[2][x] ]=1
#(obs):#   lO=abbot.gradientOperator.laplacianOperatorPupil(geometry)
#(obs):#   lM=numpy.dot(lO.op,mappingM)
#(obs):#      # /\ the map converts from offset position to physical position
#(obs):#      #  increasing in the 1D sense, within the array.
#(obs):#   lTlM=numpy.dot( lM.T,lM )
   if laplacianRegln:
      if sparse: raise RuntimeError("Not implemented")
      # Calculate the laplacian operator
      smmntSVPosnsA=numpy.array( list(
            set(smmtnsDefChStrts[0][1]+smmtnsDefChStrts[1][1])
         ) )
      smmntSVPosnsMask=numpy.zeros(gO.n_)
      smmntSVPosnsMask.ravel()[smmntSVPosnsA]=1
      #
      lO=abbot.gradientOperator.laplacianOperatorType1(
            pupilMask=smmntSVPosnsMask)
      lM=lO.returnOp()
      #
      rowsToKeep=[]
      for i,j in enumerate(smmntSVPosnsMask.ravel().nonzero()[0]):
         if j in lO.illuminatedCornersIdx:
            rowsToKeep.append((i,(lO.illuminatedCornersIdx==j).nonzero()[0]))
      #
      rowsToKeep=numpy.array(rowsToKeep)
      lReducedM=numpy.zeros([len(smmntSVPosnsA)]*2)
      lReducedM.ravel()[ 
            ((rowsToKeep[:,0]*lReducedM.shape[0]).reshape([-1,1])
            +rowsToKeep[:,0]).ravel()
         ]=lM.ravel()[
            ((rowsToKeep[:,1]*lM.shape[0]).reshape([-1,1])
            +rowsToKeep[:,1]).ravel() ]
      regularizationM=numpy.dot( lReducedM.T, lReducedM)*reglnVal
   else:
      if sparse:
         from scipy.sparse import identity as spidentity
         regularizationM=spidentity(smmtnsDefChStrts[2],'d','csr')*reglnVal
      else:
         regularizationM=numpy.identity(A.shape[1])*reglnVal

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
      if matrices:
         offsetEstM=(A,B)
      else:
         offsetEstM=numpy.dot( numpy.dot(
               numpy.linalg.inv( numpy.dot( A.T,A )+regularizationM ), A.T ),
               -B )
   else:
      if matrices:
         offsetEstM=(A,B)
      else:
         ATA=A.transpose().tocsr().dot(A) ; ATB=A.transpose().tocsr().dot(-B)
         ATA=ATA+regularizationM
   # >      ATA=ATA+scipy.sparse.csr_matrix(
   # >            (1e-2*numpy.ones(len(smmtnsOvlps)),
   # >             range(len(smmtnsOvlps)), range(len(smmtnsOvlps)+1) ) )
         offsetEstM=( ATA, ATB ) # return as a tuple for CG algorithm to use
   #
   return(smmtnsDef,smmtnsDefChStrts,smmtnsMap,offsetEstM)

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

def doHWROnePoke(gradsV,smmtnsDef,gO,offsetEstM,smmtnsDefChStrts,smmtnsMap,
      thisAct,doWaffleReduction=1,doPistonReduction=1):
#   comp,numbers,ts=doHWRIntegration(
   comp,numbers=doHWRIntegration(
         gradsV, smmtnsDef, gO, offsetEstM, smmtnsDefChStrts,smmtnsMap )
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
def doHWRIntegration(gradsV,smmtnsDef,gO,offsetEstM,smmtnsDefChStrts,
      smmtnsMap,sparse=False):
   rgradV=rotateVectors(gradsV).ravel()
   smmtns=smmtnsIntegrate(smmtnsDef,rgradV,smmtnsMap)
   smmtnsV,smmtnsVOffsets=smmtnsVectorize(smmtns)
#*   print("** len[smmtnsV,smmtnsVoffsets]=[{0:d},{1:d}]".format(
#*         len(smmtnsV),len(smmtnsVOffsets)))
#*   print("** len[smmtns[0],smmtns[1]]=[{0:d},{1:d}]".format(
#*         len(smmtns[0]),len(smmtns[1])))
   if not sparse:
      offsetEstV=numpy.dot( offsetEstM, smmtnsV )
   else:
      from scipy.sparse.linalg import cg as spcg
      offsetEstV=spcg( offsetEstM[0], offsetEstM[1].dot(smmtnsV),
            tol=1e-6 )
      if offsetEstV[1]==0:
         offsetEstV=offsetEstV[0]
      else:
         raise ValueError("Sparse CG did not converge")
   
   comp=numpy.zeros([2*gO.n_[0]*gO.n_[1]], numpy.float64)
   updates=numpy.zeros([2*gO.n_[0]*gO.n_[1]], numpy.float64)

   # do one way...
   for x in range(len(smmtns[0])):
      toeI=x
      for i in range((smmtnsDef[0][x][1])):
         comp[ smmtnsDef[0][x][0][i] ]+=(smmtns[0][x][i]+offsetEstV[toeI])
         updates[ smmtnsDef[0][x][0][i] ]+=1
   # ...then another
   for x in range(len(smmtns[1])):
      toeI=smmtnsDefChStrts[1][2][x]
      for i in range((smmtnsDef[1][x][1])):
         comp[ comp.shape[0]/2+smmtnsDef[1][x][0][i] ]+=\
               (smmtns[1][x][i]+offsetEstV[toeI])
         updates[ updates.shape[0]/2+smmtnsDef[1][x][0][i] ]+=1
   return ( comp.reshape([2]+list(gO.n_)),
            updates.reshape([2]+list(gO.n_)) 
          ),(smmtnsV,smmtnsVOffsets)
###(disabled because of unknown bug)    smmtnsV=doHWRSmmtnsAddOffsets(
###(disabled because of unknown bug)          smmtns,smmtnsV,smmtnsVOffsets,smmtnsDef,smmtnsDefChStrts,offsetEstV)
###(disabled because of unknown bug)     
###(disabled because of unknown bug)    return doHWRSmmtnsFormatToConventional(
###(disabled because of unknown bug)          gO,smmtns,smmtnsDef,smmtnsV,smmtnsVOffsets),(smmtnsV,smmtnsVOffsets)

###(disabled because of unknown bug) def doHWRSmmtnsAddOffsets(smmtns,smmtnsV,smmtnsVOffsets,smmtnsDef,
###(disabled because of unknown bug)       smmtnsDefChStrts,offsetEstV):
###(disabled because of unknown bug)    #      
###(disabled because of unknown bug)    # do one way...
###(disabled because of unknown bug)    for x in range(len(smmtns[0])):
###(disabled because of unknown bug)       toeI=x
###(disabled because of unknown bug)       smmtnsV[smmtnsVOffsets[x]:smmtnsVOffsets[x]+smmtnsDef[0][x][1]]+=\
###(disabled because of unknown bug)             offsetEstV[toeI]
###(disabled because of unknown bug)    # ...then the other
###(disabled because of unknown bug)    for x in range(len(smmtns[1])):
###(disabled because of unknown bug)       toeI=smmtnsDefChStrts[1][2][x]
###(disabled because of unknown bug)       smmtnsV[smmtnsVOffsets[x+len(smmtns[0])]:
###(disabled because of unknown bug)               smmtnsVOffsets[x+len(smmtns[0])]+smmtnsDef[1][x][1]]+=\
###(disabled because of unknown bug)             offsetEstV[toeI]
###(disabled because of unknown bug)    return smmtnsV

###(disabled because of unknown bug) def doHWRSmmtnsFormatToConventional(gO,smmtns,smmtnsDef,smmtnsV,smmtnsVOffsets):
###(disabled because of unknown bug)    '''Create a 2D version of the summations, separate for each chain.'''
###(disabled because of unknown bug)    comp=numpy.zeros([2,gO.n_[0]*gO.n_[1]], numpy.float64)
###(disabled because of unknown bug)    numbers=numpy.zeros([2,gO.n_[0]*gO.n_[1]], numpy.float64)
###(disabled because of unknown bug)    for x in range(len(smmtns[0])):
###(disabled because of unknown bug)       comp[0][ smmtnsDef[0][x][0] ]=\
###(disabled because of unknown bug)          smmtnsV[smmtnsVOffsets[x]:smmtnsVOffsets[x]+smmtnsDef[0][x][1]]
###(disabled because of unknown bug)       numbers[0, smmtnsDef[0][x][0] ]+=1
###(disabled because of unknown bug)    for x in range(len(smmtns[1])):
###(disabled because of unknown bug)       comp[1][ smmtnsDef[1][x][0] ]=\
###(disabled because of unknown bug)          smmtnsV[smmtnsVOffsets[x+len(smmtns[0])]:\
###(disabled because of unknown bug)                  smmtnsVOffsets[x+len(smmtns[0])]+smmtnsDef[1][x][1]]
###(disabled because of unknown bug)       numbers[1, smmtnsDef[1][x][0] ]+=1
###(disabled because of unknown bug)    return comp,numbers

def gradientsManagement(gradsV,gO,f=3,verbose=False):
   """Analyse the gradients and examine if any exceed the expected variance
      by the specified factor. If they do, then replace the gradient with
      the interpolation/copy of its neighbours/the neighbour depending
      on if the gradient is from the middle/edge of the sub-aperture
      array.  Split the problem into two dimensions, first one-half and then
      the other.
      gradsV: the gradients, as a vector with the first half being
         one-direction, the second half containing the other direction,
      gO: gradient operator type 1 object,
      f: std deviation limit for gradient rejection.
      verbose: print out information on what was done
   """
   ngradsOneDirn=len(gradsV)//2
   assert ngradsOneDirn*2==len(gradsV),\
         "i/p vector not a multiple of 2, wrong shape"
   for dirn in (0,1):
      gradientOffsetI=dirn*ngradsOneDirn
      tgradsV=gradsV[gradientOffsetI:gradientOffsetI+ngradsOneDirn]
      tgradsVlim=f*(tgradsV.var())**0.5
      idx=numpy.flatnonzero(abs(tgradsV)>tgradsVlim)
      if len(idx)<1: continue # nothing to be done, skip
      # Now have indices of which gradients are excessive.
      # Calculate for which row/column this excess occurs and then
      # interpolate/extrapolate.
      lineNos=numpy.unique( numpy.where(dirn,
            gO.subapMaskIdx//gO.n[0],gO.subapMaskIdx%gO.n[0]) )
      for lineNo in lineNos:
         lineIdx=numpy.flatnonzero( numpy.where(dirn,
               gO.subapMaskIdx//gO.n[0],gO.subapMaskIdx%gO.n[0])==lineNo )
         lineValidIdx=numpy.lib.setdiff1d(lineIdx,idx)
         if len(lineValidIdx)==len(lineIdx): continue # nothing to be done, skip
         lineValidCds=numpy.searchsorted(lineIdx,lineValidIdx)
            # \/ do the replacement
         if verbose:
            print( ("{2:s}={3:d}, replacing {0:3d}%/{1:d} gradients from:\n\t"
                  +str(tgradsV[lineIdx])).format(
                     int(100*(1-len(lineValidIdx)*len(lineIdx)**-1.0)),
                     len(lineIdx)-len(lineValidIdx),
                     (dirn)*"col"+(1-dirn)*"row",
                     lineNo+1
                  )) 
         tgradsV[lineIdx]=numpy.interp( range(len(lineIdx)),
               lineValidCds,tgradsV[lineValidIdx])
         if verbose:
            print(" to:\n\t"+str(tgradsV[lineIdx]))

   # NOTA BENE: in principle, the following two lines are irrelevant since
   # earlier tgradsV is a view into gradsV and (I'm not sure how) gradsV
   # is also a view into the argument: thus changes are made in-place.
   # However, the following makes it explicit.
      gradsV[gradientOffsetI:gradientOffsetI+ngradsOneDirn]=tgradsV # **
   return gradsV                                                       # **

def doHWRGeneral(gradsV,smmtnsDef,gO,offsetEstM,smmtnsDefChStrts,smmtnsMap,
      doWaffleReduction=0,doPistonReduction=1,doGradientMgmnt=0,sparse=False):
   '''gradsV: the input gradients as a vector, first one-direction and then
         the remainder,
      smmtnsDef: definitions of summations,
      gO: gradient operator type 1 object,
      offsetEstM: offset estimation matrices,
      smmtnsDefChStrts: definitions of summation overlaps,
      smmtnsMap: definitions of summations 2D<->1D geometry relationship,
      doWaffleReduction: apply subsequent waffle reduction,
      doPistonReduction: apply subsequent piston reduction,
      doGradientMgmnt: apply gradient management to eliminate atypical values,
      sparse: use sparse matrices, best for problems > 1000 gradients.
   '''
   if doGradientMgmnt: gradsV=gradientsManagement(gradsV,gO)
   (comp,numbers),(smmtnsV,smmtnsVOffsets)=\
         doHWRIntegration( gradsV, smmtnsDef, gO, offsetEstM, smmtnsDefChStrts,
               smmtnsMap, sparse )
   
   comp/=numpy.where( numbers<1,1,numbers ) # average over multiple adds
   numbers=numpy.where( numbers>0,1,0 )
   #
   hwrV=( (comp[0]+comp[1])*(numbers[0]+numbers[1]+1e-10)**-1.0
            ).ravel()[gO.illuminatedCornersIdx]
   if doWaffleReduction:
      globalWV=localWaffle(-1,gO)
      hwrV-=numpy.dot( globalWV[1], hwrV )*globalWV[1]
   if doPistonReduction:
      globalPV=localPiston(-1,gO)
      hwrV-=numpy.dot( globalPV[1], hwrV )*globalPV[1]
   #
   return comp,hwrV,(smmtnsV,smmtnsVOffsets)
# --begin--

if __name__=="__main__":

   import pdb # Python debugger
   import abbot.continuity
   import time
   import numpy.ma as ma

   def doPlots(thisHWRViz,thisTitle,invViz=None):
      import matplotlib.pyplot as pyp
      # <------->
      pyp.figure()
      pyp.spectral()
      blank=numpy.zeros([2]+gInst.n_)
      for i in smmtnsDef[0]: blank[0].ravel()[ i[0] ]=i[-1]%4
      for i in smmtnsDef[1]: blank[1].ravel()[ i[0] ]=i[-1]%4

      for c in (0,1):
         pyp.subplot(3,4,c+1)
         pyp.title("def:smmtns{0:1d}".format(c+1))
         pyp.imshow( ma.masked_array(blank[c],pupAp==0),
               origin='bottom',vmin=0 )
      blank2=numpy.zeros([2]+gInst.n_)-10
      for dirn in (0,1):
         for i in range(len(smmtnsDef[dirn])):
            blank2[dirn].ravel()[ smmtnsDef[dirn][i][0] ]=smmtns[dirn][i]
            pass

      for c in (0,1):
         pyp.subplot(3,4,c+1+2)
         pyp.title("smmtns{0:1d}".format(c+1))
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
      pyp.xlabel("smmtns1")
      pyp.colorbar()
      pyp.subplot(3,4,7)
      pyp.imshow( compViz[1], origin='bottom',
            vmin=rdmViz.ravel().min(), vmax=rdmViz.ravel().max())
      pyp.xlabel("smmtns2")
      pyp.subplot(3,4,8)
      pyp.imshow( thisHWRViz, origin='bottom',
            vmin=rdmViz.ravel().min(), vmax=rdmViz.ravel().max())
      pyp.xlabel("smmtns1&2")
      pyp.colorbar()

      pyp.subplot(3,4,9)
      pyp.imshow( ((rdmViz-thisHWRViz)),
            origin='bottom',
            vmin=(rdmViz.ravel().min()),
            vmax=(rdmViz.ravel().max()))
      pyp.xlabel("orig-smmtns1&2")
      pyp.colorbar()
      pyp.gcf().get_axes()[-1].set_ylabel("nb actual scale")

      pyp.subplot(3,4,10)
      pyp.imshow( numpy.log10(abs(rdmViz-compViz[0])), origin='bottom',
            vmin=-2+numpy.log10(rdmViz.ravel().max()),
            vmax=numpy.log10(rdmViz.ravel().max()))
      pyp.xlabel("orig-smmtns1")
      pyp.subplot(3,4,11)
      pyp.imshow( numpy.log10(abs(rdmViz-compViz[1])), origin='bottom',
            vmin=-2+numpy.log10(rdmViz.ravel().max()),
            vmax=numpy.log10(rdmViz.ravel().max()))
      pyp.xlabel("orig-smmtns2")
      pyp.subplot(3,4,12)
      pyp.imshow( numpy.log10(abs(rdmViz-thisHWRViz)),
            origin='bottom',
            vmin=-2+numpy.log10(rdmViz.ravel().max()),
            vmax=numpy.log10(rdmViz.ravel().max()))
      pyp.xlabel("orig-smmtns1&2")
      pyp.colorbar()
      pyp.gcf().get_axes()[-1].set_ylabel("nb log10 scale")

      # <------->
      pyp.figure()
      pyp.title("Summation starts, blue circles & red dots")
      pyp.imshow( rdmViz.mask, vmin=-1, vmax=0.5, origin='bottom', cmap='gray' )
      sStarts=[];sAll=[]
      for x in smmtnsDef[1]:
         sStarts.append( x[0][0] )
         sAll+=x[0]
      pyp.plot( numpy.array(sAll)%N[0],
                numpy.array(sAll)//N[0], 'k,')
      pyp.plot( numpy.array(sStarts)%N[0],
                numpy.array(sStarts)//N[0], 'o',
                  markerfacecolor='none', markeredgecolor='b',
                  markeredgewidth=1 )
      sStarts=[];sAll=[]
      for x in smmtnsDef[0]:
         sStarts.append( x[0][0] )
         sAll+=x[0]
      pyp.plot( numpy.array(sAll)%N[0],
                numpy.array(sAll)//N[0], 'k,')
      pyp.plot( numpy.array(sStarts)%N[0],
                numpy.array(sStarts)//N[0], 'r.' )
      pyp.axis([-0.5,N[0]-0.5]*2)

      # <------->
      pyp.figure()
      pyp.subplot(3,2,1)
      pyp.imshow( thisHWRViz, vmin=(rdmViz.ravel().min()),
            vmax=(rdmViz.ravel().max())) ; pyp.colorbar()
      pyp.title(thisTitle)
      pyp.subplot(3,2,2+1)
      pyp.imshow( rdmViz-thisHWRViz ) ; pyp.colorbar()
      pyp.title("residual")
      pyp.subplot(3,2,4+1)
      ps=abs(numpy.fft.fft2(rdmViz-thisHWRViz))
      pyp.imshow( ps, vmax=ps.max(),vmin=0 )
      pyp.colorbar()
      pyp.title("PS(orig-{0:s})".format(thisTitle))
      #
      if invViz!=None:
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

# \/ configuration here_______________________ 
   N=[32,-1] ; #N[1]=(N[0]*6)//39. # size
   r0=2 ; L0=4*r0#N[0]/3.0
   noDirectInv=False # if true, don't attempt MMSE
   doSparse=1
   smmtnPeriodicBound=[None,16,8,4,N[0]+1][1]# optional 
   smmtnMaxLength=None # better to use periodic-boundaries than fixed lengths
   gradNoiseVarScal=0.00000001 # multiplier of gradient noise
   smmtnOvlpType=0.10    # 0=direct x-over, 1=intermediate x-over
      # /\ (0.15 -> best with VK , 0 -> best c. random
      #     WITH NO NOISE)
   dopinv=False 
   doShortSmmtns=0#False # True means do not include summation bounday truncation overlaps
   disableNoiseReduction=1#False
   contLoopBoundaries=[ N[0]+1, N[0]/2, smmtnPeriodicBound ][0]
   laplacianSmoother=1e-6
   OEregVal=1e-6 # offset est. regularization value, 1e-2 is usually good
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
   print(" smmtnPeriodicBound={0:s}".format(
         str(numpy.where(smmtnPeriodicBound!=None,smmtnPeriodicBound,"")) ))
   print(" contLoopBoundaries={0:s}".format(
         str(numpy.where(contLoopBoundaries!=None,contLoopBoundaries,"")) ))
   print(" smmtnMaxLength={0:s}".format(
         str(numpy.where(smmtnMaxLength!=None,smmtnMaxLength,"")) ))
   print(" noiseReduction={0:d}".format(noiseReduction>0)) 
   print(" smmtnOvlpType={0:5.3f}".format(smmtnOvlpType)) 
   print(" N={0:d}".format(N[0])) 
   print(" r0={0:d}, L0={1:d}".format(int(r0),int(L0)))
   print(" doSparse={0:d}".format(doSparse>0)) 
   print(" noDirectInv={0:d}".format(noDirectInv>0)) 
   print(" dopinv={0:d}".format(dopinv>0)) 
   print(" shortSmmtns={0:d}".format(doShortSmmtns>0)) 

   if doSparse:
      class counter(object):
        n=0
        def cb(self,ip):
          self.n+=1

   cds=numpy.add.outer(
         (numpy.arange(N[0])-(N[0]-1)/2.)**2.0, 
         (numpy.arange(N[0])-(N[0]-1)/2.)**2.0 )
   pupAp=((cds<=(N[0]/2)**2)*(cds>(N[1]/2)**2))*1
   #pupAp=numpy.ones([N[0]]*2) # filled in pupil
   #   # \/ spider
   #pupAp[ pupAp.shape[0]//2-1:pupAp.shape[0]//2+2 ]=0
   #pupAp[ :, pupAp.shape[1]//2-1:pupAp.shape[1]//2+2 ]=0

   #   # \/ STFC logo
   #import matplotlib.pyplot as pyp, 
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
#   sam=numpy.load("/tmp/tmp3");gInst=abbot.gradientOperator.gradientOperatorType1(sam )
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
      loopsDef=abbot.continuity.loopsDefine( gInst, contLoopBoundaries ) 
      loopIntM=abbot.continuity.loopsIntegrationMatrix(
            loopsDef, gInst, doSparse) 
      print("({0:3.1f}s)...inversion...".format(time.time()-ts),end="")
      ts=time.time()
      sys.stdout.flush()
      noiseExtM,noiseReductionM=\
            abbot.continuity.loopsNoiseMatrices( loopIntM, gInst )
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
   
   print("N,N_={0:d},{1:d}".format(gInst.n[0],gInst.n_[0]))
   x=gInst.illuminatedCornersIdx%gInst.n_[0]
   #y=gInst.illuminatedCornersIdx//gInst.n_[0]
#   wl=8.0;ip=numpy.cos(2*numpy.pi*wl**-1.0*(x+y)*2**0.5)
   ip=x
   #ip=numpy.zeros(len(x)) ; ip[int(numpy.random.uniform(0,len(x)))]=1
   import kolmogorov
   rdm=kolmogorov.TwoScreens(N[0]*2,r0)[0][:N[0],:N[0]].ravel().take(gInst.illuminatedCornersIdx)
   
###3   rdm=ip
###3   rdm=numpy.random.normal(size=gInst.numberPhases) # what to reconstruct
###3   rdm=numpy.add.outer(
###3            numpy.arange(N[0]),
###3            numpy.arange(N[0])#,0,-1)
###3         ).T.ravel().take( gInst.illuminatedCornersIdx ) # 45deg slope
#>    import abbot.phaseCovariance
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
    
   rdmViz=ma.masked_array(numpy.zeros([N[0]]*2),pupAp==0)
   rdmViz.ravel()[ gInst.illuminatedCornersIdx ]=rdm
   rdmViz-=rdmViz.mean()

#   if not doSparse:
#      gradV=numpy.dot(gO, rdm)
#   else:
   gradV=gO.dot(rdm)
   print("...add noise",end="") ; sys.stdout.flush()
   if gradNoiseVarScal>0:
      gradV+=numpy.random.normal(
            0,gradV.var()**0.5*gradNoiseVarScal,size=len(gradV))
   if noiseReduction:
      print("...denoise (!) grads...",end="") ; sys.stdout.flush()
      gradV=noiseReductionM.dot( gradV )
   print("...",end="") ; sys.stdout.flush()

   # ----> begins ----> *** MANUAL ***

   print("smmtnsDefine...",end="") ; sys.stdout.flush()
   ts=time.time()
   smmtnsNumber,smmtnsDef,smmtnsDefChStrts=smmtnsDefine(\
         gInst, boundary=[smmtnPeriodicBound]*2, maxLen=smmtnMaxLength,
         shortSmmtns=doShortSmmtns )
   print("({0:3.1f}s, done)".format(time.time()-ts)) ; sys.stdout.flush()

   print("rot grads...",end="") ; sys.stdout.flush()
   rgradV=rotateVectors(gradV).ravel()
   print("(done)") ; sys.stdout.flush()
#   raise RuntimeError("Stop just after gradV is setup")

   print("mapping...",end="") ; sys.stdout.flush()
   ts=time.time()
   smmtnsMap=smmtnsMapping(smmtnsDef,gInst)
   print("(done, {0:3.1f})".format(time.time()-ts)) ; sys.stdout.flush()
   print("integrate...",end="") ; sys.stdout.flush()
   ts=time.time()
   smmtns=smmtnsIntegrate(smmtnsDef,rgradV,smmtnsMap)
   print("(done, {0:3.1f})".format(time.time()-ts)) ; sys.stdout.flush()
   print("vectorize...",end="") ; sys.stdout.flush()
   smmtnsV,smmtnsVOffsets=smmtnsVectorize(smmtns)
   print("(done)") ; sys.stdout.flush()

   print("overlaps...",end="") ; sys.stdout.flush()
   ts=time.time()
   smmtnsOvlps=smmtnsOverlaps(smmtnsDef,smmtnOvlpType)
   print("({0:3.1f}s, done)".format(time.time()-ts)) ; sys.stdout.flush()

   print("matrices...",end="") ; sys.stdout.flush()
   A,B=smmtnsDefMatrices(smmtnsOvlps, smmtnsDef, smmtnsDefChStrts,
         sparse=doSparse)
   print("(done)") ; sys.stdout.flush()

   if not doSparse:
      print("(A^T.A)^-1.A^T.B; inversion.",end="") ; sys.stdout.flush()
#      zeroRows=(abs(A).sum(axis=1)==0).nonzero()[0]
#      if len(zeroRows)>0:
#         nonzeroRows=(abs(A).sum(axis=1)!=0).nonzero()[0]
#         A=A.take(nonzeroRows,axis=0)
#         B=B.take(nonzeroRows,axis=0)
#         print("** REMOVING ROWS FROM A,B:"+str(zeroRows))
      #offsetEstM=numpy.dot( numpy.linalg.pinv( A, rcond=0.1 ), -B )
#      pupApStartsOnly=numpy.zeros(pupAp.shape)
#      pupApStartsOnly.ravel()[smmtnsDefChStrts[0][1]+smmtnsDefChStrts[1][1]]=1
      print(".",end="") ; sys.stdout.flush()
#      chOScovM=abbot.phaseCovariance.covarianceMatrixFillInMasked(
#         directPCOne, pupApStartsOnly )
      print(".",end="") ; sys.stdout.flush()
#<<<>>>      pdb.set_trace() # <<<>>>
#      invchOScovM=numpy.linalg.pinv(chOScovM)
      invchOScovM=numpy.identity(A.shape[1])
      offsetEstM=numpy.dot( numpy.dot(
            numpy.linalg.inv(
               numpy.dot( A.T,A )+invchOScovM*OEregVal ), A.T ), -B )
      offsetEstM=numpy.linalg.pinv(A).dot(-B)
      if fractionalZeroingoeM>0:
         raise RuntimeError("DISABLED")
###(disabled)         maxoeM=max(abs(offsetEstM).ravel())
###(disabled)         offsetEstM*=abs(offsetEstM)>=(maxoeM*fractionalZeroingoeM)
###(disabled)         print("oeM modification, fraction filled={0:5.3f}".format(
###(disabled)           (offsetEstM!=0).sum()*(offsetEstM.shape[0]*offsetEstM.shape[1])**-1.)
###(disabled)            ,end="")
      print(".",end="") ; sys.stdout.flush()
   else:
      import scipy.sparse
      ATA=A.transpose().tocsr().dot(A) ; ATB=A.transpose().tocsr().dot(B)
      ATA=ATA+scipy.sparse.csr_matrix(
            (OEregVal*numpy.ones(smmtnsDefChStrts[2]),
             range(smmtnsDefChStrts[2]), range(smmtnsDefChStrts[2]+1) ) )
      offsetEstM=None

   print("(done)") ; sys.stdout.flush()


   if not doSparse:
      print("offset discovery (MVM)...",end="") ; sys.stdout.flush()
      offsetEstV=numpy.dot( offsetEstM, smmtnsV )
   else:
      print("offset discovery (sparse CG)...",end="") ; sys.stdout.flush()
      import scipy.sparse.linalg
      _A=ATA
      _b=ATB.dot( smmtnsV )
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
   for x in range(len(smmtns[0])):
      toeI=x
      for i in range((smmtnsDef[0][x][1])):
         comp[ smmtnsDef[0][x][0][i] ]+=(smmtns[0][x][i]+offsetEstV[toeI])
         updates[ smmtnsDef[0][x][0][i] ]+=1
         pass
   # ...then another
   for x in range(len(smmtns[1])):
      toeI=smmtnsDefChStrts[1][2][x]
      for i in range((smmtnsDef[1][x][1])):
         comp[ N[0]**2+smmtnsDef[1][x][0][i] ]+=\
               (smmtns[1][x][i]+offsetEstV[toeI])
         updates[ N[0]**2+smmtnsDef[1][x][0][i] ]+=1
         pass
   print("(done)") ; sys.stdout.flush()
   comp.resize([2]+[N[0]]*2)
   updates.resize([2]+[N[0]]*2)
   
   comp/=numpy.where( updates>1,updates,1 ) # average over multiple adds
   updates=numpy.where( updates>0,1,1e-10 )
   updatesViz=ma.masked_array(updates,[pupAp==0]*2)
   compViz=[ ma.masked_array(comp[i], updatesViz[i]<1) for i in (0,1) ]
   for dirn in (0,1): compViz[dirn]-=compViz[dirn].mean()
   compBothViz=ma.masked_array(
         (updatesViz[0]+updatesViz[1])**-1.0*(comp[0]+comp[1]), pupAp==0 )
   # <---- ends ------< *** MANUAL ***

   # ----> begins ----> *** HELPER FN. ***
   smmtnsDef, smmtnsDefChStrts, smmtnsMap, offsetEstM=prepHWR(
         gInst, smmtnMaxLength,
         [smmtnPeriodicBound]*2,
         smmtnOvlpType, doSparse, reglnVal=OEregVal)
   
   compHWR, hwrV, smmtnsVs=doHWRGeneral(
         gradV,
         smmtnsDef,
         gInst,
         offsetEstM,
         smmtnsDefChStrts,
         smmtnsMap,
         doWaffleReduction=0,
         doPistonReduction=1,
         doGradientMgmnt=0,
         sparse=doSparse)

   hwrViz=ma.masked_array(numpy.zeros(comp[0].shape),pupAp==0)
   hwrViz.ravel()[gInst.illuminatedCornersIdx]=hwrV
   # <---- ends ------< *** HELPER FN. ***


   if noDirectInv:
      print("Aborting here, not doing inversion")
   else:
   # >---- begins ----> *** DIRECT INV. ***
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
   # <---- ends ------< *** DIRECT INV. ***



   print("rdm.var={0:5.3f}".format(rdmViz.var()))
   print("comp.var={0:5.3f}".format(compBothViz.var()))
   print("hwr.var={0:5.3f}".format(hwrViz.var()))
   print("(rdm-comp).var={0:7.5f}".format((rdmViz-compBothViz).var()))
   print("(rdm-hwr).var={0:7.5f}".format((rdmViz-hwrViz).var()))
   if 'invViz' in dir():
      print("(rdm-mmse).var={0:7.5f}".format((rdmViz-invViz).var()))

   if len(raw_input("plot compBothViz? (blank=don't)"))>0:
      doPlots(compBothViz,"cbHWR",None if noDirectInv else invViz)
   if len(raw_input("plot hwrViz? (blank=don't)"))>0:
      doPlots(hwrViz,"HWR",None if noDirectInv else invViz)

   # plot statistics of number of summations in the grid points 
   summationLocs=[]
   for smmtnDirs in smmtnsDef:
      for smmtns in smmtnDirs: summationLocs+=smmtns[0]

   template=numpy.zeros(gInst.n_,'i').ravel()
   for smmtnIdx in summationLocs: template[smmtnIdx]+=1

   template=template[gInst.illuminatedCornersIdx] # only choose the relevant points
   print("Summations per grid point:")
   print(" max={0:d} (expect={1:d})".format(max(template),(not doShortSmmtns)*2+2))
   print(" min={0:d} (expect=1)".format(min(template)))
