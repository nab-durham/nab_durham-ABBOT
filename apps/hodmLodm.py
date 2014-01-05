from __future__ import print_function
# What is this?
# High-order/low-order separation

import dm
import gradientOperator
import numpy
import phaseCovariance
import sys
import time


# ================
# Go to the bottom of this file to see past the definitions
# ================

def makeMask(radii,N):
   radius=numpy.add.outer(
         (numpy.arange(N)-(N-1)/2.)**2,(numpy.arange(N)-(N-1)/2.)**2 )**0.5
   return numpy.where( (radius<=(radii[0]*nSubAps/2.0)) &
                       (radius>=(radii[1]*nSubAps/2.0)), 1, 0)

def generateTestPhase(gO,r0,N):
   L0=3
   rdm=numpy.random.normal(size=gO.numberPhases)

   directPCOne=phaseCovariance.covarianceDirectRegular( N, r0, L0 )
   directPC=phaseCovariance.covarianceMatrixFillInMasked(
      directPCOne, (gO.illuminatedCorners>0) )
   directcholesky=phaseCovariance.choleskyDecomp(directPC)
   return phaseCovariance.numpy.dot(directcholesky, rdm)


def formPokeMatrices():
   print("Forming poke matrices, order=",end="")
   pokeM={}
   for dmType in dm.keys():
      print(dmType,end=" ") ; sys.stdout.flush()
      # setup array for receiving poke matrix
      pokeM[dmType]=numpy.zeros([2*gO.numberSubaps,sum(dm[dmType].usable)],
            numpy.float64)
      # create gradients for each poke
      for i,j in enumerate(dm[dmType].usableIdx):
         pokeM[dmType][:,i]=numpy.dot(
               gM, dm[dmType].poke(j).take(gO.illuminatedCornersIdx) )

   print()
   return pokeM

def concatenatePokeMatrices(dm,dmOrder):
   dmLength=sum([ sum(dm[k].usable) for k in dm.keys() ])
   stackPokeM=numpy.zeros(
         [2*gO.numberSubaps,dmLength],numpy.float64)
   columnI=0
   print("Concatenating poke matrices, order=",end="")
   for dmType in dmOrder:
      print(dmType,end=" ")
      stackPokeM[:,columnI:columnI+sum(dm[dmType].usable)]=pokeM[dmType]
      columnI+=sum(dm[dmType].usable)
  
   print()
   return stackPokeM

def makeReconstructors(reconTypes, stackedPMX, dm, dmOrder, lambd):
   global filterM
   reconMs={}
   dmLength=sum([ sum(dm[k].usable) for k in dm.keys() ])
   tikhonovRegM=numpy.identity(dmLength)
   for reconType in reconTypes:
      sTsM=stackedPMX.T.dot(stackedPMX)
      print("makeReconstructors: Making '{0:s}' reconstructor".format(
            reconType),end="")
      if reconType=="SVD_only":
         reconMs[reconType]=numpy.linalg.pinv( sTsM, lambd).dot(stackedPMX.T)
      elif reconType=="Inv+Tik":
         reconMs[reconType]=\
               numpy.linalg.inv( sTsM+tikhonovRegM*lambd ).dot(stackedPMX.T)
      elif reconType=="Diagonal-regularized":
         filterM=numpy.zeros(
               [ sum(dm['ho'].usable),
                 sum(dm['ho'].usable)+sum(dm['lo'].usable)],numpy.float64)
            # \/ depending on the order of DMs, an offset to reach the HO
            # actuator commands may be necessary.
            # The concept here is to add a diagonal term only to the
            # actuator vectors corresponding to the high-order DM so as to
            # penalize their amplitudes.
         filterM.ravel()[ (dmOrder[0]=='lo')*sum(dm['lo'].usable)+
               numpy.arange(sum(dm['ho'].usable))*(
                  sum(dm['ho'].usable)+sum(dm['lo'].usable)+1) 
                       ]=1 
         print(".",end="") ; sys.stdout.flush()
         reconMs[reconType]=numpy.linalg.inv(
                     sTsM+lambd[0]*filterM.T.dot(filterM)+tikhonovRegM*lambd[1]
                                            ).dot( stackedPMX.T )
      elif reconType=="Low-order-penalization":
         filterM=numpy.zeros(
               [ 2*( (lodmN/2*lambd[1])*(lodmN*lambd[1]) )-2,
                 sum(dm['ho'].usable)+sum(dm['lo'].usable)],numpy.float64)
            # \/ depending on the order of DMs, an offset to reach the HO
            # actuator commands may be necessary.
            # The concept here is to form low-order sine and cosines that
            # correspond to the low-order modes that the HODM can form
            # and which could be duplicated by the LODM.
            # Choose all frequencies from zero to the LODM Nyquist frequency,
            # which may not be a small number.
            # These form sinusoids which are meant to be orthogonal, and as
            # the DMs are rectangular, they will be (*).
            # (*) but they actually won't be as I use actCds, but it should
            #  be close.
         modeCounter=0
         hoDMactCds=numpy.array(dm['ho'].actCds,numpy.float64).T
         for xf in range(0,int(lodmN/2*lambd[1])):
            print("_",end="")
            for yf in range(-int(lodmN/2*lambd[1]),int(lodmN/2*lambd[1])):
               print("^",end="")
               if xf==0 and yf==0: continue # piston mode
               for k,trigFn in enumerate((numpy.sin,numpy.cos)):  
#                  if xf==lodmN/2-1 and yf==lodmN/2-1 and trigFn==numpy.cos:
#                     continue # too high f.
                  #print("{0:d}/{1:d}".format(xf,yf),end="");sys.stdout.flush()
                  filterM[
                     modeCounter,
                     (dmOrder[0]=='lo')*sum(dm['lo'].usable):
                     (dmOrder[0]=='lo')*sum(dm['lo'].usable)
                      +sum(dm['ho'].usable)
                         ]=(
                        trigFn( 2*lambd[1]**-1.0*numpy.pi*hodmN**-1.0*(
                           xf*hoDMactCds[0]+yf*hoDMactCds[1]) )).take(
                              dm['ho'].usableIdx )
                  modeCounter+=1
                  #print(modeCounter,xf,yf,k)#,filterM[modeCounter-1,-10:])
         print(".",end="") ; sys.stdout.flush()
         fTfM=filterM.T.dot(filterM)
         print(".",end="") ; sys.stdout.flush()
         reconMs[reconType]=\
               numpy.linalg.pinv( sTsM+lambd[0]*fTfM).dot(stackedPMX.T)

      elif reconType=="PMX_filtering":
         # A crude but not inaccurate approach to filtering the HODM poke
         # matrix such that it does not contain the measurements produced by
         # the LODM:
         #  Store the actuators that must be poked in the LODM together
         #  with each poke on the HODM that produce those signals on the WFS
         #  which are independent from the WFS signals from pokes of the LODM
         #  alone.
         #  Return the filtered poke matrix, so LODM signals are independent
         #  from HODM signal.
         #  Return also those pokes on the HODM that correspond to producing
         #  LODM-independent signals from the HODM pokes.
         sI={};eI={}
         for dmType in 'lo','ho':
            sI[dmType],eI[dmType]=[
               sum([ numpy.sum(dm[dmOrder[i]].usable)
                  for i in range(dmOrder.index(dmType)+j) ]) for j in (0,1) ]
         loPMX=stackedPMX.take(range(sI['lo'],eI['lo']),axis=1)
         hoPMX=stackedPMX.take(range(sI['ho'],eI['ho']),axis=1)
         loPMXCovM=loPMX.T.dot(loPMX)
         cholLoPMXCovM=numpy.linalg.cholesky( loPMXCovM )
         cholLoPMXCovM_i=numpy.linalg.inv(cholLoPMXCovM) # by defn, not singular
         loPMX_orthM=loPMX.dot(cholLoPMXCovM_i.T)
         loModesHoPMX=numpy.array([
            [ (hoPMX[:,x]*loPMX_orthM[:,y]).sum()*loPMX_orthM[:,y]
               for y in range(sum(dm['lo'].usable)) ]
                  for x in range(sum(dm['ho'].usable)) ]).T.sum(axis=1)
         stackedFilteredPMX=stackedPMX.copy()
         stackedFilteredPMX[:,sI['ho']:eI['ho']]-=loModesHoPMX # remove
         fhoPMX=stackedFilteredPMX[:,sI['ho']:eI['ho']] # filtered HO PMX
         #
         hoThoD=( hoPMX.T.dot(hoPMX).diagonal() ) # X-corr of HODM signals
         chosenI=lambda r:  numpy.flatnonzero( hoThoD>=r )
            # \/ chose pokes with cross-correlation amplitude
            # from the unfilterd HO poke mtrx -> these will be those
            # that produce the wavefront, so tacitly ignore the others
         chosenPMX=hoPMX[:,chosenI(lambd[1])] 
         cholchosenPMXTchosenPMX=\
               numpy.linalg.cholesky( chosenPMX.T.dot(chosenPMX) )
            # \/ converts the matrix to being orthogonal
         cholchosenPMXTchosenPMXT_i=numpy.linalg.inv(cholchosenPMXTchosenPMX.T)
            # \/ orthogonal modes from the unfiltered but chosen HO PMX 
            #  note that these are those that produce the largest signals
            #  so are /hopefully/ those which also are sufficient to build
            #  the filtered and unchosen (i.e. full) PMX
         orthchosenPMX=chosenPMX.dot(cholchosenPMXTchosenPMXT_i)
            # \/ the filtered HO modes correspond to the pokes which produce
            # the filtered HO PMX i.e. which pokes produce signals that
            # are orthogonal to the LODM signals.
         filteredHOModesM=numpy.zeros([sum(dm['ho'].usable)]*2)
         filteredHOModesM[chosenI(lambd[1])]=\
               ( cholchosenPMXTchosenPMXT_i.dot(orthchosenPMX.T.dot(fhoPMX)) )
         filtered2pokesM=numpy.zeros(
               [sum(dm['ho'].usable)+sum(dm['lo'].usable)]*2)
         filtered2pokesM[sI['lo']:eI['lo'],sI['lo']:eI['lo']]=\
               numpy.identity( sum(dm['lo'].usable) )
         filtered2pokesM[sI['ho']:eI['ho'],sI['ho']:eI['ho']]=\
               filteredHOModesM

         sTsM=stackedFilteredPMX.T.dot(stackedFilteredPMX)
            # \/ reconstructor in the LODM pokes & LODM-filtered HO modes bases
         rmxLFMB=numpy.linalg.pinv( sTsM, lambd[0]).dot(stackedFilteredPMX.T)
            # \/ reconstructor in the LODM pokes & HODM pokes bases
         rmx=filtered2pokesM.dot( rmxLFMB )
         reconMs[reconType]={
             'stackedFilteredPMX':stackedFilteredPMX,
             'filteredHOModesM':filteredHOModesM,
             'rmxLFMB':rmxLFMB,
             'rmx':rmx
            }

      # end if... elif...
      print("done")
   for key in reconMs.keys(): # check if reqd to put recon matrix into a dict
      if type(reconMs[key])!=dict:
         reconMs[key]={'rmx':reconMs[key]}
      elif 'rmx' not in reconMs[key].keys():
         raise ValueError("'rmx' not a key in reconMs")
   return reconMs

def doPlotsAndPrints(vizAs, ipVs, doPlotting=True):
   if doPlotting: import matplotlib.pyplot as pyp
   import string
   if doPlotting: pyp.figure(1)
   print(" "+"-"*70)
   phsMin,phsMax=vizAs['ipphase'].min(), vizAs['ipphase'].max()
   for i,dataToPlot in enumerate((
         [ ("ipphase","+"), ],
         [ ("ho","+"), ],
         [ ("lo","+"), ],
         [ ("ho","+"),("lo","+"),("ipphase","-") ] )):
      if doPlotting:
         pyp.figure(1)
         pyp.subplot(2,2,1+i)
         op=eval(string.join([ "{0[1]:s}vizAs['{0[0]:s}']".format(thisData)
               for thisData in dataToPlot ]))
         opTitle=string.join([ "{0[1]:s}{0[0]:s}".format(thisData)
               for thisData in dataToPlot ])
      if doPlotting:
         pyp.imshow( op, vmin=phsMin, vmax=phsMax )
         pyp.colorbar()
         pyp.title( opTitle )
      #

      opCmd=string.join([ "{0[1]:s}ipVs['{0[0]:s}']".format(thisData)
            for thisData in dataToPlot ])
      op=eval(opCmd)
      if doPlotting:
         pyp.figure(2)
         pyp.plot( op, label=opTitle )
      print(" range of {0:s}={1:f}".format( opCmd,op.ptp() ))
   #
   print(" "+"-"*70)
   if doPlotting:
      pyp.figure(2)
      pyp.xlabel("phase index (0...nPoints-1)")
      pyp.ylabel("phase (less mean)")
      pyp.legend(loc=0)
      dmLength=sum([ sum(dm[k].usable) for k in dm.keys() ])
      pyp.plot([0,dmLength-1],[ipVs['ho'].min()]*2,'k--',lw=1,alpha=0.5)
      pyp.plot([0,dmLength-1],[ipVs['ho'].max()]*2,'k--',lw=1,alpha=0.5)
      pyp.show()
   #


# --- main logic follows below ---

if __name__=="__main__":
   # -- variables --
   scaling=30
   nSubAps=scaling-1      # number of sub-apertures
   hodmN=scaling          # high-order DM number of actuators
   lodmN=scaling//4       # low-order DM number of actuators
   reconTypes=("SVD_only","Diagonal-regularized","Low-order-penalization",
         "Inv+Tik","PMX_filtering")
                          # /\ which reconstructors to calculate
                          # \/ regularization parameters
   lambds=[0.000001,(0.1,0.001),(1.0,1.0,0.001),0.00001,(0.00001,0.91)]
   try:
      reconTypeIdx=int(sys.argv[1])
      doPlotting=False
   except:
      reconTypeIdx=4         # which reconstructor to use (makes several types)
      doPlotting=True
   dmOrder=('lo','ho')    # which order to concatenate the poke matrices
   radii=[1,0.2]          # sub-aperture mask radii (relative)
   #numpy.random.seed(int(time.time()%18071977)) # set the seed, changing
   numpy.random.seed(18071977) # set the seed, fixed

   # (ends)
   # -- calculated variables ---

   mask=makeMask(radii,nSubAps)
   maskPupilDMOversized=\
         makeMask([max(1,radii[0]+0.1),min(0,radii[1]-0.1)],nSubAps+1)
   
   # form the gradient operator class and the operator matrix
   gO=gradientOperator.gradientOperatorType1( mask )
   gM=gO.returnOp()
   
   directTestPhase=generateTestPhase(gO,nSubAps/32.0,nSubAps+1) # test phase

   # make DMs
   dm={'ho':dm.dm(gO.n_,[hodmN]*2,maskPupilDMOversized),
       'lo':dm.dm(gO.n_,[lodmN]*2,maskPupilDMOversized) }
   
   pokeM=formPokeMatrices()
   stackedPMX=concatenatePokeMatrices(dm,dmOrder)
   reconT=reconTypes[reconTypeIdx]
   lambd=lambds[reconTypeIdx]
   reconMs=makeReconstructors([reconT], stackedPMX, dm, dmOrder, lambd)
   reconM=reconMs[reconT]['rmx']
   slopeV=numpy.dot( gM, directTestPhase )

   # (ends)
   # -- code logic begins --
   print("Using reconstructor: {0:s}".format(reconT))

      # generate actuator vector, split, and reconstruct DM surfaces
   actuatorV=numpy.dot( reconM, slopeV )
   actuatorVs={'lo':actuatorV[0:sum(dm[dmOrder[0]].usable)],
               'ho':actuatorV[sum(dm[dmOrder[0]].usable)
         :sum(dm[dmOrder[0]].usable)+sum(dm[dmOrder[1]].usable)]}
   
   reconPhaseVs={}
   for dmType in actuatorVs.keys():
      reconPhaseVs[dmType]=numpy.zeros([gO.numberPhases],numpy.float64)
      for j,actVal in enumerate(actuatorVs[dmType]):
         reconPhaseVs[dmType]+=(
               (dm[dmType].poke(dm[dmType].usableIdx[j])*actVal).take(
                     gO.illuminatedCornersIdx) )

      # \/ for visualisation
   ipVs={'ho':reconPhaseVs['ho'], 'lo':reconPhaseVs['lo'],
          'ipphase':directTestPhase }
   vizAs={}
   for key in ipVs:
      ipVs[key]-=ipVs[key].mean()
      thisA=numpy.zeros(gO.n_, numpy.float64)
      thisA.ravel()[gO.illuminatedCornersIdx]=ipVs[key]
      thisA=numpy.ma.masked_array( thisA, (gO.illuminatedCorners==0) ) 
      vizAs[key]=thisA-thisA.mean() 

   doPlotsAndPrints(vizAs,ipVs,doPlotting)
