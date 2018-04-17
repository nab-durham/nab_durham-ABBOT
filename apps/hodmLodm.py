from __future__ import print_function
# What is this?
# High-order/low-order separation

# TODO:
# (*) Ensure that the filterM's take into account the ptp of the relevant components of the stackedPMX

import numpy
import string

# ================
# Go to the bottom of this file to see test code
# ================

class knownReconstructors:
   SVDONLY = "SVD_only"
   INVTIK = "Inv+Tik"
   DIAGONALREGLN = "Diagonal-regularized"
   LOWORDERPNLZTN = "Low-order-penalization"
   HOPMXPNLZTN = "High-order-PMX_penalization"
   HOPMXPNLZTNTHY = "High-order-PMX_penalization_theory"
   PMXFLTRNG = "PMX_filtering"

knownReconstructorsList = [ vars(knownReconstructors)[ip]
            for ip in filter( lambda an : an[:2]!="__", vars(knownReconstructors) )
   ]

def printDot( op,extra=None):
   return # dummy code

def makeReconstructors(reconTypes, stackedPMX, dm, dmOrder,
      showSteps=False):
   reconMs={}
   dmLength=sum([ sum(dm[k].usable) for k in dm.keys() ])
   for reconType,lambd in reconTypes:
      if reconType not in knownReconstructorsList: 
         raise ValueError( "Unknown reconstructor '{0:s}', not in '{1:s}'".format(
                  reconType, string.join( knownReconstructorsList, ", " )) )
      printDot(showSteps, "makeReconstructors: Making '{0:s}' reconstructor".format(
               reconType))
      if reconType==knownReconstructors.SVDONLY:
         # SVD_only
         sTsM=stackedPMX.T.dot(stackedPMX)
         reconMs[reconType]={
               'rmx' :numpy.linalg.pinv( sTsM, lambd[0]).dot(stackedPMX.T),
               'svdtrunc': lambd[0]
            }
      elif reconType==knownReconstructors.INVTIK:
         # Inv+Tik
         sTsM=stackedPMX.T.dot(stackedPMX)
         tikhonovRegM=numpy.identity(dmLength)
         reconMs[reconType]={
               'rmx':numpy.linalg.inv( sTsM+tikhonovRegM*lambd[0] ).dot(stackedPMX.T),
               'lambd0':lambd[0],
            }
      elif reconType==knownReconstructors.DIAGONALREGLN:
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
         sTsM=stackedPMX.T.dot(stackedPMX)
         tikhonovRegM=numpy.identity(dmLength)
         reconMs[reconType]={
               'rmx':numpy.linalg.inv(
                        sTsM+lambd[0]*filterM.T.dot(filterM)+tikhonovRegM*lambd[1]
                                           ).dot( stackedPMX.T ),
               'filterM':filterM.diagonal(),
               'lambd0':lambd[0],
               'lambd1':lambd[1],
            }

      elif reconType==knownReconstructors.LOWORDERPNLZTN:
         print("\n\t*WARNING*:this method has a bug, excess waffle/high-frequency terms incorrectly regularized\n")
         lodmN=dm['lo'].actGeom[0]
         hodmN=dm['ho'].actGeom[0]
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
            printDot(showSteps,"_")
            for yf in range(-int(lodmN/2*lambd[1]),int(lodmN/2*lambd[1])):
               printDot(showSteps,"^")
               if xf==0 and yf==0: continue # piston mode
               for k,trigFn in enumerate((numpy.sin,numpy.cos)):  
#                  if xf==lodmN/2-1 and yf==lodmN/2-1 and trigFn==numpy.cos:
#                     continue # too high f.
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
         fTfM=filterM.T.dot(filterM)
         sTsM=stackedPMX.T.dot(stackedPMX)
         tikhonovRegM=numpy.identity(dmLength)
         reconMs[reconType]={\
               'rmx':numpy.linalg.pinv( 
                        sTsM+lambd[0]*fTfM+tikhonovRegM*lambd[2]
                                           ).dot( stackedPMX.T ),
               'filterM':filterM,
               'lambd0':lambd[0],
               'lambd1':lambd[1],
            }
      #
      elif reconType==knownReconstructors.HOPMXPNLZTNTHY:
         from abbot.projection import projection
         from abbot.gradientOperator import gradientOperatorType1
         ap,gO={},{}
         nSubaps,nActs={},{}
         # \/ geometry
         for tap in ['lo','ho']:
            ap[tap]       = numpy.array(dm[tap].usable).reshape(dm[tap].actGeom)
            gO[tap]       = gradientOperatorType1( pupilMask=ap[tap] )
            nSubaps[tap]  = gO[tap].numberSubaps
            nActs[tap]    = sum(dm[tap].usable)

         gM_ho = gO['ho'].returnOp()
#NOT_REQD.         ratios = [ (ap['ho'].shape[i]-1.0)/(ap['lo'].shape[i]-1.0) for i in (0,1) ]
         ratios = [ (ap['ho'].shape[i]-0.0)/(ap['lo'].shape[i]-0.0) for i in (0,1) ]
         assert ratios[0]==ratios[1], "Mismatched ratios, cannot continue"
         ratio = ratios[0] ; del ratios
         # \/ use projection to just map from low-order actuator positions to
         # high-order actuator coverage (one:many)
#NOT_REQD.         pO = projection(
#NOT_REQD.               [0], [0], [0], [gO['ho'].subapMask, gO['lo'].subapMask],
#NOT_REQD.               [None], [1, ratio]
#NOT_REQD.            )
         pO = projection( [0], [0], [0], [ap['ho'],ap['lo']], [None], [1, ratio]
            )
         pM = ( pO.layerCentreProjectionMatrix()
               ).take( pO.maskInLayerIdx(0,ap['ho']), 1 )
#NOT_REQD.         pM = ( pO.layerCentreProjectionMatrix()
#NOT_REQD.               ).take( pO.maskInLayerIdx(0,gO['ho'].subapMask), 1 )
#NOT_REQD.         spM = numpy.array(
#NOT_REQD.                  [[ pM, pM*0 ],
#NOT_REQD.                   [ pM*0, pM ] ] 
#NOT_REQD.               ).swapaxes(1,2).reshape( [2*pM.shape[0],-1]
#NOT_REQD.            )
#NOT_REQD.         spM_i = numpy.linalg.pinv( spM,1e-1 ) # no need to regularize
            # \/ form averaging matrix, over HO slopes
#NOT_REQD.         avgM = spM_i.dot( spM )
         avgM = pM
            # \/ check
#NOT_REQD.         assert avgM.sum(1).var()<1e-10, "Insufficient ho coverage for averaging"
# biljana
         filterM = numpy.zeros(
               [ avgM.shape[0], (nActs['lo']+nActs['ho']) ] )
         tikhonovRegM=numpy.identity( nActs['ho']+nActs['lo'] )
         assert filterM.shape[1]==stackedPMX.shape[1],\
               "Wrong assumption in building gradientOperator"
         if dmOrder[0]=='lo':
##            normalization = stackedPMX[nActs['lo']:].ptp()/0.25
            filterM[:,nActs['lo']:] = avgM##.dot(gM_ho*normalization)
         else:
##            normalization = stackedPMX[:nActs['ho']].ptp()/0.25
            filterM[:,:nActs['ho']] = avgM##.dot(gM_ho*normalization)
         
         fTfM=filterM.T.dot(filterM)
         sTsM=stackedPMX.T.dot(stackedPMX)
         reconMs[reconType]={\
               'rmx':numpy.linalg.inv(
                        sTsM+lambd[0]*fTfM+tikhonovRegM*lambd[1] 
                     ).dot(stackedPMX.T),
               'filterM':filterM,
               'lambd0':lambd[0],
               'lambd1':lambd[1],
            }
      #
      elif reconType==knownReconstructors.HOPMXPNLZTN:
         from abbot.projection import projection
         from abbot.gradientOperator import gradientOperatorType1
         ap,gO={},{}
         nSubaps,nActs={},{}
         # \/ geometry
         for tap in ['lo','ho']:
            nActs[tap]    = sum(dm[tap].usable)

         hoPMX=numpy.empty(stackedPMX.shape,numpy.float32)
         if dmOrder[0]=='lo':
            hoPMX[:,:nActs['lo']] = 0
            hoPMX[:,nActs['lo']:] = stackedPMX[:,nActs['lo']:]
            loPMX = stackedPMX[:,:nActs['lo']]
         else:
            hoPMX[:,nActs['ho']:] = 0
            hoPMX[:,:nActs['ho']] = stackedPMX[:,:nActs['ho']]
            loPMX = stackedPMX[:,nActs['ho']:]

         loPMX_i = numpy.linalg.pinv( loPMX, lambd[2] )
         filterM = loPMX_i.dot( hoPMX )

         tikhonovRegM=numpy.identity( nActs['ho']+nActs['lo'] )
         
         fTfM=filterM.T.dot(filterM)
         sTsM=stackedPMX.T.dot(stackedPMX)
         reconMs[reconType]={\
               'rmx':numpy.linalg.inv(
                        sTsM+lambd[0]*fTfM+tikhonovRegM*lambd[1] 
                     ).dot(stackedPMX.T),
               'filterM':filterM,
               'lambd0':lambd[0],
               'lambd1':lambd[1],
            }
      #
      elif reconType==knownReconstructors.PMXFLTRNG:
         if lambd[1]<0 or lambd[1]>1: raise ValueError("0<lambd[1]<1 != True")
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
         printDot(showSteps)
         loPMX=stackedPMX.take(range(sI['lo'],eI['lo']),axis=1)
         hoPMX=stackedPMX.take(range(sI['ho'],eI['ho']),axis=1)
         loPMXCovM=loPMX.T.dot(loPMX)
         printDot(showSteps)
         cholLoPMXCovM=numpy.linalg.cholesky( loPMXCovM )
         printDot(showSteps)
         cholLoPMXCovM_i=numpy.linalg.inv(cholLoPMXCovM) # by defn, not singular
         printDot(showSteps)
         loPMX_orthM=loPMX.dot(cholLoPMXCovM_i.T)
         loModesHoPMX=numpy.array([
            [ (hoPMX[:,x]*loPMX_orthM[:,y]).sum()*loPMX_orthM[:,y]
               for y in range(sum(dm['lo'].usable)) ]
                  for x in range(sum(dm['ho'].usable)) ]).T.sum(axis=1)
         stackedFilteredPMX=stackedPMX.copy()
         stackedFilteredPMX[:,sI['ho']:eI['ho']]-=loModesHoPMX # remove
         fhoPMX=stackedFilteredPMX[:,sI['ho']:eI['ho']] # filtered HO PMX
         printDot(showSteps)
         #
         hoThoD=( hoPMX.T.dot(hoPMX).diagonal() ) # X-corr of HODM signals
         chosenI=lambda r:  numpy.flatnonzero( hoThoD*hoThoD.max()**-1.>=r )
            # \/ chose pokes with cross-correlation amplitude
            # from the unfilterd HO poke mtrx -> these will be those
            # that produce the wavefront, so tacitly ignore the others
         chosenPMX=hoPMX[:,chosenI(lambd[1])] 
         printDot(showSteps,"+")
         chosenTchosenM = chosenPMX.T.dot(chosenPMX)
         try:
            cholchosenPMXTchosenPMX = numpy.linalg.cholesky( chosenTchosenM )
         except numpy.linalg.linalg.LinAlgError:
            print("WARNING: could not form Cholesky decomposition, will have to regularize to lambd[0]")
            eigVals = numpy.linalg.eigvals( chosenTchosenM )
#DEBUG#            print(min(eigVals.real),min(abs(eigVals)))
            regulnM = numpy.identity( sum(dm['ho'].usable)
                  )*abs(max(eigVals))*lambd[0] 
            cholchosenPMXTchosenPMX = numpy.linalg.cholesky(
                  chosenTchosenM+regulnM )
            choleskyRegularization = regulnM
         else:
            choleskyRegularization = None

            # \/ converts the matrix to being orthogonal
         printDot(showSteps,"-")
         cholchosenPMXTchosenPMXT_i=numpy.linalg.inv(cholchosenPMXTchosenPMX.T)
            # \/ orthogonal modes from the unfiltered but chosen HO PMX 
            #  note that these are those that produce the largest signals
            #  so are /hopefully/ those which also are sufficient to build
            #  the filtered and unchosen (i.e. full) PMX
         printDot(showSteps)
         orthchosenPMX=chosenPMX.dot(cholchosenPMXTchosenPMXT_i)
            # \/ the filtered HO modes correspond to the pokes which produce
            # the filtered HO PMX i.e. which pokes produce signals that
            # are orthogonal to the LODM signals.
         printDot(showSteps)
         hoPMX_i=numpy.linalg.pinv( hoPMX, lambd[0] ) # convert into poke space
         P_o = hoPMX_i.dot( orthchosenPMX ) # convert orthogonal, chosen modes into poke space of mirror
         filteredHOModesM=numpy.zeros([sum(dm['ho'].usable)]*2)
# -- \/ nuovo --------
         filteredHOModesM = P_o.dot(orthchosenPMX.T).dot(fhoPMX) # nuovo
# -- /\ nuovo --------
# -- \/ vecchio ------
#         filteredHOModesM[chosenI(lambd[1])]=\
#               ( cholchosenPMXTchosenPMXT_i.dot(orthchosenPMX.T.dot(fhoPMX)) )
# -- /\ vecchio ------
         filtered2pokesM=numpy.zeros(
               [sum(dm['ho'].usable)+sum(dm['lo'].usable)]*2)
         filtered2pokesM[sI['lo']:eI['lo'],sI['lo']:eI['lo']]=\
               numpy.identity( sum(dm['lo'].usable) )
         filtered2pokesM[sI['ho']:eI['ho'],sI['ho']:eI['ho']]=\
               filteredHOModesM

         sTsM=stackedFilteredPMX.T.dot(stackedFilteredPMX)
         printDot(showSteps)
            # \/ reconstructor in the LODM pokes & LODM-filtered HO modes bases
         rmxLFMB=numpy.linalg.pinv( sTsM, lambd[0]).dot(stackedFilteredPMX.T)
         printDot(showSteps)
            # \/ reconstructor in the LODM pokes & HODM pokes bases
         rmx=filtered2pokesM.dot( rmxLFMB )
         reconMs[reconType]={
             'stackedFilteredPMX':stackedFilteredPMX,
             'filteredHOModesM':filteredHOModesM,
             'rmxLFMB':rmxLFMB,
             'choleskyRegularization':choleskyRegularization,
             'rmx':rmx
            }

      # end if... elif...
      printDot(showSteps,"done")
   for key in reconMs.keys(): # check if reqd to put recon matrix into a dict
      if type(reconMs[key])!=dict:
         reconMs[key]={'rmx':reconMs[key]}
      elif 'rmx' not in reconMs[key].keys():
         raise ValueError("'rmx' not a key in reconMs/{:s}".format(key))
   return reconMs

# --- main logic follows below ---

if __name__=="__main__":
   import sys
   import abbot.dm as dm
   import abbot.gradientOperator as gradientOperator
   import abbot.phaseCovariance as phaseCovariance
   import time
   import argparse

   def printDot(op=True,extra=None):
      if op and type(extra)==type(None):
         print(".",end="")
         sys.stdout.flush()
      elif op and type(extra)!=type(None):
         print(str(extra),end="")
         sys.stdout.flush()

   def makeMask(radii,N):
      radius=numpy.add.outer(
            (numpy.arange(N)-(N-1)/2.)**2,(numpy.arange(N)-(N-1)/2.)**2 )**0.5
      return numpy.where( (radius<=(radii[0]*nSubAps/2.0)) &
                          (radius>=(radii[1]*nSubAps/2.0)), 1, 0)

   def generateTestPhase(gO,r0,N):
      L0=100.
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


   def doPlotsAndPrintsViz(vizAs, ipVs, doPlotting, nReps, reconPhaseVsds):
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
         print("{2:s} {0:s} = {1:5.3f}".format(
                  string.ljust(opCmd,45),
                  op.ptp(),
                  string.rjust("range of",15)
               ))
         print("{2:s} {0:s} = {1:5.3f}".format(
                  string.ljust(opCmd,45),
                  op.std(),
                  string.rjust("s.d. of",15)
               ))
      #
      if doPlotting:
         pyp.figure(2)
         pyp.xlabel("phase index (0...nPoints-1)")
         pyp.ylabel("phase (less mean)")
         pyp.legend(loc=0)
         dmLength=sum([ sum(dm[k].usable) for k in dm.keys() ])
         pyp.plot([0,dmLength-1],[ipVs['ho'].min()]*2,'k--',lw=1,alpha=0.5)
         pyp.plot([0,dmLength-1],[ipVs['ho'].max()]*2,'k--',lw=1,alpha=0.5)
         pyp.show()

      if nReps<5:
         print(" (insufficient repetitions {} for statistics)".format(nReps) )
      else:
         for dmType in 'ho','lo','resid':
            for fn in 'mean','std':
               opCmd = "numpy.{0:s}(reconPhaseVsds['{1:s}'])".format(fn,dmType)
               op=eval(opCmd)
               if fn=='mean':
                  opPrefixSuffix="<",">"
               elif fn=='std':
                  opPrefixSuffix="\sigma_{","}"
               print("{2:s} {0:s} = {1:5.3f}".format(
                     string.ljust("{0:s}".format( dmType ),30),
                     op,
                     string.rjust(
                       "{0[0]:s}s.d.{0[1]:s}".format(opPrefixSuffix),30)
                  ))
      print(" "+"-"*70)
   
   def doPlotsAndPrintsAct(vizAs, ipVs, doPlotting, nReps, actuatorVsds):
      if doPlotting: import matplotlib.pyplot as pyp
      import string
      print(" "+"=-"*35)
      for i,dataToPlot in enumerate((
               [ ("ho","+"), ],
               [ ("lo","+"), ],
            )):
         if doPlotting:
            pyp.figure(3)
            pyp.subplot(2,1,1+i)
            op=eval(string.join([ "{0[1]:s}vizAs['{0[0]:s}']".format(thisData)
                  for thisData in dataToPlot ]))
            opTitle=string.join([ "{0[1]:s}{0[0]:s}".format(thisData)
                  for thisData in dataToPlot ])
         if doPlotting:
            pyp.imshow( op )
            pyp.colorbar()
            pyp.title( opTitle )
         #

         opCmd=string.join([ "{0[1]:s}ipVs['{0[0]:s}']".format(thisData)
               for thisData in dataToPlot ])
         op=eval(opCmd)
         if doPlotting:
            pyp.figure(4)
            pyp.plot( op, label=opTitle )
         print("{2:s} {0:s} = {1:5.3f}".format(
                  string.ljust(opCmd,45),
                  op.ptp(),
                  string.rjust("range of",15)
               ))
         print("{2:s} {0:s} = {1:5.3f}".format(
                  string.ljust(opCmd,45),
                  op.std(),
                  string.rjust("s.d. of",15)
               ))
      #
      if doPlotting:
         pyp.figure(4)
         pyp.xlabel("actuator index (0...nPoints-1)")
         pyp.ylabel("actuator cmd")
         pyp.legend(loc=0)
         for k,c in ('lo','g'),('ho','b'):
            dmLength=sum(dm[k].usable)
            pyp.plot([0,dmLength-1],
                  [ipVs[k].min()]*2,c+'--',lw=1,alpha=0.5)
            pyp.plot([0,dmLength-1],
                  [ipVs[k].max()]*2,c+'--',lw=1,alpha=0.5)
         pyp.show()

      if nReps<5:
         print(" (insufficient repetitions {} for statistics)".format(nReps) )
      else:
         for dmType in 'ho','lo':
            for fn in 'mean','std':
               opCmd = "numpy.{0:s}(actuatorVsds['{1:s}'])".format(fn,dmType)
               op=eval(opCmd)
               if fn=='mean':
                  opPrefixSuffix="<",">"
               elif fn=='std':
                  opPrefixSuffix="\sigma_{","}"
               print("{2:s} {0:s} = {1:5.3f}".format(
                     string.ljust("{0:s}".format( dmType ),30),
                     op,
                     string.rjust(
                       "{0[0]:s}s.d.{0[1]:s}".format(opPrefixSuffix),30)
                  ))
      print(" "+"-="*35)

#  =========================================================================

   # \/ regularization parameters
   lambds={
         "SVD_only"                    :(0.0001,),
         "Inv+Tik"                     :(0.0001,),
         "Diagonal-regularized"        :(0.01,0.0001),
         "Low-order-penalization"      :(0.01,3.0,0.001),
         "High-order-PMX_penalization_theory" :(0.1,0.0001),
         "High-order-PMX_penalization" :(0.1,0.0001,1e-2),
         "PMX_filtering"               :(1e-6,0.001)
      }
   assert numpy.std([ lk in knownReconstructorsList for lk in lambds ]
         )==0, "Mismatch"

   parser=argparse.ArgumentParser(
         description='investigate CDRAGON-type LTAO LGS asterism diameters' )
   parser.add_argument('reconstructorIdx',
         help= 'Reconstructor name\n'+
            ("NOTE: known reconstructors are:\n"
                  +string.join(["\n\t{:s},".format(n) for n in lambds.keys() ])
               ),
         nargs=1, type=str ) 
   parser.add_argument('-f', help='fix seed', action='store_true')
   parser.add_argument('-P', help='Do plotting', action='store_true')
   parser.add_argument('-N',
         help='HO no. actuators, N x N', type=int, default=18 )
   parser.add_argument('-n',
         help='LO no. actuators, n x n', type=int, default=6 )
   parser.add_argument('-R',
         help='Outer radius', type=float, default=1 )
   parser.add_argument('-r',
         help='Inner radius', type=float, default=0.2 )
   parser.add_argument('-g',
         help='HO gain', type=float, default=1.0 )
   parser.add_argument('-#',
         help='Number of reps', type=int, default=1 )
   parser.add_argument('-S',
         help='SNR', type=float, default=None )
   args=parser.parse_args(sys.argv[1:])
   #
   # -- variables --
   fixSeed=args.f
   reconTypeIdx=args.reconstructorIdx[0]
   doPlotting=args.P
   dmOrder=('lo','ho')    # which order to concatenate the poke matrices
   radii=[args.R,args.r]  # sub-aperture mask radii (relative)
   if fixSeed:
      numpy.random.seed(18071977) # set the seed, fixed
   else:
      numpy.random.seed(int(time.time()%18071977)) # set the seed, changing
   scaling=args.N
   nSubAps=scaling-1      # number of sub-apertures
   hodmN=args.N           # high-order DM number of actuators i.e. Fried geom.
   lodmN=args.n           # low-order DM number of actuators
   hodmG=args.g           # high-order DM gain (just a multiplicative factor)
   nReps=args.__getattribute__("#")
   SNR=args.S             # SNR
   # -- calculated variables ---

   mask=makeMask(radii,nSubAps)
   maskPupilDMOversized=\
         makeMask([max(1,radii[0]+0.1),min(0,radii[1]-0.1)],nSubAps+1)
   
   # form the gradient operator class and the operator matrix
   gO=gradientOperator.gradientOperatorType1( mask )
   gM=gO.returnOp()

   # make DMs
   dm={'ho':dm.dm(gO.n_,[hodmN]*2,maskPupilDMOversized,ifScl=0.5),
       'lo':dm.dm(gO.n_,[lodmN]*2,maskPupilDMOversized,ifScl=0.5) }

   pokeM=formPokeMatrices()
   stackedPMX=concatenatePokeMatrices(dm,dmOrder)

   # >>> The following three lines show how to use the function
   # >>>
   reconT=( reconTypeIdx,lambds[reconTypeIdx] ) # configure
   print("Using reconstructor: {0:s}".format(reconT))
   reconMs=makeReconstructors([reconT], stackedPMX, dm, dmOrder, showSteps=1)
   reconM=reconMs[reconTypeIdx]['rmx'] # retrieve
   # >>>
   # >>> (ends)

   # -- test code logic begins --

   actuatorVsds,reconPhaseVsds = [
         {'ho':[],'lo':[]},
         {'ho':[],'lo':[],'resid':[]} ]
   noise=0
   print()
   printDot(True, "[ ")
   for repN in range(nReps):
         # generate phase->slopes->actuator vec->split->
      directTestPhase=generateTestPhase(gO,nSubAps/32.0,nSubAps+1) # test phase
      slopeV=numpy.dot( gM, directTestPhase ) # make slopes, the input
      if SNR is not None: # replace with actual noise
         if noise is 0: 
            signal=slopeV.std()
         noise=numpy.random.normal( 0, signal*SNR**-1.0, size=len(slopeV) )
      slopeV+=noise
      actuatorV=numpy.dot( reconM, slopeV )
      actuatorVs={'lo':actuatorV[0:sum(dm[dmOrder[0]].usable)],
                  'ho':actuatorV[sum(dm[dmOrder[0]].usable)
            :sum(dm[dmOrder[0]].usable)+sum(dm[dmOrder[1]].usable)]}
      actuatorVs['ho']*=hodmG
      printDot(True)


         # ->reconstruct DM surfaces
      reconPhaseVs={}
      for dmType in actuatorVs.keys():
         reconPhaseVs[dmType]=numpy.zeros([gO.numberPhases],numpy.float64)
         for j,actVal in enumerate(actuatorVs[dmType]):
            reconPhaseVs[dmType]+=(
                  (dm[dmType].poke(dm[dmType].usableIdx[j])*actVal).take(
                        gO.illuminatedCornersIdx) )

         actuatorVsds[dmType].append( actuatorVs[dmType].std() )
         reconPhaseVsds[dmType].append( reconPhaseVs[dmType].std() )

      reconPhaseVsds['resid'].append(
            (reconPhaseVs['lo']+reconPhaseVs['ho']-directTestPhase).std()
         )
      
   printDot(True," ]")
   print()

      # \/ for visualisation, just the last iterration
   ipVs={'ho':reconPhaseVs['ho'], 'lo':reconPhaseVs['lo'],
          'ipphase':directTestPhase }
   print("reconPhaseVs & directTestPhase:")
   vizAs={}
   for key in ipVs:
      ipVs[key]-=ipVs[key].mean()
      thisA=numpy.zeros(gO.n_, numpy.float64)
      thisA.ravel()[gO.illuminatedCornersIdx]=ipVs[key]
      thisA=numpy.ma.masked_array( thisA, (gO.illuminatedCorners==0) ) 
      vizAs[key]=thisA-thisA.mean() 

   doPlotsAndPrintsViz(vizAs,ipVs,doPlotting, nReps,reconPhaseVsds)
   
   ipVs={'ho':actuatorVs['ho'], 'lo':actuatorVs['lo']}
   print("actuatorVs:")
   vizAs={}
   for key in ipVs:
      thisA=numpy.zeros(dm[key].actGeom)
      thisA.ravel()[dm[key].usableIdx]=ipVs[key]
      thisA=numpy.ma.masked_array( thisA,
            numpy.array(dm[key].usable).reshape(dm[key].actGeom)==0 ) 
      vizAs[key]=thisA-thisA.mean() 

   doPlotsAndPrintsAct(vizAs,ipVs,doPlotting,nReps,actuatorVsds)

