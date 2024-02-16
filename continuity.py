"""ABBOT : Compute loop integrals for noise determination/removal

Use the fact that Gamma.(s+n1+n2), where Gamma is a loop integral operator,
s is the vector of slopes (this is Fried), and n is the noise on s,
Gamma.(s+n1+n2)==Gamma.n2 => s+n1=(s+n1+n2)-(Gamma^-1).(Gamma.(s+n1+n2))
Then n2 is non-physical noise (loop integrals are not equal to zero) and
n1 is physical noise (loop integrals=0).
For a NxN grid of points there are 2xNx(N-1) slopes but only (N-1)x(N-1)
loops. Thus always underdetermined which is why only n2 can be estimated.
Coded for the Fried geometry.
"""

from __future__ import print_function
import collections
import gradientOperator
import numpy

class loops( gradientOperator.geometryType1 ):
   gridAN=lambda self,an : [(an%self.n_[1]), (an//self.n_[1])]
   AN=lambda self,gan : gan[0]+gan[1]*self.n_[1]
   partitionReached=lambda self,pos: (
         ((pos[0])%self.partitionPeriod[0])==self.partitionPeriodOffset[0]
      or ((pos[1])%self.partitionPeriod[1])==self.partitionPeriodOffset[1] ) 
            # /\ vert/horiz
#         ((pos[0]+pos[1])%period)==0 or ((pos[0]-pos[1])%period)==0 ) # diag

   def _gradN( self, an, dirn ):
      cornerAN=(an-(dirn==1 or dirn==-2)*(self.n_[1])-(dirn>0) )
      cornerSA=cornerAN//self.n_[1]*(self.n_[1]-1)+cornerAN%self.n_[1]
      if cornerSA not in self.subapMask.ravel().nonzero()[0]:
         return None
      else:
         return self.subapMask.ravel().nonzero()[0].searchsorted(cornerSA)

   def __init__( self, subapMask=None, pupilMask=None, partitionPeriod=None, 
         partitionPeriodOffset=[0,0], rotated=False, loopTemplates=[1,2,-1,-2],
         verbose=False ):
      ''' partitionPeriod defines the partitioning of the geometry over which
      loops may not extend, although they may begin at a partition,
         
      partitionPeriodOffset shifts defines the partitioning,
         
      rotated implies if the gradients are pre-rotated.

      loopTemplates are based on a simple drawing language,
           so need a simple loop drawing language, 1=horiz, 2=vert, +ve or -ve
           (4 steps)
      Some examples,
        [1,1,2,-1,-1,-2], # 1x2
        [1,2,2,-1,-2,-2], # 2x1
        [1,1,2,2,-1,-2,-1,-2],
        [1,1,2,-1,2,-1,-2,-2],
        [1,2,1,2,-1,-1,-2,-2],
        [1,2,2,-1,-1,-2,1,-2],
        [1,2,1,2,-1,-2,-1,-2], # corner touching
        [1,2,2,1,-2,-1,-1,-2], # corner touching
        [1,1,2,2,-1,-1,-2,-2], # 2x2
      '''
      gradientOperator.geometryType1.__init__( self, subapMask, pupilMask )
      #
      self.partitionPeriod=partitionPeriod
      if type(self.partitionPeriod) in (int,float):
         self.partitionPeriod=[self.partitionPeriod]*2
      self.partitionPeriodOffset=partitionPeriodOffset
      self.rotated=rotated
      self.loopsDef=[]
      self.loopTemplates=loopTemplates
      if not isinstance( self.loopTemplates[0], collections.Iterable ):
         self.loopTemplates=[loopTemplates]
      self.verbose=verbose

   def calculateLoopsDef(self):
      if len(self.loopsDef)>0:
         return self.loopsDef # has already been calcualted
      for loopTNum,thisLoopTemplate in enumerate( self.loopTemplates ):
         if self.verbose:
            print(loopTNum,end=",")
         for xc in numpy.arange(self.n_[1]-1):
            for yc in numpy.arange(self.n_[0]-1):
               valid=True
               tan=xc+yc*self.n_[1] # this actuator number
               if tan not in self.illuminatedCornersIdx: continue
               tloopsDef=[]
               for tcmd in thisLoopTemplate:  
                  tgan=self.gridAN(tan) 
                  # run through template and make sure it fits
                  tgan[0]+=(abs(tcmd)==1)*numpy.sign(tcmd)\
                        +(abs(tcmd)==2)*numpy.sign(tcmd)
                  tgan[1]+=(abs(tcmd)==1)*numpy.sign(tcmd)\
                        -(abs(tcmd)==2)*numpy.sign(tcmd)
                  test_an=self.AN(tgan) # set new actuator number
                  if (test_an not in self.illuminatedCornersIdx
                        or tgan[0]>=self.n_[1] or tgan[0]<0
                        or self._gradN(test_an,tcmd)==None
                        or ((self.partitionPeriod!=None and
                             tcmd!=-2 and tcmd!=-1) and
                             self.partitionReached(tgan) )):
                     if (self.verbose and self.partitionPeriod!=None and
                           self.partitionReached(tgan)):
                        print(tgan[0],tgan[1])
                     valid=False
                     break
                  tan=test_an # so far, so good
                  if abs(tcmd)==1:
                  # the translation we want is that,
                  #  if abs(tmcd)==1
                  #   sign(tcmd)*[+1,+1] if not self.rotated else sign(tcmd)*[0,+1]
                  #  else
                  #   sign(tcmd)*[+1,-1] if not self.rotated else sign(tcmd)*[+1,0]
                     tloopsDef.append( (self._gradN(tan,tcmd),numpy.sign(tcmd),
                              (not self.rotated)*1, 1) )
                  else:
                     tloopsDef.append( (self._gradN(tan,tcmd),numpy.sign(tcmd),
                              1,(not self.rotated)*-1) )
   #                  tloopsDef.append(
   #                        (gradN(tan,tcmd,gO),1*numpy.sign(tcmd),-1*numpy.sign(tcmd)) )
               if valid:
                  self.loopsDef.append( tloopsDef )
                  if self.verbose:
                     print(":found, {0:d}".format(tan),end=",")
                     print("\t",tloopsDef)
      if self.verbose: print()
      return 

class loopsIntegrationMatrix( loops ):
   def __init__( self, subapMask=None, pupilMask=None, partitionPeriod=None, 
         partitionPeriodOffset=[0,0], rotated=False, loopTemplates=[1,2,-1,-2],
         sparse=False, verbose=False, reorderSlopes=False ):
      ''' reorderSlopes [False] : If False then assume slopes are (XXXX...YYYY)
         else slopes are (XYXY...XYXY) ordering. The latter produces a
         block-structured matrix.
      '''
      loops.__init__( self, subapMask, pupilMask, partitionPeriod, 
         partitionPeriodOffset, rotated, loopTemplates,
         verbose )
      self.reorderSlopes=reorderSlopes
      self.sparse=sparse
      self.loopIntM=None
      self.interleaveM=None
      self.interleaveIdx=None
      self.separateM=None
      self.separateIdx=None

   def _createSwappingMatrix(self,indices):
      Ngradients=self.numberSubaps*2
      if not self.sparse:
         sM=numpy.zeros([Ngradients]*2,numpy.int32)
         sM.ravel()[indices]=1
      else:
         import scipy.sparse
         sM=scipy.sparse.csr_matrix(
            ([1]*Ngradients,indices,range(Ngradients+1)),dtype=numpy.int32)
      return sM

   def createInterleaveMatrix(self):
      if self.interleaveM is not None:
         return self.interleaveM
      Ngradients=self.numberSubaps*2
         ## indices=( [ i*(Ngradients+1) for i in range(Ngradients/2) ]+
         ##           [ Ngradients+Ngradients/2+i*(Ngradients*2+1) for i in
         ##             range(Ngradients/2) ] )
      self.interleaveIdx=( (numpy.arange(Ngradients)%2)*(Ngradients*1.5)
            +(numpy.arange(Ngradients)//2)*(Ngradients*2+1)
              ).astype(numpy.int32)
      self.interleaveM=self._createSwappingMatrix(self.interleaveIdx)
      return self.interleaveM

   def createSeparateMatrix(self):
      if self.separateM is not None:
         return self.separateM
      Ngradients=self.numberSubaps*2
         ## indices=( [ i*(Ngradients+2) for i in range(Ngradients/2) ]+
         ##           [ Ngradients*Ngradients/2+1+i*(Ngradients+2) 
         ##             for i in range(Ngradients/2) ] )
      self.separateIdx=( numpy.arange(Ngradients)*(Ngradients+2)
            +(numpy.arange(Ngradients)>=(Ngradients/2))*(-Ngradients+1)
              ).astype(numpy.int32)
      self.separateM=self._createSwappingMatrix(self.separateIdx)
      return self.separateM

   def returnOp( self ):
      if self.loopIntM is not None:
         return self.loopIntM # has already been calculated
      self.calculateLoopsDef()
      Ngradients=self.numberSubaps*2
      Nloops=len(self.loopsDef)
      if self.reorderSlopes:
         self.createSeparateMatrix()
         mapper=lambda ip : self.separateIdx[ip]%Ngradients
      else:
         mapper=lambda ip : ip
      #
      if self.sparse:
         import scipy.sparse, scipy.sparse.linalg
         loopInt={'dat':[],'col':[],'i':[0],'counter':0}
         for i,tloopsDef in enumerate( self.loopsDef ):
            for k in tloopsDef:
               if k[2]:
                  loopInt['dat'].append( k[2]*k[1] )
                  loopInt['col'].append( mapper(k[0]) )
                  loopInt['counter']+=1
               if k[3]:
                  loopInt['dat'].append( k[3]*k[1] )
                  loopInt['col'].append( mapper(k[0]+Ngradients//2) )
                  loopInt['counter']+=1
            loopInt['i'].append(loopInt['counter'])
         self.loopIntM=scipy.sparse.csr_matrix(
               (loopInt['dat'],loopInt['col'],loopInt['i']),
               [Nloops,Ngradients], dtype=numpy.float32)
      else:
         self.loopIntM=numpy.zeros([Nloops,Ngradients], numpy.int16)
         for i,tloopsDef in enumerate( self.loopsDef ):
            for k in tloopsDef:
               if k[2]: self.loopIntM[i,mapper(k[0])]=k[2]*k[1]
               if k[3]: self.loopIntM[i,mapper(k[0]+Ngradients//2)]=k[3]*k[1]
      #
      return self.loopIntM

class loopsNoiseMatrices( loopsIntegrationMatrix ):
   def __init__( self, subapMask=None, pupilMask=None, partitionPeriod=None, 
         partitionPeriodOffset=[0,0], rotated=False, loopTemplates=[1,2,-1,-2],
         sparse=False, verbose=False, reorderSlopes=False,
         regularization=None
      ):
      ''' reorderSlopes [False] : If False then assume slopes are (XXXX...YYYY)
         else slopes are (XYXY...XYXY) ordering. The latter produces a
         block-structured matrix.
      '''
      loopsIntegrationMatrix.__init__(
            self, subapMask, pupilMask, partitionPeriod, 
            partitionPeriodOffset, rotated, loopTemplates,
            sparse, verbose, reorderSlopes
         )
      self.regularization=regularization

   def returnOp( self ):
      if 'noiseExtM' in dir(self) and 'noiseReductionM' in dir(self):
         # has already been calculated
         return self.noiseExtM, self.noiseReductionM
      if self.loopIntM is None:
         # prepare the loop integration matrix
         loopsIntegrationMatrix.returnOp(self) 
      if not self.sparse:
         # define the inverse
         if type(self.regularization) in (int,float,type(None)):
            self.regularizationM=numpy.identity(
                  self.numberSubaps*2 )*(
                  0.1 if self.regularization==None else self.regularization )
         elif isinstance(self.regularization,numpy.ndarray):
            assert self.regularization.shape==\
                     ( self.numberSubaps*2, self.numberSubaps*2 ),\
                  "Incorrectly sized regularization"
            self.regularizationM=self.regularization
         else:
            raise TypeError(
                  "Except regularization to be int,float,ndarray,None")
         ilIM=numpy.dot(
            numpy.linalg.inv(
                     numpy.dot(self.loopIntM.T, self.loopIntM)
                     +self.regularizationM
                  ), self.loopIntM.T 
               )
         identM=numpy.identity( self.numberSubaps*2 )
      else:
         # define the inverse
         import scipy.sparse, scipy.sparse.linalg
         if type(self.regularization) in (int,float,type(None)):
            self.regularizationM=scipy.sparse.identity(
                  self.numberSubaps*2 )*(
                  0.1 if self.regularization==None else self.regularization )
         elif isinstance(self.regularization,scipy.sparse,spmatrix):
            assert self.regularization.shape==\
                     ( self.numberSubaps*2, self.numberSubaps*2 ),\
                  "Incorrectly sized regularization"
            self.regularizationM=self.regularization
         else:
            raise TypeError(
                  "Except regularization to be int,float,ndarray,None")
         #
         ilTlvc, ilTlic, ilTll = [], [], [0]
         luliTliMsplu = scipy.sparse.linalg.splu(
               self.loopIntM.T.dot(self.loopIntM)+self.regularizationM )
         for jj in range( self.loopIntM.shape[1] ):
            this_liTliM_col=luliTliMsplu.solve(
                  numpy.arange( self.loopIntM.shape[1] )==jj )
            ilTli = this_liTliM_col.nonzero()[0]
            #
            ilTlic+=ilTli.tolist()
            ilTll.append( ilTll[-1]+len( ilTli ) )
            ilTlvc+=( this_liTliM_col.take( ilTli ) ).tolist()
         #
         ilIM=scipy.sparse.csr_matrix( (ilTlvc,ilTlic,ilTll),
               dtype=numpy.float32).dot(self.loopIntM.T)
         identM=scipy.sparse.identity( self.numberSubaps*2 )
      #
      self.noiseExtM=ilIM.dot(self.loopIntM) # matrix to return the noises, n2
      self.noiseReductionM=identM-self.noiseExtM # and to remove noise->(s+n1)
      #
      return self.noiseExtM, self.noiseReductionM

## #######################################################################
## test code follows
##

if __name__=="__main__":
   import datetime
   import sys

   def doSubapMask(roundAp,nfft):
      if roundAp:
         subapCds=numpy.add.outer(
               (numpy.arange(nfft-1)-(nfft-2)/2.)**2.0, 
               (numpy.arange(nfft-1)-(nfft-2)/2.)**2.0 )
         return (subapCds<=(nfft/2-0.5)**2)*\
                (subapCds>(((nfft*6)//39.0)/2-0.5)**2)
      else:
         return numpy.ones([nfft-1]*2,numpy.int32) # square
      
   def doFormalTest():
      global rdmIp_N, rdmIp_R, noiseExtM, noiseReductionM,\
            loopsNoiseReduction,rdmIp, success, failure,\
            loopTestslopes
      success={}
      failure={}
      #
      subapMask=doSubapMask(1,32)
      gO=gradientOperator.gradientOperatorType1(
          subapMask=subapMask, sparse=0 )
      gM_d=gO.returnOp()
      gM_inv=numpy.linalg.pinv( gM_d,1e-2)
      numpy.random.seed(18071977)
      rdmIp=numpy.random.normal(size=740*2).astype(numpy.float64)
      assert abs(rdmIp.var()-1.0337514094811717)<1e-5,\
            "FAILURE: i/p unexpected : "+str(
                  abs(rdmIp.var()-1.0337514094811717) )
      rdmIp_clean=gM_d.dot(gM_inv.dot(rdmIp))
      reorderingIdx=[]
      for i in range(gO.numberSubaps):
         reorderingIdx+=[i,i+gO.numberSubaps]
      loopTestslopes={
            'x':(gO.subapMaskIdx//gO.n[0]%2*2-1).tolist()
                  +[0]*gO.numberSubaps,
            'y':[0]*gO.numberSubaps+(gO.subapMaskIdx%gO.n[0]%2*2-1).tolist()
            }
      for dirn in 'x','y':
         ts=numpy.array(loopTestslopes[dirn])
         loopTestslopes[dirn]=[ ts, numpy.take(ts, reorderingIdx) ]
         tsR=ts.copy()
         tsR[:gO.numberSubaps]=ts[:gO.numberSubaps]-ts[gO.numberSubaps:]
         tsR[gO.numberSubaps:]=ts[:gO.numberSubaps]+ts[gO.numberSubaps:]
         loopTestslopes[dirn]+=[ tsR, numpy.take(tsR, reorderingIdx) ]
      #
      def loopIntegrationTest(testIp,success,failure):
         sparse,reorderSlopes,rotated,testIp=testIp
         #
         loopsIntegration=loopsIntegrationMatrix(
             subapMask=subapMask, pupilMask=None,
             partitionPeriod=None,
             partitionPeriodOffset=[0,0], rotated=rotated,
             loopTemplates=([1,2,-1,-2]),
             sparse=sparse, verbose=False, reorderSlopes=reorderSlopes )
         print("+++",end="") ; sys.stdout.flush()
         loopIntM=loopsIntegration.returnOp()
         print("...",end="") ; sys.stdout.flush()
         loopInt=loopIntM.dot( testIp )
         successCondition=sum( abs(loopInt)!=4 ),sum( abs(loopInt)!=4 )==0
         if successCondition[1] :
            success[testNo][0]+=1
            success[testNo].append( successCondition[0] )
         else:
            failure[testNo][0]+=1
            failure[testNo].append( successCondition[0] )
         return success,failure
         
      def noiseReductionTest(testIp,success,failure):
         (partitionPeriod,partitionPeriodOffset,sparse,
            reorderSlopes,rdmIp_R_var,rdmIp_N_var)=testIp
         loopsNoiseReduction=loopsNoiseMatrices(
             subapMask=subapMask, pupilMask=None,
             partitionPeriod=partitionPeriod,
             partitionPeriodOffset=partitionPeriodOffset, rotated=False,
             loopTemplates=([1,2,-1,-2]),
             sparse=sparse, verbose=False, reorderSlopes=reorderSlopes )
         print("+++",end="") ; sys.stdout.flush()
         noiseExtM,noiseReductionM=loopsNoiseReduction.returnOp()
         print("...",end="") ; sys.stdout.flush()
         rdmIp_N,rdmIp_R=[ thisM.dot( rdmIp_clean ) for thisM in 
                  ( noiseExtM,noiseReductionM ) ]
         for thisVal in (rdmIp_N.var()-rdmIp_N_var,
               rdmIp_R.var()-rdmIp_R_var):
            successCondition=( thisVal, thisVal<=1e-5 )
            if successCondition[1] :
               success[testNo][0]+=1
               success[testNo].append( successCondition[0] )
            else:
               failure[testNo][0]+=1
               failure[testNo].append( successCondition[0] )
         return success,failure
      # 
      tests=[
         ('dense, loop integration (X)',loopIntegrationTest,
            [0,0,0, loopTestslopes['x'][0]]),
         ('dense, loop integration (Y)',loopIntegrationTest,
            [0,0,0, loopTestslopes['y'][0]]),
         ('sparse, loop integration (X)',loopIntegrationTest,
            [1,0,0, loopTestslopes['x'][0]]),
         ('sparse, loop integration (Y)',loopIntegrationTest,
            [1,0,0, loopTestslopes['y'][0]]),
         ('dense, loop integration, XY (X)',
            loopIntegrationTest,[0,1,0, loopTestslopes['x'][1]]),
         ('dense, loop integration, XY (Y)',
            loopIntegrationTest,[0,1,0, loopTestslopes['y'][1]]),
         ('sparse, loop integration, XY (X)',
            loopIntegrationTest,[1,1,0, loopTestslopes['x'][1]]),
         ('sparse, loop integration, XY (Y)',
            loopIntegrationTest,[1,1,0, loopTestslopes['y'][1]]),
         ('dense, loop integration rotated (X)',loopIntegrationTest,
            [0,0,1, loopTestslopes['x'][2]]),
         ('dense, loop integration rotated (Y)',loopIntegrationTest,
            [0,0,1, loopTestslopes['y'][2]]),
         ('sparse, loop integration rotated (X)',loopIntegrationTest,
            [1,0,1, loopTestslopes['x'][2]]),
         ('sparse, loop integration rotated (Y)',loopIntegrationTest,
            [1,0,1, loopTestslopes['y'][2]]),
         ('dense, loop integration rotated, XY (X)',
            loopIntegrationTest,[0,1,1, loopTestslopes['x'][3]]),
         ('dense, loop integration rotated, XY (Y)',
            loopIntegrationTest,[0,1,1, loopTestslopes['y'][3]]),
         ('sparse, loop integration rotated, XY (X)',
            loopIntegrationTest,[1,1,1, loopTestslopes['x'][3]]),
         ('sparse, loop integration rotated, XY (Y)',
            loopIntegrationTest,[1,1,1, loopTestslopes['y'][3]]),
         #
         ('dense, zero expectation',noiseReductionTest,
            [ None,[0,0], 0,0, 0.54883125491680962,0.0]),
         ('sparse, zero expectation',noiseReductionTest,
            [ None,[0,0], 1,0, 0.54883125542142386,0.0]),
         ('dense, no partition period, X-then-Y slopes',
            noiseReductionTest,
            [ None,[0,0], 0,0, 0.549805281825,0.465663756734]),
         ('sparse, no partition period, X-then-Y slopes',
            noiseReductionTest,
            [ None, [0,0], 1,0, 0.549805281825,0.465663756734]),
         ('dense, no partition period, XY-pair slopes',
            noiseReductionTest,
            [ None,[0,0], 0,0, 0.549805281825,0.465663756734]),
         ('sparse, no partition period, XY-pair slopes',
            noiseReductionTest,
            [ None,[0,0], 1,0, 0.549805281825,0.465663756734]),
         ('dense, partition period=[8,8], X-then-Y slopes',
            noiseReductionTest,
            [ [8]*2,[0,0], 0,0, 0.74923128288715413, 0.27550468419389873]),
         ('sparse, partition period=[8,8], X-then-Y slopes',
            noiseReductionTest,
            [ [8]*2,[0,0], 1,0, 0.74923128288715413,0.27550468419389873]),
         ('dense, p-period,p-offset=[8, 8],[8,8], X-then-Y',
            noiseReductionTest,
            [ [8]*2,[8]*2, 0,0, 0.54980528182540556,0.46566375673395421]),
         ('sparse, p-period,p-offset=[8, 8],[8,8], X-then-Y',
            noiseReductionTest,
            [ [8]*2,[8]*2, 1,0, 0.54980528315082711,0.46566370408117008]),
      ]
      for testNo,(thisTestDesc,thisTestFn,thisTestIp) in enumerate(tests):
         if testNo in success.keys() or testNo in failure.keys():
            raise RuntimeError("Already have test no. {0:d}".format(testNo))
         else:
            success[testNo],failure[testNo]=[0],[0]
         print("TEST {0:2d}: / :{1:s}...".format(testNo,thisTestDesc),end="")
         sys.stdout.flush()
         #
         success,failure=thisTestFn(thisTestIp,success,failure)
         #
         print("(done)",end="")
         print("\rTEST {0:2d}:{1[0]:1d}/{2[0]:1d}".format(
               testNo, success[testNo] ,failure[testNo]) )
      # 
      return (success, failure)
   #
   titleStr="continuity.py, automated testing"
   print("\n{0:s}\n{1:s}\n".format(titleStr,len(titleStr)*"^"))
   print("BEGINS:"+str(datetime.datetime.now()))
   success,failure=doFormalTest()
   succeeded,failed,total=0,0,0
   for tk in failure:
      failed+=failure[tk][0]
      succeeded+=success[tk][0]
      total+=failure[tk][0]+success[tk][0]
   print("SUMMARY: {0:d}->{1:d} successes and {2:d} failures".format(
         total, succeeded, failed))
   print("ENDS:"+str(datetime.datetime.now()))
   sys.exit( failed>0 )
