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
import gradientOperator
import collections
import numpy as np
import types

class loops( gradientOperator.geometryType1 ):
   gridAN=lambda self,an : [(an%self.n_[1]), (an//self.n_[1])]
   AN=lambda self,gan : gan[0]+gan[1]*self.n_[1]
   partitionReached=lambda pos,period: (
         ((pos[0])%period)==self.partitionPeriodOffset[0]
      or ((pos[1])%period)==self.partitionPeriodOffset[1] ) # vert/horiz
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
      self.partitionPeriodOffset=partitionPeriodOffset
      self.rotated=rotated
      self.loopsDef=[]
      self.loopTemplates=loopTemplates
      if not isinstance( self.loopTemplates[0], collections.Iterable ):
         self.loopTemplates=[loopTemplates]
      #
      for loopTNum,thisLoopTemplate in enumerate( self.loopTemplates ):
         if verbose:
            print(loopTNum,end=",")
         for xc in np.arange(self.n_[1]-1):
            for yc in np.arange(self.n_[0]-1):
               valid=True
               tan=xc+yc*self.n_[1] # this actuator number
               if tan not in self.illuminatedCornersIdx: continue
               tloopsDef=[]
               for tcmd in thisLoopTemplate:  
                  tgan=self.gridAN(tan) 
                  # run through template and make sure it fits
                  tgan[0]+=(abs(tcmd)==1)*np.sign(tcmd)\
                        +(abs(tcmd)==2)*np.sign(tcmd)
                  tgan[1]+=(abs(tcmd)==1)*np.sign(tcmd)\
                        -(abs(tcmd)==2)*np.sign(tcmd)
                  test_an=self.AN(tgan) # set new actuator number
                  if (test_an not in self.illuminatedCornersIdx
                        or tgan[0]>=self.n_[1] or tgan[0]<0
                        or self._gradN(test_an,tcmd)==None
                        or ((partitionPeriod!=None and
                             tcmd!=-2 and tcmd!=-1) and
                             self.partitionReached(tgan) )):
                     if (verbose and partitionPeriod!=None and
                           self.partitionReached(tgan)):
                        print(tgan[0],tgan[1])
                     valid=False
                     break
                  tan=test_an # so far, so good
                  if abs(tcmd)==1:
                  # the translation we want is that,
                  #  if abs(tmcd)==1
                  #   sign(tcmd)*[+1,+1] if not rotated else sign(tcmd)*[0,+1]
                  #  else
                  #   sign(tcmd)*[+1,-1] if not rotated else sign(tcmd)*[+1,0]
                     tloopsDef.append( (self._gradN(tan,tcmd),np.sign(tcmd),
                              (not rotated)*1, 1) )
                  else:
                     tloopsDef.append( (self._gradN(tan,tcmd),np.sign(tcmd),
                              1,(not rotated)*-1) )
   #                  tloopsDef.append(
   #                        (gradN(tan,tcmd,gO),1*np.sign(tcmd),-1*np.sign(tcmd)) )
               if valid:
                  self.loopsDef.append( tloopsDef )
                  if verbose:
                     print(":found, {0:d}".format(tan),end=",")
                     print("\t",tloopsDef)
      if verbose: print()
      return self.loopsDef

class loopsIntegrationMatrix( loops ):
   def __init__( self, subapMask=None, pupilMask=None, partitionPeriod=None, 
         partitionPeriodOffset=[0,0], rotated=False, loopTemplates=[1,2,-1,-2],
         sparse=False, verbose=False ):
      loops.__init__( self, subapMask, pupilMask, partitionPeriod, 
         partitionPeriodOffset, rotated, loopTemplates,
         verbose )
      self.sparse=sparse
      self.loopIntM=None

   def returnOp( self ):
      if type( self.loopIntM )!=types.NoneType: return self.loopIntM
      Ngradients=self.numberSubaps*2
      Nloops=len(self.loopsDef)
      if not self.sparse:
         self.loopIntM=np.zeros([Nloops,Ngradients], np.int16)
      else:
         import scipy.sparse, scipy.sparse.linalg
         loopInt={'dat':[],'col':[],'i':[0],'counter':0}
      for i,tloopsDef in enumerate( self.loopsDef ):
         for k in tloopsDef:
            if not sparse:
               if k[2]: self.loopIntM[i,k[0]]=k[2]*k[1]
               if k[3]: self.loopIntM[i,k[0]+Ngradients//2]=k[3]*k[1]
            else:
               if k[2]:
                  loopInt['dat'].append( k[2]*k[1] )
                  loopInt['col'].append( k[0] )
                  loopInt['counter']+=1
               if k[3]:
                  loopInt['dat'].append( k[3]*k[1] )
                  loopInt['col'].append( k[0]+Ngradients//2 )
                  loopInt['counter']+=1
         if sparse: loopInt['i'].append(loopInt['counter'])
      if sparse:
         self.loopIntM=scipy.sparse.csr_matrix(
               (loopInt['dat'],loopInt['col'],loopInt['i']),[Nloops,Ngradients],
               dtype=np.float32)
      #
      return self.loopIntM

class loopsNoiseMatrices( loopsIntegrationMatrix ):
   def returnOp( self ):
      self.loopIntM=loopsIntegrationMatrix.returnOp(self) # prepare the loop integration matrix
      if not self.sparse:
         # define the inverse
         ilIM=np.dot(
            np.linalg.inv(
                  np.dot(self.loopIntM.T, self.loopIntM)
                  +np.identity(self.loopIntM.shape[1])*0.1
               ), self.loopIntM.T )
      else:
         import scipy.sparse, scipy.sparse.linalg
         luliTliM=scipy.sparse.linalg.splu(
               self.loopIntM.T.dot(self.loopIntM)+
               0.1*scipy.sparse.csr_matrix(
                  (np.ones(self.loopIntM.shape[1]),
                   np.arange(self.loopIntM.shape[1]),
                   np.arange(self.loopIntM.shape[1]+1)
                  )) )
         #
         inv_liTliMT_list=[
            luliTliM.solve(np.arange(self.loopIntM.shape[1])==jj)
               for jj in range(self.loopIntM.shape[1]) ]

         ilTli=[ x.nonzero()[0] for x in inv_liTliMT_list ]
         ilTll=[0]+[ len(x) for x in ilTli ]
         ilTlv=[ inv_liTliMT_list[i].take(ilTli[i])
               for i in range(len(inv_liTliMT_list)) ]
         #
         ilTlvc=[]
         ilTlic=[]
         for x in ilTli: ilTlic+=x.tolist()
         for x in ilTlv: ilTlvc+=x.tolist()
         #
         ilIM=scipy.sparse.csr_matrix(
               (ilTlvc,ilTlic,np.cumsum(ilTll)),dtype=np.float32)
         ilIM=ilIM.dot(self.loopIntM.T)
      self.noiseExtM=ilIM.dot(self.loopIntM) # matrix to return the noises, n2
      if not sparse:
         self.noiseReductionM=np.identity(self.numberSubaps*2)-self.noiseExtM 
      else:
         a=scipy.sparse.csr_matrix(
                  (np.ones(self.numberSubaps*2),
                   np.arange(self.numberSubaps*2),
                   np.arange(self.numberSubaps*2+1)
                  ))
         self.noiseReductionM=a-self.noiseExtM 
      return self.noiseExtM, self.noiseReductionM

## #######################################################################
## test code follows
##

if __name__=="__main__":
   import pylab
   import sys

   # -- config begins -----------------------------------
   if len(sys.argv)>1:
      nfft=int(sys.argv[1])
   else:
      nfft=10
   roundAp=True
   sparse=False # True does *not* work at the moment
   partitionPeriod=None#[2,2]
   partitionPeriodOffset=[0,0]
   sparsifyFrac=0#.01 # fraction to eliminate
   nReps=100
   # -- config ends -------------------------------------

   numpy=np
   subapMask=numpy.ones([nfft-1]*2,numpy.int32)
   if 'roundAp' in dir():
      subapCds=numpy.add.outer(
            (numpy.arange(nfft-1)-(nfft-2)/2.)**2.0, 
            (numpy.arange(nfft-1)-(nfft-2)/2.)**2.0 )
      subapMask=(subapCds<=(nfft/2-0.5)**2)*\
                (subapCds>(((nfft*6)//39.0)/2-0.5)**2)
   else:
      subapMask.ravel()[:]=1 # square

   if partitionPeriod!=None:
      print("Using partitioning ({0[0]:d},{0[1]:d}) for block-reduction".format(
            partitionPeriod))
   gO=gradientOperator.gradientOperatorType1(
       subapMask=subapMask, sparse=sparse )
   loopsNoiseReduction=loopsNoiseMatrices(
       subapMask=subapMask, pupilMask=None,
       partitionPeriod=partitionPeriod,
       partitionPeriodOffset=partitionPeriodOffset, rotated=False,
       loopTemplates=([1,2,-1,-2]),
       sparse=sparse, verbose=False )
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
   noiseExtM,noiseReductionM=loopsNoiseReduction.returnOp()
   print("(done)") ; sys.stdout.flush()
   print("noiseReductionM!=0 fraction = {0:5.3f}".format(
         ((noiseReductionM-np.identity(Ngradients))!=0).sum()
         *Ngradients**-2.0 ))
   loopIntM=loopsNoiseReduction.loopIntM
  
   def doForceSparsify(sparsifyFrac,ipM):
      # \/ sparsify
      ipM.ravel()[np.arange(Ngradients)*(Ngradients+1)]-=1
      maxInM=abs(ipM).max()
      ipM=np.where( abs(ipM)>(maxInM*sparsifyFrac),
            ipM, 0 )
      ipM.ravel()[np.arange(Ngradients)*(Ngradients+1)]+=1
      return ipM
   
   if sparsifyFrac!=0:
      noiseReductionM = doForceSparsify(sparsifyFrac,noiseReductionM)
      print("Sparsified by {0:f}".format(sparsifyFrac)) 

   gM=gO.returnOp()
   if sparse: gM=np.array( gM.todense() )
   reconM=np.dot(
       np.linalg.inv( np.dot( gM.T,gM )+1e-4*np.identity(gO.numberPhases) ), 
       gM.T )

   # input
   # \/
#   rdmV=np.random.normal(0,1,size=gO.numberPhases)
#   import phaseCovariance as abbotPC
#   directPCOne=abbotPC.covarianceDirectRegular( N, N/4.0, N*10 )
#   directPC=abbotPC.covarianceMatrixFillInMasked( directPCOne, corners )
#   directcholesky=abbotPC.choleskyDecomp(directPC)
#   testipV=np.dot(directcholesky, rdmV)
   testipV = np.zeros( gO.numberPhases )
   gradV = np.dot( gM, testipV )

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
      ngradV.append( gradV+np.random.normal(0,1,size=Ngradients) )
      loopV=np.dot( loopIntM, ngradV[-1] )
      lessngradV=np.dot( noiseReductionM, ngradV[-1] )
      nvars['noise'].append(
            (ngradV[-1]-gradV).var() )
      nvars['left'].append(
            (lessngradV-gradV).var() )
      nvars['noisy_recon_var'].append(
            (np.dot(reconM,ngradV[-1])).var() )
      nvars['less_noisy_recon_var'].append(
            (np.dot(reconM,lessngradV)).var() )
      nvars['delta_noisy_recon_var'].append(
            (np.dot(reconM,ngradV[-1])-testipV).var() )
      nvars['delta_less_noisy_recon_var'].append(
            (np.dot(reconM,lessngradV)-testipV).var() )
   
   for k in avars.keys(): print("<{0:s}>={1:5.3f}".format(k,np.mean(avars[k])))
   print("--")
   for k in nvars.keys(): print("<{0:s}>={1:5.3f}".format(k,np.mean(nvars[k])))

   print("remnant gradient noise={0:5.3f}+/-{1:5.3f}".format(
         np.mean(nvars['left'])*np.mean(nvars['noise'])**-1.0,
         np.var(np.array(nvars['left'])*np.array(nvars['noise'])**-1.0)**0.5
#         (np.var(nvars['left'])+
#            (np.mean(nvars['left'])**2.0*np.mean(nvars['noise'])**-4.0)*
#               np.var(nvars['noise']) )**0.5 )
         ))
