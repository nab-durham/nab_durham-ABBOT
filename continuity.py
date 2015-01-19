# What is this?
#
# Use the fact that Gamma.(s+n1+n2), where Gamma is a loop integral operator,
# s is the vector of slopes (this is Fried), and n is the noise on s,
# Gamma.(s+n1+n2)==Gamma.n2 => s+n1=(s+n1+n2)-(Gamma^-1).(Gamma.(s+n1+n2))
# Then n2 is non-physical noise (loop integrals are not equal to zero) and
# n1 is physical noise (loop integrals=0).
# For a NxN grid of points there are 2xNx(N-1) slopes but only (N-1)x(N-1)
# loops. Thus always underdetermined which is why only n2 can be estimated.
# Coded for the Fried geometry.
#
# TODO: Make a class, inherited from gradientOperatorType1

from __future__ import print_function
import gradientOperator
import numpy as np

def loopsDefine( gO, partitionPeriod=None, partitionPeriodOffset=[0,0],
      verbose=False, rotated=False ):
   '''gO defines the geometry,
      partitionPeriod defines the partitioning of the geometry over which
       loops may not extend, although they may begin at a partition,
      partitionPeriodOffset shifts defines the partitioning,
      rotated implies if the gradients are pre-rotated.
   '''
   loopsDef=[]
   # use the following template:
   # 2x1, 1x2, 2x1 + 1 more at each corner
   # so need a simple loop drawing language, 1=horiz, 2=vert, +ve or -ve (4
   # states)
   loopTemplates=[
     #[1,1,2,-1,-1,-2], # 1x2
     #[1,2,2,-1,-2,-2], # 2x1
     #[1,1,2,2,-1,-2,-1,-2],
     #[1,1,2,-1,2,-1,-2,-2],
     #[1,2,1,2,-1,-1,-2,-2],
     #[1,2,2,-1,-1,-2,1,-2],
     #[1,2,1,2,-1,-2,-1,-2], # corner touching
     #[1,2,2,1,-2,-1,-1,-2], # corner touching
     [1,2,-1,-2],
     #[1,1,2,2,-1,-1,-2,-2], # 2x2
    ] 

   gridAN=lambda an : [(an%gO.n_[1]), (an//gO.n_[1])]
   AN=lambda gan : gan[0]+gan[1]*gO.n_[1]
   def gradN(an,dirn,gO):
      cornerAN=(an-(dirn==1 or dirn==-2)*(gO.n_[1])-(dirn>0) )
      cornerSA=cornerAN//gO.n_[1]*(gO.n_[1]-1)+cornerAN%gO.n_[1]
      if cornerSA not in gO.subapMask.ravel().nonzero()[0]:
         return None
      else:
         return gO.subapMask.ravel().nonzero()[0].searchsorted(cornerSA)

   partitionReached=lambda pos,period: (
         ((pos[0])%period)==partitionPeriodOffset[0]
      or ((pos[1])%period)==partitionPeriodOffset[1] ) # vert/horiz
#         ((pos[0]+pos[1])%period)==0 or ((pos[0]-pos[1])%period)==0 ) # diag
   for loopTNum in range(len(loopTemplates)):
      if verbose:
         print(loopTNum,end=",")
      for xc in np.arange(gO.n_[1]-1):
         for yc in np.arange(gO.n_[0]-1):
            valid=True
            tan=xc+yc*gO.n_[1] # this actuator number
            if tan not in gO.illuminatedCornersIdx: continue
            tloopsDef=[]
            for tcmd in loopTemplates[loopTNum]:  
               tgan=gridAN(tan) 
               # run through template and make sure it fits
               tgan[0]+=(abs(tcmd)==1)*np.sign(tcmd)\
                     +(abs(tcmd)==2)*np.sign(tcmd)
               tgan[1]+=(abs(tcmd)==1)*np.sign(tcmd)\
                     -(abs(tcmd)==2)*np.sign(tcmd)
               test_an=AN(tgan) # set new actuator number
               if (test_an not in gO.illuminatedCornersIdx
                     or tgan[0]>=gO.n_[1] or tgan[0]<0
                     or gradN(test_an,tcmd,gO)==None
                     or ((partitionPeriod!=None and
                          tcmd!=-2 and tcmd!=-1) and
                          partitionReached(tgan,partitionPeriod) )):
                  if (verbose and partitionPeriod!=None and
                        partitionReached(tgan,partitionPeriod)):
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
                  tloopsDef.append( (gradN(tan,tcmd,gO),np.sign(tcmd),
                           (not rotated)*1, 1) )
               else:
                  tloopsDef.append( (gradN(tan,tcmd,gO),np.sign(tcmd),
                           1,(not rotated)*-1) )
#                  tloopsDef.append(
#                        (gradN(tan,tcmd,gO),1*np.sign(tcmd),-1*np.sign(tcmd)) )
            if valid:
               loopsDef.append( tloopsDef )
               if verbose:
                  print(":found, {0:d}".format(tan),end=",")
                  print("\t",tloopsDef)
   if verbose: print()
   return loopsDef

def loopsIntegrationMatrix( loopsDef, gO, sparse=False ):
   Ngradients=gO.numberSubaps*2
   Nloops=len(loopsDef)
   if not sparse:
      loopIntM=np.zeros([Nloops,Ngradients], np.int16)
   else:
      loopInt={'dat':[],'col':[],'i':[0],'counter':0}
   for i in range(len(loopsDef)):
      for k in loopsDef[i]:
         if not sparse:
            if k[2]: loopIntM[i,k[0]]=k[2]*k[1]
            if k[3]: loopIntM[i,k[0]+Ngradients//2]=k[3]*k[1]
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
      import scipy.sparse, scipy.sparse.linalg
      loopIntM=scipy.sparse.csr_matrix(
            (loopInt['dat'],loopInt['col'],loopInt['i']),[Nloops,Ngradients],
            dtype=np.float32)
   return loopIntM

def loopsNoiseMatrices( loopIntM, gO ):
   if type(loopIntM)==np.ndarray:
      sparse=False
   else:
      sparse=True
   if not sparse:
      # define the inverse
      ilIM=np.dot(
         np.linalg.inv( np.dot(loopIntM.T,loopIntM)
               +np.identity(loopIntM.shape[1])*0.1), loopIntM.T )
   else:
      import scipy.sparse, scipy.sparse.linalg
      luliTliM=scipy.sparse.linalg.splu(
            loopIntM.T.dot(loopIntM)+
            0.1*scipy.sparse.csr_matrix(
               (np.ones(loopIntM.shape[1]),
                np.arange(loopIntM.shape[1]),
                np.arange(loopIntM.shape[1]+1)
               )) )

      inv_liTliMT_list=[ luliTliM.solve(np.arange(loopIntM.shape[1])==jj)
            for jj in range(loopIntM.shape[1]) ]

      ilTli=[ x.nonzero()[0] for x in inv_liTliMT_list ]
      ilTll=[0]+[ len(x) for x in ilTli ]
      ilTlv=[ inv_liTliMT_list[i].take(ilTli[i])
            for i in range(len(inv_liTliMT_list)) ]

      ilTlvc=[]
      ilTlic=[]
      for x in ilTli: ilTlic+=x.tolist()
      for x in ilTlv: ilTlvc+=x.tolist()
      
      ilIM=scipy.sparse.csr_matrix(
            (ilTlvc,ilTlic,np.cumsum(ilTll)),dtype=np.float32)
      ilIM=ilIM.dot(loopIntM.T)
   noiseExtM=ilIM.dot(loopIntM) # matrix to return the noises, n2
   if not sparse:
      noiseReductionM=np.identity(gO.numberSubaps*2)-noiseExtM # reduce noise
   else:
      a=scipy.sparse.csr_matrix(
               (np.ones(gO.numberSubaps*2),
                np.arange(gO.numberSubaps*2),
                np.arange(gO.numberSubaps*2+1)
               ))
      noiseReductionM=a-noiseExtM # reduce noise
#   return ilIM,loopIntM
   return noiseExtM,noiseReductionM

if __name__=="__main__":
   import abbot.phaseCovariance as abbotPC
   import pylab
#?   import Zernike
   import sys

   if len(sys.argv)>1:
      nfft=int(sys.argv[1])
   else:
      nfft=10
#?   corners=Zernike.anyZernike(1,N,N//2+0.5,ongrid=1)
#?   #corners[3,3]=0 # N=7 then get CANARY
#?   if N>10: corners-=Zernike.anyZernike(1,N,((N//2)*6)//39.,ongrid=1)

   numpy=np
   subapMask=numpy.ones([nfft-1]*2,numpy.int32)
   subapCds=numpy.add.outer(
         (numpy.arange(nfft-1)-(nfft-2)/2.)**2.0, 
         (numpy.arange(nfft-1)-(nfft-2)/2.)**2.0 )
   subapMask=(subapCds<=(nfft/2-0.5)**2)*(subapCds>(((nfft*6)//39.0)/2-0.5)**2)
#>   pupilMask*=0 ; pupilMask+=1 # square
   gO=gradientOperator.gradientOperatorType1( subapMask, sparse=True )
   print(
      "Number subaps/corners={0.numberSubaps:d}/{0.numberPhases:d}".format(gO))

#?   gO=XXX.gradientOperatorType1( pupilMask=(corners) )
   corners=gO.illuminatedCorners!=0

   partitionPeriod=None
   if partitionPeriod!=None:
      print("Using partitioning ({0:d}) for block-reduction".format(
            partitionPeriod))
   print("Loops definition...",end="") ; sys.stdout.flush()
   loopsDef=loopsDefine( gO, partitionPeriod )
   Nloops=len(loopsDef) ; Ngradients=gO.numberSubaps*2
   print("Ngradients={0:3d}, Nloops={1:3d}".format(Ngradients, Nloops))
   
   
   print("...loop integration...") ; sys.stdout.flush()
   loopIntM=loopsIntegrationMatrix( loopsDef, gO, sparse=True )
#<sp>   # > the next bit makes it sparse
#<sp>   lIi=numpy.array([ x.nonzero()[0] for x in loopIntM ])
#<sp>   lIv=numpy.array([
#<sp>         loopIntM[i].take(lIi[i]) for i in range(loopIntM.shape[0]) ])
#<sp>   import scipy.sparse, scipy.sparse.linalg
#<sp>   sloopIntM=scipy.sparse.csr_matrix(
#<sp>         (lIv.ravel(),lIi.ravel(),
#<sp>          numpy.arange(loopIntM.shape[0]+1)*8),loopIntM.shape)
#<sp>   #> end of sparsification
   raise RuntimeError("STOP, have produced loopIntM") 
   print("Matrix creation...",end="") ; sys.stdout.flush()
   noiseExtM,noiseReductionM=loopsNoiseMatrices( loopIntM, gO )
   print("(done)") ; sys.stdout.flush()
   
   # \/ sparsify
   sparsifyFrac=0#.01 # fraction to eliminate
   if sparsifyFrac!=0:
      noiseReductionM.ravel()[np.arange(Ngradients)*(Ngradients+1)]-=1
      maxInM=abs(noiseReductionM).max()
      noiseReductionM=np.where( abs(noiseReductionM)>(maxInM*sparsifyFrac),
            noiseReductionM, 0 )
      noiseReductionM.ravel()[np.arange(Ngradients)*(Ngradients+1)]+=1
      print("SPARSE") 
   else:
      print("Non-sparse")

   gM=gO.returnOp()
   reconM=np.dot(
       np.linalg.inv( np.dot( gM.T,gM )+1e-4*np.identity(gO.numberPhases) ), 
       gM.T )
#   if Ngradients>Nloops:
#      print("Under-determined")
#   elif Ngradients==Nloops:
#      print("Well-determined")
#   elif Ngradients<Nloops:
#      print("Over-determined")
#
   # input
   # \/
#   rdmV=np.random.normal(0,1,size=gO.numberPhases)
#   directPCOne=abbotPC.covarianceDirectRegular( N, N/4.0, N*10 )
#   directPC=abbotPC.covarianceMatrixFillInMasked( directPCOne, corners )
#   directcholesky=abbotPC.choleskyDecomp(directPC)
#   testipV=np.dot(directcholesky, rdmV)
   testipV=np.zeros(gO.numberPhases)
   gradV=np.dot( gM, testipV )

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
   print("(",end="") ; sys.stdout.flush()
   for i in range(1000):
      if (i%100)==0: print(".",end="") ; sys.stdout.flush()
      ngradV.append( gradV+np.random.normal(0,1,size=Ngradients) )
#            np.random.normal(0,avars['grads']**0.5,size=len(gradV)) )
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
   
   print(")") ; sys.stdout.flush()

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
   print("Fraction nonzero={0:5.3f}".format(
        ((noiseReductionM-np.identity(Ngradients))!=0).sum()*Ngradients**-2.0 ))
