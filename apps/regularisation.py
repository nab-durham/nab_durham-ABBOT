'''ABBOT : demonstration of using gradientOperator to build reconstructors
The reconstructors assume the Fried geoemetry and test several types of
reconstruction (inversion with least squares, minimum variance, SVD)
and with variable regularization.
'''
from __future__ import print_function
import commonSeed 
import gradientOperator
import matplotlib.pyplot as pg
import numpy
import sys
import time


## variables begin -----------------------------------------

# test code
r0=1 # in pixels
L0=2
nfft=24 # pixel size
noise=2e-0 # fraction of signal which is noise
#
rcond=1e-6 # for SVD, relative to *largest* eigenvalue of G^T.G
# alpha!=0 with Tikhonov regularisation
# beta!=0 with Biharmonic approximation to MAP/waffle regularisation
# gamma!=0 with full inverse covariance regularisation
alpha=1e-6
beta=0.25
gamma,gammaN = beta,(noise**2)**-1.0

## variables end -------------------------------------------

print("r0=",r0)
print("L0=",L0)
print("nfft=",nfft)
print("noise=",noise)
print("rcond=",rcond)
print("alpha,beta,gamma=",alpha,beta,gamma)
print("gammaN=",gammaN)

timings={}

def printOutputVariances():
   # remnant variances & waffle operator
   opValPairs=[
         ("input var=",
            onePhaseD.var()),
         ("waffle input amp=",
            numpy.dot(onePhaseV, waffleV)),
      ]
   for key in invgTgMs:
      opValPairs+=[
         ("input-recon ({0:s}) var=".format(key),
            (reconPhaseD[key]-onePhaseD).var()),
         ("waffle recon ({0:s}) amp=".format(key),
               numpy.dot(reconPhaseV[key], waffleV))
      ]
   #
   for opName,val in opValPairs:
      print(" "*(max([40-len(opName),0]))
            + opName + "{0:6.3g}".format(val))

def printTimings():
   for dat in (
         ("Phase covariance","phaseC"),
         ("Covariance regularization","phaseCReg"),
         ("Eig Values time","eigV"),
         ("gTgM","gtgM"),("Laplacian/Biharmonic","lOM"),
        ):
      print("{1}={0:5.3f}s".format(
         timings['e:'+dat[1]]-timings['s:'+dat[1]],dat[0]))

def doPlots():
   pg.figure(2)
   pg.subplot(221)
   pg.imshow( onePhaseD, interpolation='nearest', origin='lower',
      extent=[-1.5,nfft+0.5,-1.5,nfft+0.5] )
   pg.title("type 1: input phase")
   pg.colorbar()
   pg.subplot(222)
   pg.imshow( reconPhaseD[0], interpolation='nearest', origin='lower',
      extent=[-1.5,nfft+0.5,-1.5,nfft+0.5] )
   pg.title("reconstructed phase ({0:s})".format(typeOfRegularization))
   pg.colorbar()
   pg.subplot(234)
   pg.imshow( reconPhaseD[0]-onePhaseD, interpolation='nearest', origin='lower',
      extent=[-1.5,nfft+0.5,-1.5,nfft+0.5] )
   pg.title("diff ({0:s})".format(typeOfRegularization))
   pg.colorbar()
   pg.subplot(235)
   if type(reconM[1])!=type(None):
      pg.imshow( reconPhaseD[1]-onePhaseD, interpolation='nearest', origin='lower',
         extent=[-1.5,nfft+0.5,-1.5,nfft+0.5] )
      pg.title("diff (No reg.)")
      pg.colorbar()
   else:
      pg.text(0.1,0.1,"Inv failed")
   pg.subplot(236)
   if type(reconM[2])!=type(None):
      pg.imshow( reconPhaseD[2]-onePhaseD, interpolation='nearest', origin='lower',
         extent=[-1.5,nfft+0.5,-1.5,nfft+0.5] )
      pg.title("diff (SVD)")
      pg.colorbar()
   else:
      pg.text(0.1,0.1,"SVD failed")

def setup():
   # define pupil mask as sub-apertures
   pupilMask=numpy.ones([nfft]*2,numpy.int32)
   pupilCds=numpy.add.outer(
      (numpy.arange(nfft)-(nfft-1)/2.)**2.0,
      (numpy.arange(nfft)-(nfft-1)/2.)**2.0 )
   pupilMask=(pupilCds>(nfft/6)**2)*(pupilCds<(nfft/2)**2)
   gO=gradientOperator.gradientOperatorType1(pupilMask)
   gM=gO.returnOp()
   timings['s:gtgM']=time.time()
   gTgM=numpy.dot( gM.transpose(), gM )
   timings['e:gtgM']=time.time()
   waffleV=gradientOperator.waffleOperatorType1(pupilMask).returnOp()
   timings['s:lOM']=time.time()
   lO=gradientOperator.laplacianOperatorType1(pupilMask)
   lM=lO.returnOp() # laplacian operator
   timings['e:lOM']=time.time()
   #
   return pupilMask,gO,gM,gTgM,waffleV,lO,lM

def setupPhase():
   # define phase at corners of pixels, each of which is a sub-aperture
   timings['s:phaseC']=time.time()
   import phaseCovariance as pc
   covPlus=pc.covarianceMatrixFillInRegular(
         pc.covarianceDirectRegular(nfft+2,r0,L0)
      )
   choleskyC=pc.choleskyDecomp(covPlus)
   timings['e:phaseC']=time.time()
   timings['s:phaseCReg']=time.time()
   if gamma!=0:
      cov=pc.covarianceMatrixFillInMasked(
            pc.covarianceDirectRegular(nfft+1,r0,L0),
            gO.illuminatedCorners 
         )
      cov_I=numpy.linalg.inv(cov)
   else:
      cov_I=0
   timings['e:phaseCReg']=time.time()

   # generate phase and gradients
   onePhase=numpy.dot( choleskyC, numpy.random.normal(size=(nfft+2)**2 ) )

      # \/ for comparison purposes onePhase is deliberately too big, so 
      #  it is now fixed by calculating a mean
   meanO=numpy.zeros([(nfft+1)**2,(nfft+2)**2], numpy.float64)
   for i in range((nfft+1)**2):
      meanO[i,i//(nfft+1)*(nfft+2)+i%(nfft+1)]=0.25
      meanO[i,i//(nfft+1)*(nfft+2)+i%(nfft+1)+1]=0.25
      meanO[i,(i//(nfft+1)+1)*(nfft+2)+i%(nfft+1)]=0.25
      meanO[i,(i//(nfft+1)+1)*(nfft+2)+i%(nfft+1)+1]=0.25
   onePhase=numpy.dot(meanO,onePhase)

   onePhaseV=onePhase[gO.illuminatedCornersIdx]
   onePhase-=onePhaseV.mean() # normalise
   onePhase.resize([nfft+1]*2)
   onePhaseD=numpy.ma.masked_array(onePhase,gO.illuminatedCorners==0)
   #
   return( covPlus, cov_I, onePhase, onePhaseV, onePhaseD )

#################################################################
## logic begins
##

( pupilMask,gO,gM,gTgM,waffleV,lO,lM )=setup()
( covPlus, cov_I, onePhase, onePhaseV, onePhaseD )=setupPhase()


gradV=numpy.dot(gM,onePhaseV)
if 'noise' in dir():
   gradV+=numpy.random.normal(0,gradV.var()**0.5*noise,size=gO.numberSubaps*2)
   print("(Adding noise)")
lTlM_scaled=lM.T.dot(lM)/4.0

def doInversion(alpha,beta,gamma,gammaN):
   # now build control matrix,
   # linear least squares is (G^T G+alpha I+beta R+gamma C^-1)^-1 G^T
   try:
      invgTgM=numpy.linalg.inv( gTgM*gammaN
             + (0 if alpha==0 else alpha*numpy.identity(gO.numberPhases))
             + (0 if beta==0 else beta*lM.T.dot(lM)/4.0)
             + (0 if gamma==0 else gamma*cov_I)
         )
   except numpy.linalg.LinAlgError as exceptionErr:
      print("FAILURE: Inversion didn't work "+
         "({0:f},{1:f},{2:f},{3:f}), {4:s}".format(
               alpha,beta,gamma,gammaN, exceptionErr))
      invgTgM=None
   return invgTgM

def doSVD(rcond):
   # now build control matrix with SVD 
   try:
      # \/ SVD reconstruction
      invgTgSVDM=numpy.linalg.pinv( gTgM, rcond=rcond )
   except numpy.linalg.LinAlgError as exceptionErr:
      print("FAILURE: SVD didn't converge,{0:s}".format(exceptionErr))
      invgTgSVDM=None
   return invgTgSVDM

   # \/ no regularisation 

#(old)   # \/ Tikhonov reconstruction
#(old)timings['s:eigV']=time.time()
#(old)eigV=numpy.linalg.eigvals(gTgM) ; eigV.sort()
#(old)timings['e:eigV']=time.time()
#(old)assert sum([x==0 for x in alpha,beta,gamma])==2,\
#(old)    "Can only choose one of alpha, beta, or gamma"
#(old)typeOfRegularization="Tikhonov"*(alpha!=0)+\
#(old)      "Sparse C_phi^{-1} approx."*(beta!=0)+\
#(old)      "Dense C_phi^{-1}"*(gamma!=0)

invgTgMs={
      'alpha':doInversion(alpha,0,0,1), # Tikhonov
      'beta': doInversion(0,beta,0,1), # Covariance approximation, least squares
      'gamma':doInversion(0,0,gamma,gammaN), # Minimum variance
      'svd':  doSVD(rcond)
   }

print("Reconstructor::",end="") ; sys.stdout.flush()
# three reconstructor matrices
reconM,reconPhaseV,reconPhaseD={},{},{}
for invType in 'alpha','beta','gamma','svd':
   print("<{0:s}:".format(invType),end="") ; sys.stdout.flush()
   if not invgTgMs[invType] is None:
      print("MVM:".format(invType),end="") ; sys.stdout.flush()
      reconM[invType] = numpy.dot( invgTgMs[invType], gM.transpose() ) 
      if invType=='gamma' and gamma!=0: reconM[invType]*=gammaN
      #
      print("recon:".format(invType),end="") ; sys.stdout.flush()
      reconPhaseV[invType] = numpy.dot(reconM[invType], gradV)
      #
      thisPhaseD=numpy.zeros((nfft+1)**2,numpy.float64)
      thisPhaseD[gO.illuminatedCornersIdx]=reconPhaseV[invType]
      print("reimage>".format(invType),end="") ; sys.stdout.flush()
      reconPhaseD[invType] = numpy.ma.masked_array(
            thisPhaseD.reshape([nfft+1]*2), [gO.illuminatedCorners==0] )
   else: 
      print("*N/A*>".format(invType),end="") ; sys.stdout.flush()
      reconM[invType] = None
      #
      reconPhaseV[invType] = None
      #
      reconPhaseD = None

print()
printOutputVariances()
printTimings()
print("For plots, use subroutine doPlots()")
