'''ABBOT : demonstration of using gradientOperator to build reconstructors
The reconstructors assume the Fried geoemetry and test several types of
reconstruction (inversion with least squares, minimum variance, SVD)
and with variable regularization.
'''
from __future__ import print_function
##import commonSeed 
import gradientOperator
import numpy
import sys
import time


## variables begin -----------------------------------------
## configurable variables begin-----------------------------

# test code
r0=0.3333 # in pixels
L0=1.0
nfft=12 # pixel size
noise=0.1# fraction of signal which is noise
#
rcond=1e-6 # for SVD, relative to *largest* eigenvalue of G^T.G
# alpha!=0 with Tikhonov regularisation
# beta!=0 with Biharmonic approximation to MAP/waffle regularisation
# gamma!=0 with full inverse covariance regularisation
alpha=1e-6
beta=noise**2*0.25*4/7.0*(7/1.0e2)
if beta<1e-6: beta=1e-6
gamma,gammaN = 0.25,(noise**2)**-1.0

## configurable variables end ------------------------------
timings={}
obscuration=(0.5,1/4.0) # outer diameter and inner diameter
## variables end -------------------------------------------


def printOutputVariances():
   # remnant variances & waffle operator
   opValPairs=[
         ("input var=",
            onePhaseV.var()),
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
   print(" "+"-"*40)
   title="Output variances"
   print(" +"+title+"-"*(40-len(title)))
   for opName,val in opValPairs:
      print(" "*(max([40-len(opName),0])) + opName + "{0:6.3g}".format(val))
   #   
   print(" "+"-"*40)

def printTimings():
   title="Timings (in seconds)"
   print(" +"+title+"="*(40-len(title)))
   for dat in (
         ("Phase covariance","phaseC"),
         ("Covariance regularization","phaseCReg"),
         ("Eig Values time","eigV"),
         ("gTgM","gtgM"),("Laplacian/Biharmonic","lOM"),
        ):
      if 's:'+dat[1] not in timings.keys() or 'e:'+dat[1] not in timings.keys():
         print("(WARNING: timing key '{0:s}' not found)".format(dat[1]))
      else:
         opName=dat[1]+":= "
         val=timings['e:'+dat[1]]-timings['s:'+dat[1]]
         #
         print(" "*(max([40-len(opName),0])) + opName + "{0:6.3g}s".format(val))
   #
   print(" "+"="*40)

def doPlots():
   import matplotlib.pyplot as pg
   pg.figure(2)
   pg.subplot(231)
   pg.imshow( onePhaseD, interpolation='nearest', origin='lower',
      extent=[-1.5,nfft+0.5,-1.5,nfft+0.5] )
   pg.title("t.1 i/p")
   pg.colorbar()
   pg.subplot(232)
   pg.imshow( reconPhaseD['gamma'], interpolation='nearest', origin='lower',
      extent=[-1.5,nfft+0.5,-1.5,nfft+0.5] )
   pg.title("recon.\nphase ({0:s})".format('gamma'),size=10)
   pg.colorbar()
   pg.subplot(233)
   pg.imshow( reconPhaseD['svd'], interpolation='nearest', origin='lower',
      extent=[-1.5,nfft+0.5,-1.5,nfft+0.5] )
   pg.title("recon. phase\n({0:s})".format('svd'),size=10)
   pg.colorbar()
   for i,key in enumerate( ['alpha','beta2','gamma','svd'] ):
      pg.subplot(245+i)
      if not reconPhaseD[key] is None:
         pg.imshow( reconPhaseD[key]-onePhaseD,
            interpolation='nearest', origin='lower',
            extent=[-1.5,nfft+0.5,-1.5,nfft+0.5] )
         pg.title("diff ({0:s})".format(key),size=8)
         pg.colorbar()
      else:
         pg.text(0.1,0.1,key+" failed")

def setupGeometry(nfft,obscuration=[0.5,0]):
   pupilMask=numpy.ones([nfft]*2,numpy.int32)# pupil mask:=sub-apertures
   pupilCds=numpy.add.outer(
      (numpy.arange(nfft)-(nfft-1)/2.)**2.0,
      (numpy.arange(nfft)-(nfft-1)/2.)**2.0 )**0.5*nfft**-1.0
   pupilMask=(pupilCds>obscuration[1])*(pupilCds<obscuration[0])
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

def setupPhase(r0,L0):
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

def doVariableNoise(noises, ipV):
   """noises is a iterable containing fractions of noise for the analysis
   """
   title="VARIABLE NOISE ANALYSIS"
   print(" "+title+int(0.5*(60-len(title)))*"-="+"\n")
   #
   invgTgMs={
         'alpha':doInversion(alpha,0,0,1),
         'beta': doInversion(0,beta,0,1),
         'svd':  doSVD(rcond)
      }
   opReconDiff,opReconDiffVars={},{}
   #
   for noise in noises:
      print("(noise:="+str(noise),end="") ; sys.stdout.flush()
      gammaN = 1.0 if noise==0 else (noise**2)**-1.0
      #
      # add next two reconstructors,
      #  . Covariance approximation, min var 
      #  . Minimum variance
      invgTgMs['beta2']=doInversion(0,beta,0,gammaN)
      invgTgMs['gamma']=doInversion(0,0,gamma,gammaN) 
      #
      gradV=numpy.dot(gM,ipV)
      if noise>0:
         gradV+=numpy.random.normal(0,gradV.var()**0.5*noise,
               size=gO.numberSubaps*2)
      print(".",end="") ; sys.stdout.flush()
      #
      for key in invgTgMs.keys():
         reconM = numpy.dot( invgTgMs[key], gM.transpose() ) 
         if key in ('beta2','gamma'): reconM*=gammaN
         print("+",end="") ; sys.stdout.flush()
         #
         if key not in opReconDiff: opReconDiff[key]=[]
         if key not in opReconDiffVars: opReconDiffVars[key]=[]
         opReconDiff[key].append( numpy.dot(reconM, gradV)-ipV)
         opReconDiffVars[key].append( 
               (numpy.dot(reconM, gradV)-ipV).var() )
      
      print(".)",end="") ; sys.stdout.flush()

   #
   reconKeys=[]
   for i,key in enumerate(invgTgMs): reconKeys.append((i,key))
   #
   opReconDiffs=numpy.empty( [len(reconKeys),len(noises)], numpy.float64 )
   for i,key in reconKeys:
      opReconDiffs[i]=numpy.array(opReconDiffVars[key]).T
   #
   return (noises,reconKeys,opReconDiffs,opReconDiff)


def doVariableNoiseComparison():
   deltaNoise=0.1
   nReps=10
   noises=[]
   for thisNoise in numpy.linspace(0,2,(2/deltaNoise)+1):
      noises+=[thisNoise]*nReps
   rKs,rDsK = doVariableNoise( noises, onePhaseV )[1:3]
   rDsR = doVariableNoise( noises,
         numpy.random.normal(0,onePhaseV.var()**0.5,onePhaseV.shape[0]) )[2]
   rDsK/=onePhaseV.var()
   rDsR/=onePhaseV.var()
   rDsK={
         'm':rDsK.reshape([ -1,(2/deltaNoise)+1, nReps ]).mean(axis=-1),
         'v':rDsK.reshape([ -1,(2/deltaNoise)+1, nReps ]).var(axis=-1),
      }
   rDsR={
         'm':rDsR.reshape([ -1,(2/deltaNoise)+1, nReps ]).mean(axis=-1),
         'v':rDsR.reshape([ -1,(2/deltaNoise)+1, nReps ]).var(axis=-1),
      }
   import pylab
   pylab.figure(1)
   pylab.subplot(2,2,1)
   pylab.imshow( rDsK['m'], vmin=0, vmax=1.0,
         extent=[-deltaNoise/2.,max(noises)+deltaNoise/2.,-0.5,len(rKs)-0.5] )
   pylab.subplot(2,2,2)
   pylab.imshow( rDsR['m'], vmin=0, vmax=1.0,
         extent=[-deltaNoise/2.,max(noises)+deltaNoise/2.,-0.5,len(rKs)-0.5] )
   pylab.subplot(2,1,2)
   cols=['b','g','r','m','k','c']
   for i,k in rKs:
      pylab.errorbar( noises[::nReps], rDsK['m'][i], yerr=rDsK['v'][i]**0.5,
            label=k, linestyle='-', color=cols[i] )
      pylab.errorbar( noises[::nReps], rDsR['m'][i], yerr=rDsR['v'][i]**0.5,
            linestyle='--', color=cols[i] )

   pylab.legend(loc=0)


####################################
## Main code logic begins now
##
##

title="Output variables"
opVarNames=('r0','L0','nfft','noise','rcond','alpha','beta','gamma','gammaN')
opValPairs=[]
for opVarName in opVarNames:
   if opVarName not in dir():
      opValPairs.append((opVarName,"*NOT FOUND*"))
   else:
      opValPairs.append( (opVarName+" = ",str( eval(opVarName) )) )

print(" +"+title+"+"*(40-len(title)))
for opName,val in opValPairs:
   print(" "*(max([40-len(opName),0])) + opName + "{0:s}".format(val))
#
print(" "+"+"*40)

print("[Setup]",end="") ; sys.stdout.flush()
( pupilMask,gO,gM,gTgM,waffleV,lO,lM )=setupGeometry(nfft,obscuration)
( covPlus, cov_I, onePhase, onePhaseV, onePhaseD )=setupPhase(r0,L0)
### onePhaseV=numpy.random.normal(0,onePhaseV.var()**0.5,len(onePhaseV))
onePhaseV = numpy.where(
      numpy.random.uniform(-1,1,size=gO.numberPhases)>0, 1, -1 )
onePhaseD.ravel()[gO.illuminatedCornersIdx]=onePhaseV

print("[Input]",end="") ; sys.stdout.flush()
gradV=numpy.dot(gM,onePhaseV)
if 'noise' in dir():
   gradV+=numpy.random.normal(0,gradV.var()**0.5*noise,size=gO.numberSubaps*2)
   print("(Adding noise)",end="")

print("[Reconstructors]",end="") ; sys.stdout.flush()
invgTgMs={
      'alpha':doInversion(alpha,0,0,1), # Tikhonov
      'beta': doInversion(0,beta,0,1), # Covariance approximation, least squares
      'beta2': doInversion(0,beta,0,gammaN), # Covariance approximation, min var 
      'gamma':doInversion(0,0,gamma,gammaN), # Minimum variance
      'svd':  doSVD(rcond)
   }

print("Reconstruction::",end="") ; sys.stdout.flush()
reconM,reconPhaseV,reconPhaseD={},{},{}
for invType in invgTgMs.keys():
   print("<{0:s}:".format(invType),end="") ; sys.stdout.flush()
   if not invgTgMs[invType] is None:
      print("MVM:".format(invType),end="") ; sys.stdout.flush()
      reconM[invType] = numpy.dot( invgTgMs[invType], gM.transpose() ) 
      if invType in ('beta2','gamma'): reconM[invType]*=gammaN
      #
      print("recon:".format(invType),end="") ; sys.stdout.flush()
      reconPhaseV[invType] = numpy.dot(reconM[invType], gradV)
      #
      print("reimage>".format(invType),end="") ; sys.stdout.flush()
      thisPhaseD=numpy.zeros((nfft+1)**2,numpy.float64)
      thisPhaseD[gO.illuminatedCornersIdx]=reconPhaseV[invType]
      reconPhaseD[invType] = numpy.ma.masked_array(
            thisPhaseD.reshape([nfft+1]*2), [gO.illuminatedCorners==0] )
   else: 
      print("*N/A*>".format(invType),end="") ; sys.stdout.flush()
      reconM[invType],reconPhaseV[invType],reconPhaseD = [None]*3

print()

title="STREHLS"
print(" +"+title+"+"*(40-len(title)))
opValPairs=[]
for k in invgTgMs.keys():
   opValPairs.append(( k+" -> ","{0:04.1f}%".format(
         abs( numpy.exp(1.0j*(onePhaseD-reconPhaseD[k])).sum()
            )**2.*len(onePhaseV)**-2.0*100
            )
         ))
for opName,val in opValPairs:
   print(" "*(max([40-len(opName),0])) + opName + "{0:s}".format(val))

printOutputVariances()
printTimings()
print("For plots, use subroutine doPlots()")
print("For analysis of variable noise, use subroutine  "+
         "doVariableNoiseComparison()")
