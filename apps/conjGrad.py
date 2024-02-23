from __future__ import print_function
import time
# What is this?
# Test conjugate gradient algorithm

#if __name__=="__main__":
#   # call main
#   main()
timings={}

timings['s:Modules']=time.time()
import matplotlib.pyplot as pylab
import numpy
import gradientOperator
import commonSeed 
import sys
timings['e:Modules']=time.time()
nfft=200# pixel size
r0=nfft/10.0 # in pixels
L0=nfft
sparsity=(nfft**2.0*numpy.pi/4.0)>=1000

if sparsity:
   print("WARNING: This might take a while, switched to sparse matrices...")
   sys.stdout.flush()


# define pupil mask as sub-apertures
pupilMask=numpy.ones([nfft]*2,numpy.int32)
pupilCds=numpy.add.outer(
   (numpy.arange(nfft)-(nfft-1)/2.)**2.0,
   (numpy.arange(nfft)-(nfft-1)/2.)**2.0 )
pupilMask=(pupilCds>(nfft/6)**2)*(pupilCds<(nfft/2)**2)

   # \/ use Fried geometry
timings['s:gO']=time.time() ; print("s:gO",end="") ; sys.stdout.flush()
gO=gradientOperator.gradientOperatorType1(pupilMask,sparse=sparsity)
gM=gO.returnOp()
timings['e:gO']=time.time() ; print("(done)") ; sys.stdout.flush()

# define phase at corners of pixels, each of which is a sub-aperture
timings['s:pC']=time.time() ; print("s:pC",end="") ; sys.stdout.flush()
#import phaseCovariance as pc
#cov=pc.covarianceMatrixFillInRegular(
#   pc.covarianceDirectRegular(nfft+2,r0,L0) )
#choleskyC=pc.choleskyDecomp(cov)
timings['e:pC']=time.time() ; print("(done)") ; sys.stdout.flush()

# generate phase and gradients
timings['s:pvC']=time.time() ; print("s:pvC",end="") ; sys.stdout.flush()
#onePhase=numpy.dot( choleskyC, numpy.random.normal(size=(nfft+2)**2 ) )
import kolmogorov
onePhase=kolmogorov.TwoScreens(nfft*4,r0,flattening=2*numpy.pi/L0)[0][:nfft+2,:nfft+2]#.ravel()
timings['e:pvC']=time.time() ; print("(done)") ; sys.stdout.flush()

   # \/ for comparison purposes onePhase is too big, so take a mean
# Nota Bene, When using the FFT method for input generation, can
# just apply a simply mean process
onePhase=\
   (onePhase[1:,1:]+onePhase[:-1,1:]+onePhase[:-1,:-1]+onePhase[1:,:-1])/4.0
#?meanO=numpy.zeros([(nfft+1)**2,(nfft+2)**2], numpy.float64)
#?for i in range((nfft+1)**2):
#?   meanO[i,i//(nfft+1)*(nfft+2)+i%(nfft+1)]=0.25
#?   meanO[i,i//(nfft+1)*(nfft+2)+i%(nfft+1)+1]=0.25
#?   meanO[i,(i//(nfft+1)+1)*(nfft+2)+i%(nfft+1)]=0.25
#?   meanO[i,(i//(nfft+1)+1)*(nfft+2)+i%(nfft+1)+1]=0.25
#?onePhase=numpy.dot(meanO,onePhase)
onePhase=onePhase.ravel()

onePhaseV=onePhase[gO.illuminatedCornersIdx]
onePhase-=onePhaseV.mean() # normalise
onePhase.resize([nfft+1]*2)
onePhaseD=numpy.ma.masked_array(onePhase,gO.illuminatedCorners==0)

timings['s:gvC']=time.time() ; print("s:gvC",end="") ; sys.stdout.flush()
gradV=gM.dot(onePhaseV) # return an array
timings['e:gvC']=time.time() ; print("(done)") ; sys.stdout.flush()

# now try solving
# linear least squares is (G^T G+alpha I+beta R)^-1 G^T
# alpha!=0 with Tikhonov regularisation
lO=gradientOperator.laplacianOperatorType1(pupilMask,sparse=sparsity)
lM=lO.returnOp() # laplacian operator

   # \/ Tikhonov reconstruction
timings['s:gTg']=time.time() ; print("s:gTg",end="") ; sys.stdout.flush()
gTg_=numpy.dot( gM.transpose(), gM )
timings['e:gTg']=time.time() ; print("(done)") ; sys.stdout.flush()
#> timings['s:ExpliciteigV']=time.time()
#> eigV=numpy.linalg.eigvals(gTg_) ; eigV.sort()
#> eigVEst=eigV[-1]
#> timings['e:ExpliciteigV']=time.time()
   # \/ power method for largest eigenvalue
timings['s:eigV']=time.time() ; print("s:eigV",end="") ; sys.stdout.flush()
vecLen=gTg_.shape[0]
eigVEst=numpy.random.uniform(size=vecLen)
eigEstV=numpy.dot(eigVEst,eigVEst)**0.5
relChangeEigV=1
iterNum=1
while relChangeEigV>0.01:
   eigVEst=gTg_.dot(eigVEst) # iterate, will return an array
   oldEigEstV=eigEstV
   eigEstV=numpy.dot(eigVEst,eigVEst)**0.5
   eigVEst/=eigEstV
   relChangeEigV=abs(oldEigEstV-eigEstV)/abs(eigEstV)
   iterNum+=1 ; print(".",end="") ; sys.stdout.flush()
timings['e:eigV']=time.time() ; print("(done)") ; sys.stdout.flush()
alpha=eigEstV**0.5*1e-4 # quash about largest eigenvalue
beta=alpha

from scipy.sparse import issparse,identity as spidentity
if issparse( gTg_ ):
   gTg_=gTg_+alpha*spidentity(gO.numberPhases)\
      +beta*numpy.dot(lM,lM)
else:
   gTg_+=alpha*numpy.identity(gO.numberPhases)+beta*numpy.dot(lM,lM)

timings['s:CGp']=time.time() ; print("s:CGp",end="") ; sys.stdout.flush()
A=gTg_ ; b=gM.transpose().dot( gradV )
r=[b] ; k=0 ; x=[b*0] ; p=[None,b] ; nu=[None]
timings['e:CGp']=time.time() ; print("(done)") ; sys.stdout.flush()


import sys
rNorm=1
timings['s:CGl']=time.time() ; print("s:CGl",end="") ; sys.stdout.flush()
from scipy.sparse.linalg import cg as spcg
class C:
   x=[]
   def callback(self,rk):
      try:
         self.x.append(rk.copy()) 
      except AttributeError:
         self.x.append(rk+0.0)
thisC=C()
timings['s:CGl']=time.time() ; print("s:gO",end="") ; sys.stdout.flush()
spcg( gTg_, b, maxiter=1e3, tol=1e-3, callback=thisC.callback  )
x=numpy.array(thisC.x) # make a copy
timings['e:CGl']=time.time() ; print("(done)") ; sys.stdout.flush()

#>k=0
#> while rNorm>1e-5 and k<1000:
#>    k+=1
#>    z=A.dot( p[k] )
#>    nu_k=(r[k-1]**2.0).sum()/(p[k]*z).sum()
#>    nu.append( nu_k )
#>    x.append( x[k-1]+nu[k]*p[k] )
#>    r.append( r[k-1]-nu[k]*z )
#>    mu_k_1= (r[k]**2.0).sum()/(r[k-1]**2.0).sum()
#>    p.append( r[k]+mu_k_1*p[k] )
#>    rNorm=(r[-1]**2.0).sum()
#> timings['e:CGl']=time.time()

print("INFO: CG # iters={0:d}".format(len(thisC.x)))

if not issparse(gTg_):
   # trad inv matrices
   timings['s:inv']=time.time() ; print("s:inv",end="") ; sys.stdout.flush()
   A_dagger=numpy.linalg.pinv( gTg_ )
   timings['e:inv']=time.time() ; print("(done)") ; sys.stdout.flush()
   timings['s:dp']=time.time() ; print("s:dp",end="") ; sys.stdout.flush()
   x_inv=numpy.dot( A_dagger, b )
   timings['e:dp']=time.time() ; print("(done)") ; sys.stdout.flush()
      
else:
   from scipy.sparse.linalg import gmres as spgmres
   timings['s:inv']=time.time()
   timings['e:inv']=time.time()
   k=1
   thisC.x=[numpy.zeros(gO.numberPhases)]
   x_inv=(None,1)
   while k<1000 and x_inv[1]!=0:
      timings['s:dp']=time.time() ; print("s:dp",end="") ; sys.stdout.flush()
      x_inv=spgmres( gTg_, b, maxiter=k, tol=1e-3/onePhaseV.var() ) 
      timings['e:dp']=time.time() ; print("(done)") ; sys.stdout.flush()
      thisC.x.append( x_inv[0] )
      k+=1
   print("INFO: GMRES # iters={0:d}".format(len(thisC.x)))
   if x_inv[1]!=0:
      print (" **WARNING** :-  GMRES didn't converge. Nulling result.")
      x_inv=numpy.zeros(gO.numberPhases)
   else:
      x_inv=x_inv[0]

reconPhaseV=[ x[-1], x_inv ]

# imaging of phases0
reconPhaseD=numpy.zeros([2,(nfft+1)**2])
reconPhaseD[:,gO.illuminatedCornersIdx]=reconPhaseV
reconPhaseD=numpy.ma.masked_array(
   reconPhaseD.reshape([2]+[nfft+1]*2), [gO.illuminatedCorners==0]*2 )

pylab.figure(1)
pylab.subplot(231)
pylab.imshow( onePhaseD, interpolation='nearest', origin='lower',
   extent=[-0.5,nfft+0.5,-0.5,nfft+0.5] )
pylab.title("type 1: input phase")
pylab.colorbar()
pylab.subplot(232)
pylab.imshow( reconPhaseD[1], interpolation='nearest', origin='lower',
   extent=[-0.5,nfft+0.5,-0.5,nfft+0.5] )
pylab.title("recon, inv")
pylab.colorbar()
pylab.subplot(233)
pylab.imshow( reconPhaseD[0], interpolation='nearest', origin='lower',
   extent=[-0.5,nfft+0.5,-0.5,nfft+0.5] )
pylab.title("recon, CG")
pylab.colorbar()
pylab.subplot(223)
pylab.imshow( reconPhaseD[1]-onePhaseD, interpolation='nearest', origin='lower',
   extent=[-0.5,nfft+0.5,-0.5,nfft+0.5] )
pylab.title("diff (inv-input)")
pylab.colorbar()
pylab.subplot(224)
pylab.imshow( reconPhaseD[0]-onePhaseD, interpolation='nearest', origin='lower',
   extent=[-0.5,nfft+0.5,-0.5,nfft+0.5] )
pylab.title("diff (inv-CG)")
pylab.colorbar()

# remnant variances
print("RES: input var=",onePhaseD.var())
print("RES: input-recon (inv) var=\t",(reconPhaseD[1]-onePhaseD).var())
print("RES: input-recon (CG) var=\t",(reconPhaseD[0]-onePhaseD).var())

# waffle operator
if not issparse(gTg_):
   waffleO=gradientOperator.waffleOperatorType1(pupilMask)
   waffleV=waffleO.returnOp()
   print("RES: waffle input amp=\t",numpy.dot(onePhaseV, waffleV))
   print("RES: waffle recon (inv) amp=\t",numpy.dot(reconPhaseV[1], waffleV))
   print("RES: waffle recon (CG) amp=\t",numpy.dot(reconPhaseV[0], waffleV))
else:
   print("INFO: no waffle computations")

# timings
print("-"*10)
for dat in (
      ("Phase creation","pC"), ("CG loop time","CGl"),
      ("Inv time","inv"), ("Eig Values time","eigV"),
      ("Phase vec. time","pvC"), ("Gradient vec. time","gvC"),
      ("Module loading","Modules"),
      ("MVM"*(not sparsity)+"(scipy) GMRES"*(sparsity),"dp")
     ):
   print("TIME: {1}={0:5.3f}s".format(
      timings['e:'+dat[1]]-timings['s:'+dat[1]],dat[0]))


# plot CG iteration residual variance
pylab.figure()
pylab.semilogy( [ (x[i]-x_inv).var()/onePhaseV.var() for i in range(len(x)) ],
      '.-',label="$\mathrm{var}(x[i]-x_{inv})$" )
pylab.semilogy( [ (x[i]-onePhaseV).var()/onePhaseV.var() for i in range(len(x)) ],
      '.-',label="$\mathrm{var}(x[i]-ip)$" )
if sparsity:
   pylab.semilogy( [ (thisC.x[i]-onePhaseV).var()/onePhaseV.var()
      for i in range(len(thisC.x)) ], 'r.-',label="GMRES,$\mathrm{var}(x[i]-ip)$" )
pylab.plot( [0,len(x)-1], [(x_inv-onePhaseV).var()/onePhaseV.var()]*2,
      'k--',label="$\mathrm{var}(x_{inv}-ip)$" )
pylab.legend(loc=0)
pylab.title(sys.argv[0]+": variance, CG iterrations")
pylab.xlabel("CG iterrations")
pylab.ylabel("Residual variance")
