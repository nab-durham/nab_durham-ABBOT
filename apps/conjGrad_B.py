from __future__ import print_function
import time
# What is this?
# Test conjugate gradient algorithm, using an alternative description
# for linear-least squares and alternative matrix inversion (SVD)

#if __name__=="__main__":
#   # call main
#   main()
timings={}

timings['s:Modules']=time.time()
import matplotlib.pyplot as pg
import numpy
import gradientOperator
import sys
import commonSeed 
import phaseCovariance as pc

timings['e:Modules']=time.time()
nfft=32 # pixel size
r0=4#nfft/10.0 # in pixels
L0=nfft


# define pupil mask as sub-apertures
pupilMask=numpy.ones([nfft]*2,numpy.int32)
pupilCds=numpy.add.outer(
   (numpy.arange(nfft)-(nfft-1)/2.)**2.0,
   (numpy.arange(nfft)-(nfft-1)/2.)**2.0 )
pupilMask=(pupilCds>(nfft/6)**2)*(pupilCds<=(nfft/2)**2)

   # \/ use Fried geometry
timings['s:gO']=time.time()
gO=gradientOperator.gradientOperatorType1(pupilMask)
gM=gO.returnOp()
timings['e:gO']=time.time()

# define phase at corners of pixels, each of which is a sub-aperture
timings['s:pC']=time.time()
cov=pc.covarianceMatrixFillInRegular(
   pc.covarianceDirectRegular(nfft+2,r0,L0) )
choleskyC=pc.choleskyDecomp(cov)
timings['e:pC']=time.time()

# generate phase and gradients
timings['s:pvC']=time.time()
onePhase=numpy.dot( choleskyC, numpy.random.normal(size=(nfft+2)**2 ) )
timings['e:pvC']=time.time()

   # \/ for comparison purposes onePhase is too big, so take a mean
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

timings['s:gvC']=time.time()
gradV=numpy.dot(gM,onePhaseV)
timings['e:gvC']=time.time()

# now try solving
# conjugate gradient with linear least squares 

timings['s:CGp']=time.time()
A=gM ; b=onePhaseV*0 ; AT=A.T
r=[numpy.dot(AT,gradV)] ; k=0 ; x=[b*0] ; p=[None,r[0]] ; nu=[None]
timings['e:CGp']=time.time()


import sys
rNorm=1
timings['s:CGl']=time.time()
while rNorm>1e-5 and k<100:
   k+=1
   z=numpy.dot(A,p[k])
   nu_k=(r[k-1]**2.0).sum()/(z**2).sum()
   nu.append( nu_k )
   x.append( x[k-1]+nu[k]*p[k] )
   r.append( r[k-1]-nu[k]*numpy.dot(AT,z) )
   mu_k_1= (r[k]**2.0).sum()/(r[k-1]**2.0).sum()
   p.append( r[k]+mu_k_1*p[k] )
   rNorm=(r[-1]**2.0).sum()
timings['e:CGl']=time.time()

print("CG took {0:d} iterrations".format(k))

# trad inv matrices
timings['s:inv']=time.time()
A_dagger=numpy.linalg.pinv( gM, 1e-2)
timings['e:inv']=time.time()
timings['s:dp']=time.time()
x_inv=numpy.dot( A_dagger, gradV )
timings['e:dp']=time.time()
      
reconPhaseV=[ x[-1], x_inv ]

# imaging of phases
reconPhaseD=numpy.zeros([2,(nfft+1)**2])
reconPhaseD[:,gO.illuminatedCornersIdx]=[x[-1],x_inv]
reconPhaseD=numpy.ma.masked_array(
   reconPhaseD.reshape([2]+[nfft+1]*2), [gO.illuminatedCorners==0]*2 )

pg.figure(1)
pg.subplot(231)
pg.imshow( onePhaseD, interpolation='nearest', origin='lower',
   extent=[-0.5,nfft+0.5,-0.5,nfft+0.5] )
pg.title("type 1: input phase")
pg.colorbar()
pg.subplot(232)
pg.imshow( reconPhaseD[1], interpolation='nearest', origin='lower',
   extent=[-0.5,nfft+0.5,-0.5,nfft+0.5] )
pg.title("recon, inv")
pg.colorbar()
pg.subplot(233)
pg.imshow( reconPhaseD[0], interpolation='nearest', origin='lower',
   extent=[-0.5,nfft+0.5,-0.5,nfft+0.5] )
pg.title("recon, CG")
pg.colorbar()
pg.subplot(223)
pg.imshow( reconPhaseD[1]-onePhaseD, interpolation='nearest', origin='lower',
   extent=[-0.5,nfft+0.5,-0.5,nfft+0.5] )
pg.title("diff (inv-input)")
pg.colorbar()
pg.subplot(224)
pg.imshow( reconPhaseD[0]-onePhaseD, interpolation='nearest', origin='lower',
   extent=[-0.5,nfft+0.5,-0.5,nfft+0.5] )
pg.title("diff (inv-CG)")
pg.colorbar()

# remnant variances
print("input var=",onePhaseD.var())
print("input-recon (inv) var=\t",(reconPhaseD[1]-onePhaseD).var())
print("input-recon (CG) var=\t",(reconPhaseD[0]-onePhaseD).var())

# waffle operator
waffleO=gradientOperator.waffleOperatorType1(pupilMask)
waffleV=waffleO.returnOp()
print("waffle input amp=\t",numpy.dot(onePhaseV, waffleV))
print("waffle recon (inv) amp=\t",numpy.dot(reconPhaseV[1], waffleV))
print("waffle recon (CG) amp=\t",numpy.dot(reconPhaseV[0], waffleV))

# timings
print("-"*10)
for dat in (
      ("Phase creation","pC"), ("CG loop time","CGl"),
      ("Inv time","inv"),
      ("Phase vec. time","pvC"), ("Gradient vec. time","gvC"),
      ("Module loading","Modules"), ("MVM","dp")
     ):
   print("{1}={0:5.3f}s".format(
      timings['e:'+dat[1]]-timings['s:'+dat[1]],dat[0]))


# plot CG iteration residual variance
pg.figure()
pg.semilogy( [ (x[i]-x_inv).var()/onePhaseV.var() for i in range(len(x)) ],
      '.-',label="$\mathrm{var}(x[i]-x_{inv})$" )
pg.semilogy( [ (x[i]-onePhaseV).var()/onePhaseV.var() for i in range(len(x)) ],
      '.-',label="$\mathrm{var}(x[i]-ip)$" )
pg.plot( [0,len(x)-1], [(x_inv-onePhaseV).var()/onePhaseV.var()]*2,
      'k--',label="$\mathrm{var}(x_{inv}-ip)$" )
pg.legend(loc=0)
pg.title(sys.argv[0]+": variance, CG iterrations")
pg.xlabel("CG iterrations")
pg.ylabel("Residual variance")
