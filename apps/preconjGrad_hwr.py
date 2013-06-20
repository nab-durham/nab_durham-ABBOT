from __future__ import print_function
import time
# What is this?
# Test conjugate gradient algorithm, using hwr to first integrate
# up the gradient matrix and the gradient vector

#if __name__=="__main__":
#   # call main
#   main()
timings={}

timings['s:Modules']=time.time()
import matplotlib.pyplot as pylab
import numpy
import gradientOperator
import hwr
import sys
import commonSeed 
import phaseCovariance as pc

timings['e:Modules']=time.time()
nfft=16 # pixel size
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
timings['s:prepC']=time.time()
chainsDef,chainsDefChStrts,chainsMap,offsetEstM=hwr.prepCure(
      gO,boundary=[4]*2,overlapType=0)
timings['e:prepC']=time.time()
timings['s:igM']=time.time()
igM=numpy.zeros( [gO.numberPhases]*2, numpy.float64 )
#?meants=0
for i in range(gM.shape[1]):
#?   zzz,tigM,ts=hwr.doCureOnePoke(
   (zzz,tigM)=hwr.doCureOnePoke(
         gM[:,i], chainsDef,gO,offsetEstM,chainsDefChStrts,chainsMap,i )
   igM[:,i]=tigM
#?   meants+=ts
   if (i%100)==0:
      print(".",end="") ; sys.stdout.flush()
      print("(",end="");
#?      for j in meants: print("{0:5.3f}".format(j),end=" ")
      print(")",end="");
#?      meants=0
igM=numpy.array(igM).T
timings['e:igM']=time.time()
# define phase at corners of pixels, each of which is a sub-aperture
timings['s:pC']=time.time()
#? cov=pc.covarianceMatrixFillInRegular(
#?    pc.covarianceDirectRegular(nfft+2,r0,L0) )
#? choleskyC=pc.choleskyDecomp(cov)
timings['e:pC']=time.time()

#? # generate phase and gradients
timings['s:pvC']=time.time()
#? onePhase=numpy.dot( choleskyC, numpy.random.normal(size=(nfft+2)**2 ) )
timings['e:pvC']=time.time()
#? 
#?    # \/ for comparison purposes onePhase is too big, so take a mean
#? meanO=numpy.zeros([(nfft+1)**2,(nfft+2)**2], numpy.float64)
#? for i in range((nfft+1)**2):
#?    meanO[i,i//(nfft+1)*(nfft+2)+i%(nfft+1)]=0.25
#?    meanO[i,i//(nfft+1)*(nfft+2)+i%(nfft+1)+1]=0.25
#?    meanO[i,(i//(nfft+1)+1)*(nfft+2)+i%(nfft+1)]=0.25
#?    meanO[i,(i//(nfft+1)+1)*(nfft+2)+i%(nfft+1)+1]=0.25
#? onePhase=numpy.dot(meanO,onePhase)
import kolmogorov
onePhase=kolmogorov.TwoScreens(nfft*4,r0,flattening=2*numpy.pi/L0)[0][:nfft+2,:nfft+2]#.ravel()
onePhase=\
   (onePhase[1:,1:]+onePhase[:-1,1:]+onePhase[:-1,:-1]+onePhase[1:,:-1])/4.0
onePhase=onePhase.ravel()

onePhaseV=onePhase[gO.illuminatedCornersIdx]
onePhase-=onePhaseV.mean() # normalise
onePhase.resize([nfft+1]*2)
onePhaseD=numpy.ma.masked_array(onePhase,gO.illuminatedCorners==0)

timings['s:gvC']=time.time()
gradV=numpy.dot(gM,onePhaseV)
gradV+=numpy.random.normal(0,0.5*gradV.var()**0.5,size=len(gradV))
timings['e:gvC']=time.time()
timings['s:curegv']=time.time()
igradV=hwr.doCureGeneral(
         gradV, chainsDef,gO,offsetEstM,chainsDefChStrts,chainsMap )[1]
timings['e:curegv']=time.time()

# now try solving
# linear least squares is (G^T G+alpha I+beta R)^-1 G^T
# alpha!=0 with Tikhonov regularisation

lO=gradientOperator.laplacianOperatorType1(pupilMask)
lM=lO.returnOp() # laplacian operator

   # \/ Tikhonov reconstruction
timings['s:igTig']=time.time()
igTig_=numpy.dot( igM.transpose(), igM )
timings['e:igTig']=time.time()
timings['s:gTg']=time.time()
gTg_=numpy.dot( gM.transpose(), gM )
timings['e:gTg']=time.time()

timings['s:eigV']=time.time()
eigV=abs(numpy.linalg.eigvals(igTig_)) ; eigV.sort()
eigEstV=eigV[-1]
timings['e:eigV']=time.time()
timings['s:eigV_b']=time.time()
eigV=abs(numpy.linalg.eigvals(gTg_)) ; eigV.sort()
eigEstV_b=eigV[-1]
timings['e:eigV_b']=time.time()

#>   # \/ power method for largest eigenvalue
#>timings['s:eigV']=time.time()
#>vecLen=igTig_.shape[0]
#>eigVEst=numpy.random.uniform(size=vecLen)
#>eigEstV=numpy.dot(eigVEst,eigVEst)**0.5
#>relChangeEigV=1
#>iterNum=1
#>while relChangeEigV>0.01:
#>   eigVEst=numpy.dot(igTig_,eigVEst) # iterate
#>   oldEigEstV=eigEstV
#>   eigEstV=numpy.dot(eigVEst,eigVEst)**0.5
#>   eigVEst/=eigEstV
#>   relChangeEigV=abs(oldEigEstV-eigEstV)/abs(eigEstV)
#>   iterNum+=1
#>timings['e:eigV']=time.time()
alpha=eigEstV*1e-3 # quash about largest eigenvalue
alpha_b=eigEstV_b*1e-3 # quash about largest eigenvalue
beta=alpha**2.0
igTig_+=alpha*numpy.identity(gO.numberPhases)
gTg_+=alpha_b*numpy.dot(lM.T,lM)

A_orig=gTg_ ; b_orig=numpy.dot( gM.transpose(), gradV )
import sys
rNorm_tol=1e-2

#?? # >>> CG with modified matrices
#?? timings['s:CGp']=time.time()
#?? A=igTig_ ; b=numpy.dot( igM.transpose(), igradV )
#?? r=[b] ; k=0 ; x=[b*0] ; p=[None,b] ; nu=[None]
#?? timings['e:CGp']=time.time()
#?? 
#?? 
#?? rNorm=1
#?? timings['s:CGl']=time.time()
#?? while rNorm>rNorm_tol and k<100:
#??    k+=1
#??    z=numpy.dot(A,p[k])
#??    nu_k=(r[k-1]**2.0).sum()/(p[k]*z).sum()
#??    nu.append( nu_k )
#??    x.append( x[k-1]+nu[k]*p[k] )
#??    r.append( r[k-1]-nu[k]*z )
#??    mu_k_1= (r[k]**2.0).sum()/(r[k-1]**2.0).sum()
#??    p.append( r[k]+mu_k_1*p[k] )
#??    rNorm=(r[-1]**2.0).sum()
#?? timings['e:CGl']=time.time()
#?? print("CG took {0:d} iterrations".format(k))
#?? 
#?? # >>> reset and do CG on original matrices
#?? r=[b_orig] ; k=0 ; x_orig=[b_orig*0] ; p=[None,b_orig] ; nu=[None]
#?? rNorm=1
#?? timings['s:oCGl']=time.time()
#?? while rNorm>rNorm_tol and k<100:
#??    k+=1
#??    z=numpy.dot(A_orig,p[k])
#??    nu_k=(r[k-1]**2.0).sum()/(p[k]*z).sum()
#??    nu.append( nu_k )
#??    x_orig.append( x_orig[k-1]+nu[k]*p[k] )
#??    r.append( r[k-1]-nu[k]*z )
#??    mu_k_1= (r[k]**2.0).sum()/(r[k-1]**2.0).sum()
#??    p.append( r[k]+mu_k_1*p[k] )
#??    rNorm=(r[-1]**2.0).sum()
#?? timings['e:oCGl']=time.time()
#?? print("(original CG took {0:d} iterrations)".format(k))
#?? 
#?? factor=0.75
#?? x_st=[ factor*hwr.doCureGeneral(
#??          gradV, chainsDef,gO,offsetEstM,chainsDefChStrts,chainsMap )[1] ]
#?? r=[b_orig-A_orig.dot(x_st[0])] ; k=0 ; p=[None,r[0]] ; nu=[None]
#?? # >>> reset and do CG on original matrices, with hwr estimation as start
#?? rNorm=1
#?? timings['s:o+CGl']=time.time()
#?? while rNorm>rNorm_tol and k<100:
#??    k+=1
#??    z=numpy.dot(A_orig,p[k])
#??    nu_k=(r[k-1]**2.0).sum()/(p[k]*z).sum()
#??    nu.append( nu_k )
#??    x_st.append( x_st[k-1]+nu[k]*p[k] )
#??    r.append( r[k-1]-nu[k]*z )
#??    mu_k_1= (r[k]**2.0).sum()/(r[k-1]**2.0).sum()
#??    p.append( r[k]+mu_k_1*p[k] )
#??    rNorm=(r[-1]**2.0).sum()
#?? timings['e:o+CGl']=time.time()
#?? print("(original CG + hwr/hwr took {0:d} iterrations)".format(k))

# > build a mask and then use this to reduce the complexity of gTg, invert
# > that and it becomes a preconditioner
domainSize=nfft/4.0
maskS=( ((gO.subapMaskIdx//(gO.n[1]*domainSize)*domainSize)*gO.n_[1]), 
       ((gO.subapMaskIdx%gO.n[1])//domainSize)*domainSize )
maskSR=[ ((gO.illuminatedCornersIdx)>=maskS[0][i])
       *((gO.illuminatedCornersIdx)<=(maskS[0][i]+
         (((maskS[0][i]==maskS[0][-1]) +domainSize)*gO.n_[1]))) 
#      * ((gO.illuminatedCornersIdx%gO.n_[1])>=maskS[1][i])
#       *((gO.illuminatedCornersIdx%gO.n_[1])<(maskS[1][i]+domainSize))
         for i in range(gO.numberSubaps) ]*2
maskSR=numpy.array(maskSR)
#
maskP=( (gO.illuminatedCornersIdx//(gO.n_[1]*domainSize)*domainSize*gO.n_[1]), 
       ((gO.illuminatedCornersIdx%gO.n_[1])//domainSize)*domainSize )
maskPR=[ ((gO.illuminatedCornersIdx)>=maskP[0][i])
       *((gO.illuminatedCornersIdx)<=(maskP[0][i]+
         (((maskP[0][i]==maskP[0][-1]) +domainSize)*gO.n_[1])))
#      * ((gO.illuminatedCornersIdx%gO.n_[1])>=maskP[1][i])
#       *((gO.illuminatedCornersIdx%gO.n_[1])<(maskP[1][i]+domainSize))
         for i in range(gO.numberPhases) ]
maskPR=numpy.array(maskPR)
#
gMl=gM*maskSR
lMl=lM*maskPR
def enforceZero(ip): # concept here is that dot product with constant should be flat
   for i in range(ip.shape[0]):
     idx=ip[i].nonzero()[0]
     ip[i][idx]-=ip[i][idx].sum()*len(idx)**-1.0
   return(ip)

gMl=enforceZero(gMl)
lMl=enforceZero(lMl)
gTg_l=gMl.T.dot( gMl )
gTg_l+=alpha_b*lMl.T.dot(lMl)
M=numpy.linalg.inv(gTg_l)
gTg_li=numpy.linalg.inv(gTg_l)
gTg_i=numpy.linalg.inv(gTg_)
#
r=[b_orig] ; r_=[M.dot(r[0])] ; x_orig=[b_orig*0] ; p=[None,b_orig] ; nu=[None]
rNorm=1 ; k=0 
timings['s:o+CGl']=time.time()
while rNorm>rNorm_tol and k<100:
   k+=1
   z=numpy.dot(A_orig,p[k])
   nu_k=(r[k-1]**2.0).sum()/(p[k]*z).sum()
   nu.append( nu_k )
   x_st.append( x_st[k-1]+nu[k]*p[k] )
   r.append( r[k-1]-nu[k]*z )
   mu_k_1= (r[k]**2.0).sum()/(r[k-1]**2.0).sum()
   p.append( r[k]+mu_k_1*p[k] )
   rNorm=(r[-1]**2.0).sum()
timings['e:o+CGl']=time.time()
print("(original CG + hwr/hwr took {0:d} iterrations)".format(k))
# trad inv matrices
timings['s:inv']=time.time()
A_dagger=numpy.linalg.inv( gTg_ )
timings['e:inv']=time.time()
timings['s:dp']=time.time()
x_inv=numpy.dot( A_dagger, b_orig )
timings['e:dp']=time.time()
      
reconPhaseV=[ x[-1], x_inv, x_orig[-1], x_st[-1] ]

# imaging of phases
reconPhaseD=numpy.zeros([4,(nfft+1)**2])
reconPhaseD[:,gO.illuminatedCornersIdx]=reconPhaseV
reconPhaseD=numpy.ma.masked_array(
   reconPhaseD.reshape([4]+[nfft+1]*2), [gO.illuminatedCorners==0]*4 )

pylab.figure(1)
pylab.subplot(331)
pylab.imshow( onePhaseD, interpolation='nearest', origin='lower',
   extent=[-0.5,nfft+0.5,-0.5,nfft+0.5] )
pylab.title("type 1: input phase")
pylab.colorbar()
pylab.subplot(332)
pylab.imshow( reconPhaseD[1], interpolation='nearest', origin='lower',
   extent=[-0.5,nfft+0.5,-0.5,nfft+0.5] )
pylab.title("recon, inv")
pylab.colorbar()
pylab.subplot(333)
pylab.imshow( reconPhaseD[0], interpolation='nearest', origin='lower',
   extent=[-0.5,nfft+0.5,-0.5,nfft+0.5] )
pylab.title("recon, CG")
pylab.colorbar()
pylab.subplot(335)
pylab.imshow( reconPhaseD[2], interpolation='nearest', origin='lower',
   extent=[-0.5,nfft+0.5,-0.5,nfft+0.5] )
pylab.title("recon, oCG")
pylab.colorbar()
pylab.subplot(336)
pylab.imshow( reconPhaseD[3], interpolation='nearest', origin='lower',
   extent=[-0.5,nfft+0.5,-0.5,nfft+0.5] )
pylab.title("recon, oCG+d")
pylab.colorbar()
#
pylab.subplot(325)
pylab.imshow( reconPhaseD[1]-onePhaseD, interpolation='nearest', origin='lower',
   extent=[-0.5,nfft+0.5,-0.5,nfft+0.5] )
pylab.title("diff (inv-input)")
pylab.colorbar()
pylab.subplot(326)
pylab.imshow( reconPhaseD[0]-onePhaseD, interpolation='nearest', origin='lower',
   extent=[-0.5,nfft+0.5,-0.5,nfft+0.5] )
pylab.title("diff (inv-CG)")
pylab.colorbar()

# remnant variances
print("input var=",onePhaseD.var())
print("input-recon (inv) var=\t",(reconPhaseD[1]-onePhaseD).var())
print("input-recon (CG) var=\t",(reconPhaseD[0]-onePhaseD).var())
print("input-recon (orig CG) var=\t",(reconPhaseD[2]-onePhaseD).var())
print("input-recon (orig+hwr/hwr CG) var=\t",(reconPhaseD[3]-onePhaseD).var())

# waffle operator
waffleO=gradientOperator.waffleOperatorType1(pupilMask)
waffleV=waffleO.returnOp()
print("waffle input amp=\t",numpy.dot(onePhaseV, waffleV))
print("waffle recon (inv) amp=\t",numpy.dot(reconPhaseV[1], waffleV))
print("waffle recon (CG) amp=\t",numpy.dot(reconPhaseV[0], waffleV))

# timings
print("-"*10)
for dat in (
      ("Phase creation","pC"),
      ("CG loop time","CGl"),("orig CG loop time","oCGl"),
      ("orig+hwr/hwr CG loop time","o+CGl"),
      ("Inv time","inv"), ("Eig Values time","eigV"),
      ("Phase vec. time","pvC"), ("Gradient vec. time","gvC"),
      ("Module loading","Modules"), ("MVM","dp"), ("Prep cure","prepC"),
      ("integrated gM","igM"), ("integrated grad","curegv")
     ):
   print("{1}={0:5.3f}s".format(
      timings['e:'+dat[1]]-timings['s:'+dat[1]],dat[0]))


# plot CG iteration residual variance
pylab.figure()
pylab.semilogy( [ (x[i]-x_inv).var()/onePhaseV.var() for i in range(len(x)) ],
      '.-',label="$\mathrm{var}(x[i]-x_{inv})$" )
pylab.semilogy( [ (x[i]-onePhaseV).var()/onePhaseV.var() for i in range(len(x)) ],
      '.-',label="$\mathrm{var}(x[i]-ip)$" )
pylab.plot( [0,len(x)-1], [(x_inv-onePhaseV).var()/onePhaseV.var()]*2,
      'k--',label="$\mathrm{var}(x_{inv}-ip)$" )
pylab.legend(loc=0)
pylab.title(sys.argv[0]+": variance, CG iterrations")
pylab.xlabel("CG iterrations")
pylab.ylabel("Residual variance")

pylab.figure()
pylab.semilogy( [ (x_orig[i]-x_inv).var()/onePhaseV.var() for i in range(len(x_orig)) ],
      '.-',label="$\mathrm{var}(x[i]-x_{inv})$" )
pylab.semilogy( [ (x_orig[i]-onePhaseV).var()/onePhaseV.var() for i in range(len(x_orig)) ],
      '.-',label="$\mathrm{var}(x[i]-ip)$" )
pylab.plot( [0,len(x)-1], [(x_inv-onePhaseV).var()/onePhaseV.var()]*2,
      'k--',label="$\mathrm{var}(x_{inv}-ip)$" )
pylab.legend(loc=0)
pylab.title(sys.argv[0]+": variance, original CG iterrations")
pylab.xlabel("CG iterrations")
pylab.ylabel("Residual variance")

pylab.figure()
pylab.semilogy( [ (x_st[i]-x_inv).var()/onePhaseV.var() for i in range(len(x_st)) ],
      '.-',label="$\mathrm{var}(x[i]-x_{inv})$" )
pylab.semilogy( [ (x_st[i]-onePhaseV).var()/onePhaseV.var() for i in range(len(x_st)) ],
      '.-',label="$\mathrm{var}(x[i]-ip)$" )
pylab.plot( [0,len(x)-1], [(x_inv-onePhaseV).var()/onePhaseV.var()]*2,
      'k--',label="$\mathrm{var}(x_{inv}-ip)$" )
pylab.legend(loc=0)
pylab.title(sys.argv[0]+": variance, original +hwr/hwr CG iterrations")
pylab.xlabel("CG iterrations")
pylab.ylabel("Residual variance")
