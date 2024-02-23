from __future__ import print_function
import time
# What is this?
# Test conjugate gradient algorithm, using an alternative description
# for linear-least squares and alternative matrix inversion (SVD)
# Accelerate CG with HWR output

#if __name__=="__main__":
#   # call main
#   main()

timings={}

timings['s:Modules']=time.time()
import sys
import argparse
parser=argparse.ArgumentParser(
      description='conj grad algos for w/f recon inc. HWR accel' )
parser.add_argument('-N', help='N', type=int, default=10)
parser.add_argument('-r0', help='r0 [subaps]', type=int, default=4)
parser.add_argument('-L0', help='L0 [subaps]', type=int, default=(30/0.15)/4)
import numpy
import gradientOperator
import hwr
import os.path
args=parser.parse_args(sys.argv[1:])
import commonSeed 
import phaseCovariance as pc
timings['e:Modules']=time.time()


N=args.N # no. of sub-apertures
r0=args.r0#N/10.0 # in pixels
L0=args.L0
cacheFn="conjGrad_C_cache_N-{0:d}.pickle".format(N)

if os.path.exists(cacheFn):
   timings['s:CacheLoad']=time.time()
   data=numpy.load(cacheFn)
   cacheLoaded=True
   timings['e:CacheLoad']=time.time()
else:
   cacheLoaded=False

if cacheLoaded:
   print("**read cache "+cacheFn)

# define pupil mask as sub-apertures
pupilMask=numpy.ones([N]*2,numpy.int32)
pupilCds=numpy.add.outer(
   (numpy.arange(N)-(N-1)/2.)**2.0,
   (numpy.arange(N)-(N-1)/2.)**2.0 )
pupilMask=(pupilCds>(N/6)**2)*(pupilCds<=(N/2)**2)

   # \/ use Fried geometry
timings['s:gO']=time.time()
print("[grad op]") ; sys.stdout.flush()
gO=gradientOperator.gradientOperatorType1(pupilMask)
gM=gO.returnOp() if not cacheLoaded else data['gM']
timings['e:gO']=time.time()

# define phase at corners of pixels, each of which is a sub-aperture
timings['s:pC']=time.time()
print("[phase cov]") ; sys.stdout.flush()
if cacheLoaded:
   choleskyC=data['choleskyC']
else:
   cov=pc.covarianceMatrixFillInRegular(
      pc.covarianceDirectRegular(N+2,r0,L0) )
   choleskyC=pc.choleskyDecomp(cov)
timings['e:pC']=time.time()

# generate phase and gradients
timings['s:pvC']=time.time()
print("[mkphs]") ; sys.stdout.flush()
onePhase=numpy.dot( choleskyC, numpy.random.normal(size=(N+2)**2 ) )
timings['e:pvC']=time.time()

   # \/ for comparison purposes onePhase is too big, so take a mean
meanO=numpy.zeros([(N+1)**2,(N+2)**2], numpy.float64)
for i in range((N+1)**2):
   meanO[i,i//(N+1)*(N+2)+i%(N+1)]=0.25
   meanO[i,i//(N+1)*(N+2)+i%(N+1)+1]=0.25
   meanO[i,(i//(N+1)+1)*(N+2)+i%(N+1)]=0.25
   meanO[i,(i//(N+1)+1)*(N+2)+i%(N+1)+1]=0.25
onePhase=numpy.dot(meanO,onePhase)

onePhaseV=onePhase[gO.illuminatedCornersIdx]
onePhase-=onePhaseV.mean() # normalise
onePhaseV-=onePhaseV.mean()
phasesToVisualize=[ (onePhaseV,'input') ]
onePhase.resize([N+1]*2)
onePhaseD=numpy.ma.masked_array(onePhase,gO.illuminatedCorners==0)

timings['s:gvC']=time.time()
gradV=numpy.dot(gM,onePhaseV)
timings['e:gvC']=time.time()
# ---
# now try solving 
# ---
#

   # \/ conjugate gradient with linear least squares 
timings['s:CGp']=time.time()
A=gM ; AT=A.T
r=[numpy.dot(AT,gradV)] ; k=0 ; x=[numpy.zeros(len(onePhaseV))]
p=[None,r[0]] ; nu=[None]
timings['e:CGp']=time.time()


print("[CG]") ; sys.stdout.flush()
rNorm=1
timings['s:CGl']=time.time()
while rNorm>(len(r[-1])*1e3)**-1.0 and k<1000:
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
phasesToVisualize.append((x[-1],'CG'))

   # \/ CG with HWR
timings['s:HWRprep']=time.time()
print("[hwr prep]") ; sys.stdout.flush()
if not cacheLoaded:
   (smmtnsDef,smmtnsDefChStrts,smmtnsMap,offsetEstM)=\
      hwr.prepHWR(gO,boundary=None,overlapType=0.01,sparse=True,
            matrices=False,reglnVal=1e-2,laplacianRegln=0)
else:
   smmtnsDef=data['smmtnsDef']
   smmtnsDefChStrts=data['smmtnsDefChStrts']
   smmtnsMap=data['smmtnsMap']
   offsetEstM=data['offsetEstM']

timings['e:HWRprep']=time.time()
timings['s:HWRcalc']=time.time()
(comp,hwrV,(smmtnsV,smmtnsVOffsets))=\
   hwr.doHWRGeneral(gradV,smmtnsDef,gO,offsetEstM,smmtnsDefChStrts,smmtnsMap,
      doWaffleReduction=0,doPistonReduction=1,doGradientMgmnt=0,sparse=True)
timings['e:HWRcalc']=time.time()

timings['s:CGHWRp']=time.time()
A=gM ; AT=A.T
r=[numpy.dot(AT,gradV)-numpy.dot(numpy.dot(AT,A),hwrV)] ; k=0 ; x_hwr=[hwrV]
p=[None,r[0]] ; nu=[None]
timings['e:CGHWRp']=time.time()

rNorm=1
print("[cg]") ; sys.stdout.flush()
timings['s:CGHWRl']=time.time()
while rNorm>(len(r[-1])*1e3)**-1.0 and k<100:
   k+=1
   z=numpy.dot(A,p[k])
   nu_k=(r[k-1]**2.0).sum()/(z**2).sum()
   nu.append( nu_k )
   x_hwr.append( x_hwr[k-1]+nu[k]*p[k] )
   r.append( r[k-1]-nu[k]*numpy.dot(AT,z) )
   mu_k_1= (r[k]**2.0).sum()/(r[k-1]**2.0).sum()
   p.append( r[k]+mu_k_1*p[k] )
   rNorm=(r[-1]**2.0).sum()
timings['e:CGHWRl']=time.time()

print("CG(HWR) took {0:d} iterrations".format(k))
phasesToVisualize.append((x_hwr[-1],'CG(hwr)'))


# trad inv matrices
print("[mx inv]") ; sys.stdout.flush()
timings['s:inv']=time.time()
##A_dagger=numpy.linalg.pinv( gM, 1e-2)
if not cacheLoaded:
   A_dagger=numpy.dot( numpy.linalg.inv( numpy.dot(gM.T,gM)
      +numpy.identity(gO.numberPhases)*1e-6 ), gM.T )
else:
   A_dagger=data['A_dagger']
timings['e:inv']=time.time()
timings['s:dp']=time.time()
x_inv=numpy.dot( A_dagger, gradV )
timings['e:dp']=time.time()
phasesToVisualize.append((x_inv,'inv'))
      
if not cacheLoaded:
   timings['s:CacheWrite']=time.time()
   data={
         'gM':gM,
         'choleskyC':choleskyC,
         'smmtnsDef':smmtnsDef,
         'smmtnsDefChStrts':smmtnsDefChStrts,
         'smmtnsMap':smmtnsMap,
         'offsetEstM':offsetEstM,
         'A_dagger':A_dagger,
   }
   import pickle
   with file(cacheFn,'w') as f: pickle.dump(data,f)
   print("**wrote cache to "+cacheFn)
   timings['e:CacheWrite']=time.time()

# imaging of phases
reconPhaseViz=[]
reconPhaseD=numpy.empty([(N+1)]*2)
reconPhaseD=numpy.ma.masked_array( reconPhaseD, gO.illuminatedCorners==0 )
for i,phaseToViz in enumerate(phasesToVisualize):
   reconPhaseD.ravel()[gO.illuminatedCornersIdx]=phaseToViz[0]
   reconPhaseViz.append({
         'dat':reconPhaseD.copy(),
         'title':phaseToViz[1],
         'vec':phaseToViz[0]})



# remnant variances
print("input var=",reconPhaseViz[0]['vec'].var())
for i,phaseViz in enumerate(reconPhaseViz):
   print("input-{0:s} var=\t{1:f}".format(
      phaseViz['title'],(reconPhaseViz[0]['vec']-phaseViz['vec']).var()))

# waffle operator
waffleO=gradientOperator.waffleOperatorType1(pupilMask)
waffleV=waffleO.returnOp()
print("waffle var={0:f}".format(
      numpy.dot( waffleV, reconPhaseViz[0]['vec'] ).var() ))
for i,phaseViz in enumerate(reconPhaseViz):
   print("waffle {0:s} var=\t{1:f}".format(
      phaseViz['title'],numpy.dot(waffleV,reconPhaseViz[0]['vec']) ))

# timings
print("-"*10)
timingToPrintOut=[
      ("Phase creation","pC"), ("CG loop time","CGl"),
      ("HWR prep time","HWRprep"),
      ("HWR calc time","HWRcalc"),
      ("CG (HWR) loop time","CGHWRl"),
      ("Inv time","inv"),
      ("Phase vec. time","pvC"), ("Gradient vec. time","gvC"),
      ("Module loading","Modules"), ("MVM","dp")
     ]
if cacheLoaded:
   timingToPrintOut.append( ("Cache loading","CacheLoad") )
else:
   timingToPrintOut.append( ("Cache write","CacheWrite") )

for dat in timingToPrintOut:
   print("{1}={0:5.3f}s".format(
      timings['e:'+dat[1]]-timings['s:'+dat[1]],dat[0]))

def _doPlots():
   pg.figure(1)
   for plotNo,phaseViz in enumerate(reconPhaseViz):
      pg.subplot(2,len(reconPhaseViz),1+plotNo)
      pg.imshow( phaseViz['dat'], interpolation='nearest',
         origin='lower', extent=[-0.5,N+0.5,-0.5,N+0.5] )
      pg.title(phaseViz['title'])
      pg.colorbar()
      #
      if plotNo>0:
         pg.subplot(2,
               len(reconPhaseViz),1+len(reconPhaseViz)+(plotNo))
         pg.imshow( phaseViz['dat']-reconPhaseViz[0]['dat'],
            interpolation='nearest', origin='lower',
            extent=[-0.5,N+0.5,-0.5,N+0.5] )
         pg.title("diff")
         pg.colorbar()


   # plot CG iteration residual variance
   pg.figure()
   pg.semilogy( [ (x[i]-x_inv).var()/onePhaseV.var() for i in range(len(x)) ],
         '.-',label="$\mathrm{var}(x_{cg}[i]-x_{inv})$" )
   pg.semilogy( [ (x[i]-onePhaseV).var()/onePhaseV.var() for i in range(len(x)) ],
         '.-',label="$\mathrm{var}(x_{cg}i]-ip)$" )
   pg.semilogy( [ (x_hwr[i]-onePhaseV).var()/onePhaseV.var()
         for i in range(len(x_hwr)) ],
         '.-',label="$\mathrm{var}(x_{cg;hwr}i]-ip)$" )
   pg.plot( [0,len(x)-1], [(x_inv-onePhaseV).var()/onePhaseV.var()]*2,
         'k--',label="$\mathrm{var}(x_{inv}-ip)$" )
   pg.legend(loc=0)
   pg.title(sys.argv[0]+": variance, CG iterrations")
   pg.xlabel("CG iterrations")
   pg.ylabel("Residual variance")
 
try: 
   import matplotlib.pyplot as pg
except ImportError:
   print("No plotting available?")
else:
   _doPlots()
