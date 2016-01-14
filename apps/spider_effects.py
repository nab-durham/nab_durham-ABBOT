from __future__ import print_function
import numpy
import abbot.fourierSH as FSH
import abbot.gradientOperator as AgO
import scipy.ndimage.interpolation as Sint
import sys

N=11
pix=8

def rotatedCrossMask( N, pix, width, angle ):
   S=numpy.ceil( (N*pix*2**0.5)/2.0 )*2
   cds=numpy.add.outer(numpy.arange(-S/2,S/2)+0.25,numpy.zeros(S))
      # \/ make a cross mask
   mask=numpy.where( (abs(cds)>width/2.0)*(abs(cds.T)>width/2.0),1,0)
   rmask=Sint.rotate(mask,angle)
   return ( rmask[ rmask.shape[0]/2-N*pix/2:rmask.shape[0]/2+N*pix/2,
                   rmask.shape[1]/2-N*pix/2:rmask.shape[1]/2+N*pix/2 ]
          )

widths = (1,)#2,3)
angles = ( (numpy.arange(10)*( 45/(10-1.0) ))[0],)
nCentroids = N**2*2
outputData = {
      'scaling':numpy.zeros(
            [len(widths), len(angles), nCentroids], numpy.float64 ),
      'references':numpy.zeros(
            [len(widths), len(angles), nCentroids], numpy.float64 ),
   }

print("[",end="")
for i,twidth in enumerate(widths):
   print("+",end="")
   for j,tangle in enumerate(angles):
      print(".",end="") ; sys.stdout.flush()
      rmask = rotatedCrossMask( N, pix, twidth, tangle )
#      rmask = numpy.roll(rmask,twidth)
      fsh=FSH.FourierShackHartmann(
            N,rmask,0.1,1,4,lazyTruncate=0,guardPixels=1
         )
      fsh.calibrate()
      outputData['scaling'][i,j] = fsh.slopeScaling.copy()
      outputData['references'][i,j] = fsh.refSlopes.copy()

print("]")



##-----
## N.B. The following code will use the last rmask created i.e.
##  widths[-1] and angles[-1] to define it
##-----
##
#
print("Building reconstructor...",end="") ; sys.stdout.flush()
gO = AgO.gradientOperatorType1( numpy.ones([N]*2) )
gM = gO.returnOp()
gM_I = numpy.linalg.pinv( gM, 1e-3 )
print("(done)")
#
nReps = 100
#phs = numpy.random.normal( size=[nReps]+[N*pix]*2 )
import kolmogorov
phs = [ kolmogorov.TwoScreens( N*pix, 2 )[0] for dummy in range(nReps) ]
fshCSpider=FSH.FourierShackHartmann(
      N,rmask,0.1,1,4,lazyTruncate=0,guardPixels=1
   )
fshNoSpider=FSH.FourierShackHartmann(
      N,numpy.ones([N*pix]*2),0.1,1,4,lazyTruncate=0,guardPixels=1
   )
slopesV=[]
wfV=[]
print("Reconstructing ",end="") ; sys.stdout.flush()
for i,thisFSH in enumerate([fshCSpider,fshNoSpider]):
   print( ["once","twice","thrice"][i]+"...", end="" )
   thisFSH.calibrate()
   for j in range(nReps):
      thisFSH.makeImgs( phs[j], rmask )
      slopesV.append( thisFSH.getSlopes() )
      #
      wfV.append( (gM_I.dot( slopesV[-1] )) )
      print("-",end="") ; sys.stdout.flush()

print("")
#
##
#
nWfPoints=(N+1)**2
wfV=numpy.array(wfV).reshape([nReps,2,nWfPoints]).swapaxes(0,1)
differentialWavefront = ( ( (wfV[1]-wfV[0])**2.0 ).mean(axis=0)**0.5
      ).reshape([N+1]*2)
