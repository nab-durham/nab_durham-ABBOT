# What is this?
# Test Fourier reconstruction with a Type 2 geometry

import matplotlib.pyplot as pg
import numpy
import gradientOperator
import commonSeed 

# test code
r0=1 # in pixels
L0=1000
nfft=48 # pixel size

# define pupil mask as sub-apertures
pupilMask=numpy.ones([nfft]*2,numpy.int32)
pupilCds=numpy.add.outer(
   (numpy.arange(nfft)-(nfft-1)/2.)**2.0,
   (numpy.arange(nfft)-(nfft-1)/2.)**2.0 )
#pupilMask=(pupilCds>(nfft/6)**2)*(pupilCds<(nfft/2)**2)
#>   # \/ spider
#> pupilMask[:nfft/2,nfft/2-1:nfft/2+1]=0
#> pupilMask[nfft/2:,nfft/2:nfft/2+2]=0
#> pupilMask[nfft/2-1:nfft/2+1,:nfft/2]=0
#> pupilMask[nfft/2:nfft/2+2,nfft/2:]=0
#pupilMask=(pupilCds<(nfft/2)**2)
gO=gradientOperator.gradientOperatorType2(pupilMask)
gM=gO.returnOp()

# define phase at pixels themselves, each of which is a sub-aperture
import phaseCovariance as pc

cov=pc.covarianceMatrixFillInRegular(
   pc.covarianceDirectRegular(nfft+2,r0,L0) )

choleskyC=pc.choleskyDecomp(cov)

# generate phase and gradients
numpy.random.seed( 1305903741 )
onePhase=numpy.dot( choleskyC, numpy.random.normal(size=(nfft+2)**2) )
onePhase=0.25*(onePhase.reshape([nfft+2]*2)[1:,1:]
   +onePhase.reshape([nfft+2]*2)[:-1,1:]+onePhase.reshape([nfft+2]*2)[:-1,:-1]
   +onePhase.reshape([nfft+2]*2)[1:,:-1] ).ravel()
onePhaseTrueV=onePhase[gO.illuminatedCornersIdx]
onePhase-=onePhaseTrueV.mean() # normalise
onePhase.resize([nfft+1]*2)
onePhaseD=numpy.ma.masked_array(onePhase,gO.illuminatedCorners==0)

onePhaseV=onePhase.ravel()[gO.illuminatedCornersIdx]
gradV=numpy.dot(gM,onePhaseV)


siz=64

testPhase=numpy.zeros([siz,siz],numpy.float64)
testPhase[siz/2-(nfft+1)/2:siz/2+(nfft+1)/2+1,siz/2-(nfft+1)/2:siz/2+(nfft+1)/2+1]=onePhase

testGrads=[ testPhase-numpy.roll(testPhase,1,axis=0), testPhase-numpy.roll(testPhase,1,axis=1) ]

cds=numpy.roll(numpy.arange(-siz/2,siz/2),siz/2)
testFilter=[ numpy.add.outer( (1-numpy.exp(-2.0j*numpy.pi/siz*cds)),numpy.zeros(siz) ) ] 
testFilter.append( testFilter[0].transpose() )
testFilter.append( 2*numpy.cos(2*numpy.pi/siz*cds) )
testFilter[-1]=4-testFilter[-1].reshape([-1,1])-testFilter[-1].reshape([1,-1])
testFilter[-1][0,0]=1e99

fft=numpy.fft
testRecon=(fft.fft2(testGrads[0])*testFilter[0]+fft.fft2(testGrads[1])*testFilter[1])*testFilter[2]**-1.0
testRecon[0,0]=0
testRecon=fft.ifft2(testRecon)

pg.figure()
pg.subplot(121)
pg.imshow( testRecon.real[siz/2-(nfft+1)/2:siz/2+(nfft+1)/2+1,siz/2-(nfft+1)/2:siz/2+(nfft+1)/2+1], interpolation='nearest' )
pg.subplot(122)
pg.imshow( onePhase, interpolation='nearest' )
print("Waiting for click on plot")
pg.waitforbuttonpress()
