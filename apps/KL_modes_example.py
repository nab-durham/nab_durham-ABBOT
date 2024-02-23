# -*- coding: utf-8 -*-
# What is this?
# Demonstrate the creation of Karhunen-Loève spatial modes
#

from __future__ import print_function

import abbot.phaseCovariance
import numpy
import sys
import Zernike

N=7 # sub-apertures/diameter
r0 = 1
L0 = N*100

aperture = Zernike.anyZernike(1,N,N//2+0.5)
#aperture-= Zernike.anyZernike(1,N,N//2*(1.2/4.2))
aperture-= Zernike.anyZernike(1,N,0.5)
Npts = aperture.sum()

print("VON KÁRMÁN WITH TILT::")
print("\tCalculating regular phase covariance...",end="") ; sys.stdout.flush()
directPhaseCovariance = abbot.phaseCovariance.covarianceDirectRegular(
      N, r0, L0
   )
print("(done)")

print("\tCalculating matrix form of phase covariance for aperture", end="")
sys.stdout.flush()
covarianceM = abbot.phaseCovariance.covarianceMatrixFillInMasked(
      directPhaseCovariance, aperture
   )
print("(done)")

print("\tComputing left-singular vectors...", end="") ; sys.stdout.flush()
uM,s,vMT = numpy.linalg.svd( covarianceM )
print("(done)")

blank = numpy.ma.masked_array( aperture*0, aperture==0 )
apIdx = aperture.ravel().nonzero()[0]

print("\tPlotting in figure 1")
import matplotlib.pyplot as pyplot
pyplot.figure(1)
Ncols = 17
modesRangesToPlot = (0,Ncols),(Npts//2,Npts//2+Ncols),(-Ncols,0)
modesRangesToPlot = (0,Ncols),(Ncols,Ncols*2), (Ncols*2,Ncols*3)
modesToPlot = []
for ranges in modesRangesToPlot:
   modesToPlot+=range(int(ranges[0]),int(ranges[1]))
for i,j in enumerate( modesToPlot ):
   if j>=uM.shape[1]:
    break
   pyplot.subplot( 3, Ncols, i+1 )
   if i==0: pyplot.text( -5,-5, "WITH TILT::")
   blank.ravel()[ apIdx ] = uM[:,j]
   pyplot.imshow( blank+0.0, origin="upper" )
   pyplot.title( "KL mode #{:3.0f}".format(j+1 if j>=0 else Npts+j) )

# THIS DOESN'T WORK...
# THIS DOESN'T WORK...print("VON KÁRMÁN W/O TILT::")
# THIS DOESN'T WORK...print("\tCalculating FFT phase covariance...",end="") ; sys.stdout.flush()
# THIS DOESN'T WORK...#directPhaseCovariance = REQUIRE A NEW FUNCTION HERE
# THIS DOESN'T WORK...print("(done)")
# THIS DOESN'T WORK...
# THIS DOESN'T WORK...print("\tCalculating matrix form of phase covariance for aperture", end="")
# THIS DOESN'T WORK...sys.stdout.flush()
# THIS DOESN'T WORK...covarianceM = abbot.phaseCovariance.covarianceMatrixFillInMasked(
# THIS DOESN'T WORK...      directPhaseCovariance, aperture
# THIS DOESN'T WORK...   )
# THIS DOESN'T WORK...print("(done)")
# THIS DOESN'T WORK...
# THIS DOESN'T WORK...print("\tComputing left-singular vectors...", end="") ; sys.stdout.flush()
# THIS DOESN'T WORK...uM,s,vMT = numpy.linalg.svd( covarianceM )
# THIS DOESN'T WORK...print("(done)")
# THIS DOESN'T WORK...
# THIS DOESN'T WORK...blank = numpy.ma.masked_array( aperture*0, aperture==0 )
# THIS DOESN'T WORK...apIdx = aperture.ravel().nonzero()[0]
# THIS DOESN'T WORK...
# THIS DOESN'T WORK...print("\tPlotting in figure 2")
# THIS DOESN'T WORK...import matplotlib.pyplot as pyplot
# THIS DOESN'T WORK...pyplot.figure(2)
# THIS DOESN'T WORK...Ncols = 4
# THIS DOESN'T WORK...modesRangesToPlot = (0,Ncols),(Npts//2,Npts//2+Ncols),(-Ncols,0)
# THIS DOESN'T WORK...modesRangesToPlot = (0,Ncols),(Ncols,Ncols*2), (Ncols*2,Ncols*3)
# THIS DOESN'T WORK...modesToPlot = []
# THIS DOESN'T WORK...for ranges in modesRangesToPlot:
# THIS DOESN'T WORK...   modesToPlot+=range(int(ranges[0]),int(ranges[1]))
# THIS DOESN'T WORK...for i,j in enumerate( modesToPlot ):
# THIS DOESN'T WORK...   pyplot.subplot( 3, Ncols, i+1 )
# THIS DOESN'T WORK...   if i==0: pyplot.text( -5,-5, "WITH TILT::")
# THIS DOESN'T WORK...   blank.ravel()[ apIdx ] = uM[:,j]
# THIS DOESN'T WORK...   pyplot.imshow( blank+0.0, origin="upper" )
# THIS DOESN'T WORK...   pyplot.title( "KL mode #{:3.0f}".format(j if j>0 else Npts+j) )
# THIS DOESN'T WORK...
# THIS DOESN'T WORK...# THIS DOESN'T WORK...
