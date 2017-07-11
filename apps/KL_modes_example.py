# -*- coding: utf-8 -*-
# What is this?
# Demonstrate the creation of Karhunen-Loève spatial modes
#

from __future__ import print_function

import abbot.phaseCovariance
import numpy
import sys
import Zernike

N=16 # sub-apertures/diameter
r0 = 1
L0 = N*10

aperture = Zernike.anyZernike(1,N,N//2)
#aperture-= Zernike.anyZernike(1,N,N//2*(1.2/4.2))
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
Ncols = 4
modesRangesToPlot = (0,Ncols),(Npts//2,Npts//2+Ncols),(-Ncols,0)
modesToPlot = []
for ranges in modesRangesToPlot:
   modesToPlot+=range(int(ranges[0]),int(ranges[1]))
for i,j in enumerate( modesToPlot ):
   pyplot.subplot( 3, Ncols, i+1 )
   if i==0: pyplot.text( -5,-5, "WITH TILT::")
   blank.ravel()[ apIdx ] = uM[:,j]
   pyplot.imshow( blank+0.0, origin="upper" )
   pyplot.title( "KL mode #{:3.0f}".format(j if j>0 else Npts+j) )


print("VON KÁRMÁN W/O TILT::")
print("\tCalculating FFT phase covariance...",end="") ; sys.stdout.flush()
#directPhaseCovariance = REQUIRE A NEW FUNCTION HERE
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

print("\tPlotting in figure 2")
import matplotlib.pyplot as pyplot
pyplot.figure(2)
Ncols = 4
modesRangesToPlot = (0,Ncols),(Npts//2,Npts//2+Ncols),(-Ncols,0)
modesToPlot = []
for ranges in modesRangesToPlot:
   modesToPlot+=range(int(ranges[0]),int(ranges[1]))
for i,j in enumerate( modesToPlot ):
   pyplot.subplot( 3, Ncols, i+1 )
   if i==0: pyplot.text( -5,-5, "WITH TILT::")
   blank.ravel()[ apIdx ] = uM[:,j]
   pyplot.imshow( blank+0.0, origin="upper" )
   pyplot.title( "KL mode #{:3.0f}".format(j if j>0 else Npts+j) )


