"""ABBOT: Phase covariance functions
Generate phase covariances via inverse discrete fourier transform (so regular
points, okay for SH) and more generally via direct computation

Enhance: use structure functions, piston-removed for Kolmogorov
"""

from __future__ import print_function

import numpy
import scipy.special

kv,gamma=scipy.special.kv,scipy.special.gamma

#[ 
#[ M. C. Roggemann, B. M. Welsh, D. Montera, and T. A. Rhoadarmer. Method for
#[ simulating atmospheric turbulence phase effects for multiple time slices and
#[ anisoplanatic conditions. Appl. Opt., 34:p4037, July 1995.
#[  
#, covVK=lambda r0,L0,r : r0**(-5/3.0)*numpy.pi*(
#,    (L0/2/numpy.pi)**(5/6.0)*kv(5/6.0,2*numpy.pi*r/L0)*(r**(5/6.0))
#,             )/2**(5/6.0)/gamma(11/6.0)
#                                              \/ bug in scipy->(r^4+alpha)^1/4 
covVK=lambda r0,L0,r : r0**(-5/3.0)*numpy.pi*(
   (L0/2/numpy.pi)**(5/6.0)*kv(5/6.0,2*numpy.pi*(r**4.0+1e-16)**0.25/L0)
      *(r**4.0+1e-16)**(5/(6.0*4)))/2**(5/6.0)/gamma(11/6.0)

#[#
#[# T. Butterley, R. W. Wilson, and M. Sarazin. Determination of the profile of
#[# atmospheric optical turbu- lence strength from SLODAR data. MNRAS,
#[# 369:835-845, June 2006.
#[# (Actually, Jenkins 1998 reference is given)
#[#
# constant changed from 0.17253 for consistency with covariance above
#                                                    \/ bug in scipy->where
sfVKfactor=( gamma(5/6.)*gamma(11/6.)*numpy.pi**(-8/3.)*(24/5.)**(5/6.)
      *gamma(6/5.)**(5/6.) )
sfVK=lambda r0,L0,r : sfVKfactor*(L0/r0)**(5/3.0)*numpy.where(
   r==0, 0, (1-(2*numpy.pi*r*L0**-1.0)**(5/6.0)*kv(5/6.0,2*numpy.pi*r/L0)
                                           *2**(1/6.0)*gamma(5/6.0)**-1.0))
#}sfVK=lambda r0,L0,r : 0.17621*(L0/r0)**(5/3.0)*numpy.where(
#}   r==0, 0, (1-(r/L0)**(5/6.0)*kv(5/6.0,2*numpy.pi*r/L0)
#}                                             *2*numpy.pi**(5/6.0)/gamma(5/6.0)))
# Initial factor also from,
#http://www.dei.unipd.it/~masiero/publications/masiero_josa07.pdf

#[
#[ Define phi(x)=Phi(x)-\int_A{Phi(x') dx'} s.t. the integral calculates the
#[ mean phase, \bar{Phi}, over the area A (so its a weighted integral) and
#[ means that phi(x) has finite covariance at zero separation.
#[
#[ see,
#[ C. M. Harding, R. A. Johnston, and R. G. Lane. Fast simulation of a
#[ kolmogorov phase screen. Appl.  Opt., 38(11):2161-2170, Apr 1999.
#[ 
# Thus,
# C_phi(x_1,x_2)=-0.5*D_phi(x_1,x_2)+0.5*\int_A{D_phi(x'_1,x_2) dx'_1}
#  +0.5*\int_A{D_phi(x_1,x'_2) dx'_2}
#  -0.5*\int_A{\int_A{D_phi(x'_1,x'_2) dx'_1} dx'_2}
#=>C_phi(0,x_2)=-0.5*D_phi(0,x_2)+0.5*\int_A{D_phi(x'_1,x_2) dx'_1}
#  +0.5*\int_A{D_phi(0,x'_2) dx'_2}
#  -0.5*\int_A{\int_A{D_phi(x'_1,x'_2) dx'_1} dx'_2}
# =-0.5*D_phi(0,x_2)+0.5*\int_A^{shifted}{D_phi(x'_1,0) dx'_1}
#  +0.5*\int_A{D_phi(0,x'_2) dx'_2}
#  -0.5*\int_A{\int_A{D_phi(x'_1,x'_2) dx'_1} dx'_2}
# =-0.5*D_phi(0,x_2)                                  <- [A]
#  +0.5*\int_A^{shifted}{D_phi(0,x'_2) dx'_2}         <- [B]
#  +0.5*\int_A{D_phi(0,x'_2) dx'_2}                   <- [C]
#  -0.5*\int_A{\int_A{D_phi(x'_1,x'_2) dx'_1} dx'_2}  <- [D]

def covarianceFFT( nfft, r0, L0=None, M=4, full=0):
   '''nfft=# of points in a square grid
      r0=Fried length in pixels
      M=oversampling, 2 minimum
   
    Calculate the covariance of one point, centred about zero, necessarily
    regularly spaced.
    Based on: covariance is auto-correlation which is inverse FFT of
    powerspectrum because uncorrelated terms in freq. space.'''
   if M<2: raise ValueError("Oversampling must be two or more")

   # \/ note, need nfft x M because need to oversample
   C=0.0229*numpy.power(M*nfft/(r0), 5.0/3.0)

   k=numpy.roll(numpy.arange(0,M*nfft)-M/2*nfft+0.0,M/2*nfft)
   k=numpy.add.outer(k**2.0,k**2.0)
   if L0:
      k[0,0]=1e6 # pole squashing
      powerSpec=C*numpy.power( k+4*numpy.pi**2.0/L0**2, -11/6.0 ) # V-K
   else:
      k[0,0]=1e6 # pole squashing
      powerSpec=C*numpy.power(k,-11/6.0) # Kolmogorov

      # \/ one point phase covariance, FFT based
   phaseCovF=numpy.fft.fft2(powerSpec).real
      # \/ centre and slice
   phaseCovF=numpy.roll(
      numpy.roll(phaseCovF,nfft,axis=0),nfft,axis=1)[:2*nfft,:2*nfft]
   phaseCovF=numpy.roll(numpy.roll(phaseCovF,nfft,axis=0),nfft,axis=1)
   return phaseCovF if not full else (phaseCovF, powerSpec)

def covarianceFFTZernikes( nfft, r0, j=(2,56), M=4, verbose=0, full=0):
   '''nfft=# of points in a square grid
      r0=Fried length in pixels
      j=(start,end) Zernikes to use
      M=oversampling, 2 minimum
   
    Calculate the covariance of one point, centred about zero, necessarily
    regularly spaced.
    Based on Fourier transforms of Zernikes which then force a circular
    geometry -> cannot be exact.
    Then covariance is auto-correlation which is inverse FFT of
    powerspectrum because uncorrelated terms in freq. space.
    '''
   import Zernike
##   if M<2: raise ValueError("Oversampling must be two or more")

      # \/ compute Zernike amplitude covariances 
   covAmps = []
   for zn1 in range(j[0],j[1]+1):
      zn1Cnf = Zernike.zernNumToDegFreq( zn1 )
      a1a1 = Zernike.kolmogorovCovariance(zn1,zn1,nfft,r0)
      covAmps.append( [zn1,zn1,a1a1] )
      for zn2 in range(zn1+1,j[1]+1):
         zn2Cnf = Zernike.zernNumToDegFreq( zn2 )
         if zn1Cnf[1]==zn2Cnf[2]: continue # different m are irrelevant
         #
         a1a2 = Zernike.kolmogorovCovariance(zn1,zn2,nfft,r0)
         if a1a2 == 0: continue
         covAmps.append( [zn1,zn2,2*a1a2] )

      # \/ sum Fourier representations of Zernikes based on stored amplitudes
   Qn = lambda n : Zernike.anyFourierZernike(n,M*nfft,M)
##(unfinished)   soft = Zernike.radius(M*nfft,M*nfft
##   Qn = lambda n : numpy.fft.fftshift(numpy.fft.ifft2(
##         numpy.fft.fftshift(Zernike.anyZernike(n,M*nfft,nfft/2,soft=0)) 
##      ))
   powerSpec = numpy.zeros([M*nfft]*2, numpy.complex128 )
   
   for (zn1,zn2,amp) in covAmps:
      if verbose:
         print("Adding for Z_{:d}Z_{:d} with {:5.3e}".format(zn1,zn2,amp))
      powerSpec += amp*Qn(zn1).conjugate()*Qn(zn2)

   powerSpec = numpy.fft.fftshift( powerSpec )/M**2
      # \/ one point phase covariance, FFT based
   phaseCovF=numpy.fft.fft2( powerSpec ).real
      # \/ centre and slice
   phaseCovF=numpy.roll(
      numpy.roll(phaseCovF,nfft,axis=0),nfft,axis=1)[:2*nfft,:2*nfft]
   phaseCovF=numpy.roll(numpy.roll(phaseCovF,nfft,axis=0),nfft,axis=1)
   return phaseCovF if not full else (phaseCovF, powerSpec)

def covarianceDirectOneSpacing( dist, r0, L0=None ):
   '''dist=spacing for covariance calculation
      r0=Fried length in pixels
   
    Calculate the covariance of a pair of points.
    Based on: direct covariance formula (only works for finite L0)

    or structure functions for infinite L0.
   
    C. M. Harding, R. A. Johnston, and R. G. Lane. Fast simulation of a
    kolmogorov phase screen. Appl.  Opt., 38(11):2161-2170, Apr 1999.
    '''
   if L0:
      return covVK(r0,L0,dist)
   else:
      raise NotImplemented("Not implemented for Kolmogorov")

def covarianceDirectRegular( nfft, r0, L0=None ):
   '''nfft=either # of points in a square grid or dimensions of rectagle
      r0=Fried length in pixels
   
    Calculate the covariance of one point, centred about zero, regularly
    spaced.
    '''

   if L0:
# FIXME:- change the following line to better test for a sequence
      if "__len__" in dir(type(nfft)):
         ra=numpy.roll( numpy.arange(2*nfft[0])-nfft[0],nfft[0] ) # twice as big
         rb=numpy.roll( numpy.arange(2*nfft[1])-nfft[1],nfft[1] )
      else:
         ra=numpy.roll( numpy.arange(2*nfft)-nfft,nfft )
         rb=ra
      r=numpy.add.outer(ra**2,rb**2)**0.5
      return covarianceDirectOneSpacing( r,r0,L0 )
   else:
# \/ ought to work
#}      r=numpy.arange(4*nfft)-2*nfft
#}      r=numpy.add.outer( r**2, r**2 )**0.5
#}      sf=6.88*(r/r0)**(5/3.0)
#}     
#}      averagedSf=numpy.zeros([2*nfft]*2, numpy.float64)
#}      for i in range(nfft,2*nfft+1):
#}         for j in range(1,2*nfft+1):
#}            averagedSf[i-1,j-1]=sf[i:i+2*nfft,j:j+2*nfft].mean()
      raise NotImplementedError(
         "Direct covariance, L0=None (Kolmog), doesn't work for some reason")
      # calculate structure function, using dumb method
      # must have an odd-sized grid to account for the even nature of integral [B]
      # makes centre of grid zero-point
      cds=numpy.arange(2*nfft+1)-nfft
      x=numpy.multiply.outer(cds,numpy.ones(2*nfft+1))
      y=numpy.multiply.outer(numpy.ones(2*nfft+1),cds)
      C=6.88*r0**(-5/3.0)
      sf=C*numpy.power(x**2.0+y**2.0, 5/6.0)
      averagedSf=numpy.zeros([2*nfft+1]*2, numpy.float64)
      for i in range(2*nfft):
        for j in range(2*nfft):
          averagedSf[i,j]=C*( numpy.power((y-y[i,j])**2.0+(x-x[i,j])**2.0,5/6.0) ).mean()
      covOnce=numpy.zeros( [2*nfft]*2, numpy.float64 )
         # \/ [A] & [B]
      covOnce=-0.5*sf[:-1,:-1]+0.5*averagedSf[:-1,:-1]\
              +0.5*averagedSf[nfft,nfft]-0.5*averagedSf[:-1,:-1].mean() # [B] & [D]
      return numpy.roll(numpy.roll(covOnce,nfft,axis=1),nfft,axis=0)

class covarianceMatrix(numpy.ndarray):
   '''Internal class that should only be instantiated by a following function.'''
   def __new__( self, mask ):
      mask = numpy.array(mask).astype(numpy.bool)
      assert len(mask.shape)==2, "Expect a 2D mask"
      maskI = numpy.flatnonzero( mask.ravel() )
      illuminatedNo = len(maskI)
      shapeFromMask = [illuminatedNo]*2 # size
      mask2gridI = numpy.arange(
            shapeFromMask[0]*shapeFromMask[1] )[maskI]
      obj = numpy.ndarray.__new__( self, shapeFromMask ).view(
             covarianceMatrix )
      obj.mask,obj.maskI,obj.mask2gridI = mask, maskI, mask2gridI
      return( obj )


def covarianceMatrixFillInRegular( SinglePhaseCovariance ):
   '''Calculate a regularly spaced covariance matrix given a regularly spaced
   single point covariance'''
   maskShape=[ thisShape/2 for thisShape in SinglePhaseCovariance.shape ]
   return covarianceMatrixFillInMasked(
         SinglePhaseCovariance, 
         numpy.ones( maskShape, numpy.bool )
      )

def covarianceMatrixFillInMasked( SinglePhaseCovariance, Mask ):
   '''Calculate a covariance matrix given a mask and single point covariance that is regularly spaced.'''
   covS=SinglePhaseCovariance.shape
   maskS=Mask.shape
   assert (covS[0]/2>=maskS[0] or covS[1]/2>=maskS[1]),\
      "Size mismatch between one-point covariance and mask"
   covM = covarianceMatrix( Mask )
      # \/ recentre covariances
   covCent = [ (thisDim)/2 for thisDim in covS ]
   for axis in (0,1):
      SinglePhaseCovariance=numpy.roll(
         SinglePhaseCovariance,covCent[axis], axis=axis)
      # \/ copy covariances
   for i,maskIdx in enumerate( covM.maskI ):
      maskPos=( maskIdx//maskS[1],maskIdx%maskS[1] )
         # \/ 1. remove the correctly placed and sized region
      covSlice=SinglePhaseCovariance[
         covCent[0]-maskPos[0]:covCent[0]-maskPos[0]+maskS[0],
         covCent[1]-maskPos[1]:covCent[1]-maskPos[1]+maskS[1]
            ]
         # \/ 1. convert to 1D and apply mask
      covM[i] = covSlice.ravel()[covM.maskI]
   return covM 

def covarianceMatrixExtractInto2D( covM, average=True ):
   '''Extract from a regularly spaced covariance matrix, the single point covariance function.
   By default, the covariance functions will be averaged to obtain the one
   single point function but this can be altered to get all possible single
   point functions.
   '''
   maskS = covM.mask.shape
   covCent = maskS # they are identical by defn.
   covSPS = [ (thisDim)*2 for thisDim in maskS ]
      # \/ the single point (SP) covariance
   covSPEst = numpy.zeros(
         ([] if average else [covM.mask.sum()])+covSPS, numpy.float64 ) 
      # \/ no. per SP covariance estimate
   covNPts = numpy.zeros(
         ([] if average else [covM.mask.sum()])+covSPS, numpy.int32 ) 
   #
   covRegion = numpy.ma.masked_array( numpy.zeros(maskS), ~covM.mask )
   for i,maskIdx in enumerate( covM.maskI ):
      maskPos=( maskIdx//maskS[1],maskIdx%maskS[1] )
         # \/ 1. using mask, convert to 2D
      covRegion.ravel()[covM.maskI] = covM[i]
      if average: 
            # \/ 2a. save onto a single point estimate
         covSPEst[
            covCent[0]-maskPos[0]:covCent[0]-maskPos[0]+maskS[0],
            covCent[1]-maskPos[1]:covCent[1]-maskPos[1]+maskS[1]
               ] += covRegion
         covNPts[
            covCent[0]-maskPos[0]:covCent[0]-maskPos[0]+maskS[0],
            covCent[1]-maskPos[1]:covCent[1]-maskPos[1]+maskS[1]
               ] += covM.mask
      else: 
            # \/ 2b. save into a separate single point estimate
         covSPEst[i,
            covCent[0]-maskPos[0]:covCent[0]-maskPos[0]+maskS[0],
            covCent[1]-maskPos[1]:covCent[1]-maskPos[1]+maskS[1]
               ] = covRegion
         covNPts[i,
            covCent[0]-maskPos[0]:covCent[0]-maskPos[0]+maskS[0],
            covCent[1]-maskPos[1]:covCent[1]-maskPos[1]+maskS[1]
               ] = covM.mask
   #
   if average:
      covSPEst /= numpy.where( covNPts, covNPts.astype( numpy.float64 ),1 ) 
   covSPEst = numpy.ma.masked_array( covSPEst, covNPts==0 ) # mask
   return( covSPEst, covNPts )


def choleskyDecomp( Covariance ):
   '''Cholesky decomposition, using numpy and returning a float64 array.
   '''
   return numpy.linalg.cholesky(Covariance).view(numpy.ndarray)

if __name__=='__main__':
   timing={}
   # test code
   nfft=32
   nTest=1000
   r0=nfft/2
   L0=1e3

   rdm=numpy.random.normal(size=[nfft**2.0,nTest])

   def addtime(keybase):
      if keybase in timing.keys():
         timing[keybase].append( time.time() )
      else:
         timing[keybase]=[ time.time() ]

   # expected residuals less Zernike removal
   expectedVar=numpy.array([
      1.03,  0.582, 0.134,
      0.11,  0.088, 0.0648,
      0.0587,0.0525,0.0463,
      0.0401,0.0377,0.0352,
      0.0328,0.0304,0.0279 ])*(nfft*r0**-1.0)**(5/3.0)

   import time
   # first, FFT
   addtime('ft')
   fftPCOne=covarianceFFT( nfft, r0, L0, M=8 ) ; addtime('ft')
   fftPC=covarianceMatrixFillInRegular( fftPCOne ) ; addtime('ft')
   fftcholesky=choleskyDecomp(fftPC) ; addtime('ft')
   fftTestPhase=numpy.dot(fftcholesky, rdm) ; addtime('ft')

   sqPhase=fftTestPhase.reshape([nfft,nfft,nTest])
   import sys
   print("<",end="") ; sys.stdout.flush()
   sf=numpy.array([[ [ (i**2.0+j**2.0)**0.5,
         ( (sqPhase[i:,j:]-sqPhase[:nfft-i,:nfft-j]
            )**2.0).reshape([-1,nTest]).mean(axis=0).mean(axis=0) ]
      for i in range(0,nfft,4) ] 
         for j in range(0,nfft,4)]).reshape([(nfft/4)**2,2])
   print(">")

   import matplotlib.pyplot as pylab
   pylab.figure(1)
   pylab.title("structure functions")
   sfDist=sf[:,0]+0.0 ; sfDist.sort()
   if L0:
      pylab.plot( sfDist,sfVK(r0,L0,sfDist),
         'k', lw=1, alpha=0.75, label='theory (V-K)' )
      pylab.plot( sfDist, 6.88*(sfDist/r0)**(5/3.0),
         'k--', lw=1, alpha=0.75, label='theory (K.)' )
   else:
      pylab.plot( sfDist, 6.88*(sfDist/r0)**(5/3.0),
         'k', lw=1, alpha=0.75, label='theory' )
   pylab.plot( sf[:,0], sf[:,1], 'b.', label='structure function, FFT' )

   try:
      import Zernike
      haveZernike=1
   except:
      haveZernike=0
      print("No Zernike module?")
   if haveZernike:
      z15=Zernike.Zernike15(nfft,ongrid=1)
      z15Norm=z15[0].sum() ; z15/=z15Norm**0.5
      z15f=z15.reshape([15,nfft**2])
      z15Cov=numpy.dot(z15f,z15f.transpose()) # covariance of sampled Zernikes
      chol=numpy.linalg.cholesky(z15Cov)
      cholInv=numpy.linalg.inv(chol) # orthogonalise
      zIdx=numpy.flatnonzero( z15[0].ravel() ) # non-zero parts

      zfftCoeffs=numpy.dot(cholInv, numpy.dot( z15f, fftTestPhase ) )/(z15Norm**0.5)
      phaseVar=( fftTestPhase.take(zIdx,axis=0)**2.0 ).sum(axis=0)/(z15Norm)
      remnantVar=numpy.array([
         phaseVar-(zfftCoeffs[:i+1]**2.0).sum(axis=0) for i in range(15) ]).mean(axis=1)
      pylab.figure(2)
      pylab.title("Phase variance remnants")
      pylab.plot( numpy.arange(1,16), expectedVar,'wo', label='Kolmog theory')
      pylab.plot( numpy.arange(1,16), remnantVar,'b.', label='FFT')

   print("Waiting for click in matplotlib window...",end="")
   sys.stdout.flush()
   #pylab.waitforbuttonpress()
   print("(cont.)")
#[ Polling in matplotlib does not work
#[    waiting=True
#[    def onclick(event): waiting=False
#[    cid = pylab.gcf().canvas.mpl_connect('button_press_event', onclick)
#[    import time
#[    while waiting:
#[       time.sleep(0.1) 

   # second, direct
   addtime('direct')
   directPCOne=covarianceDirectRegular( nfft, r0, L0 ) ; addtime('direct')
   directPC=covarianceMatrixFillInRegular( directPCOne ) ; addtime('direct')
   directcholesky=choleskyDecomp(directPC) ; addtime('direct')
   directTestPhase=numpy.dot(directcholesky, rdm) ; addtime('direct')

   sqPhase=directTestPhase.reshape([nfft,nfft,nTest])
   print("<",end="") ; sys.stdout.flush()
   sf=numpy.array([[ [ (i**2.0+j**2.0)**0.5,
         ( (sqPhase[i:,j:]-sqPhase[:nfft-i,:nfft-j]
            )**2.0).reshape([-1,nTest]).mean(axis=0).mean(axis=0) ]
      for i in range(0,nfft,4) ] 
         for j in range(0,nfft,4)]).reshape([(nfft/4)**2,2])
#[    sf=numpy.array([[
#[       [ (i**2.0+j**2.0)**0.5,
#[          ((sqPhase[i:,j:]-sqPhase[:nfft-i,:nfft-j])**2.0).reshape([-1,nTest]).mean(axis=0).mean(axis=0) ]
#[       for i in range(nfft) ] 
#[          for j in range(nfft)]).reshape([(nfft)**2,2])
   print(">")

   pylab.figure(1)
   pylab.plot( sf[:,0], sf[:,1], 'g.', label='structure function, direct' )
   pylab.legend(loc=0)

   if haveZernike:
      zdirectCoeffs=numpy.dot(cholInv, numpy.dot( z15f, directTestPhase ) )/(z15Norm**0.5)
      phaseVar=( directTestPhase.take(zIdx,axis=0)**2.0 ).sum(axis=0)/(z15Norm)
      remnantDirectVar=numpy.array([
         phaseVar-(zdirectCoeffs[:i+1]**2.0).sum(axis=0) for i in range(15) ]).mean(axis=1)
      pylab.figure(2)
      pylab.plot( numpy.arange(1,16), remnantDirectVar,'g.', label='direct')

   #pylab.show()
   
   print("Waiting for click in matplotlib window") ; sys.stdout.flush()
   #pylab.waitforbuttonpress()

   # check masked method
   #mask=numpy.zeros([nfft,nfft/4])
   #maskLen=int(numpy.where(1000>(nfft**2.0/4.0),nfft**2.0/4.0,1000))
   #maskIdx=numpy.random.uniform(0,nfft**2,size=[maskLen]).astype(numpy.int)
   #mask.ravel()[maskIdx]=1
   #mask[0,0]=1 # ensure first point is zero
   #mask[:nfft/4,:nfft/4]=1 ; mask[-nfft/4:,:nfft/4]=1
   mask = Zernike.anyZernike(1,nfft,nfft/2.)
   maskIdx=mask.ravel().nonzero()[0]
   maskIdx.sort()
   maskLen=len(maskIdx)

      # \/ core to test
   rdm=numpy.random.normal(size=[maskLen,nTest]) 
   addtime('masked') ; addtime('masked')
   masked=covarianceMatrixFillInMasked(directPCOne,mask) ; addtime('masked')
   CDMasked=choleskyDecomp(masked) ; addtime('masked') 
   maskedPV=numpy.dot(CDMasked,rdm) ; addtime('masked')

      # \/ decorate
   maskDist=[ ( (((maskIdx[tMI])%mask.shape[1])**2.0
                +((maskIdx[tMI])//mask.shape[1])**2.0)**0.5,tMI)
         for tMI in range(1,maskLen) ]
   maskDist.sort()
   maskDist=numpy.array(maskDist)
   SortedIdx=maskDist[:,1].astype('i') # undecorate
   SortedDist=maskDist[:,0] # undecorate

   sfMaskedPV=((maskedPV[0]-maskedPV[SortedIdx])**2.0).mean(axis=1) 
   pylab.figure(1)
   pylab.plot( SortedDist, sfMaskedPV,'r.', label='sf via masked cov')
   pylab.legend(loc=0)
   
   print("Waiting for click in matplotlib window") ; sys.stdout.flush()
   pylab.waitforbuttonpress()

   # timing analysis
   print("FFT times,")
   ftTiming=numpy.array(timing['ft'])
   print(ftTiming-ftTiming[1])
   
   print("Direct times,")
   directTiming=numpy.array(timing['direct'])
   print(directTiming-directTiming[1])
   
   print("Masked times,")
   maskedTiming=numpy.array(timing['masked'])
   print(maskedTiming-maskedTiming[1])
