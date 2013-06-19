from __future__ import print_function
# What is this?
# Simple-minded dm model with gaussian influence fns

import numpy


class dm(object):
   rotator=lambda self,x_y,ang : (
      numpy.cos(ang/180.*numpy.pi)*(x_y[0]-self.npix[0]/2.)
         -numpy.sin(ang/180.*numpy.pi)*(x_y[1]-self.npix[1]/2.)+self.npix[0]/2.,
      numpy.sin(ang/180.*numpy.pi)*(x_y[0]-self.npix[0]/2.)
         +numpy.cos(ang/180.*numpy.pi)*(x_y[1]-self.npix[1]/2.)+self.npix[1]/2. 
         )

   def __init__(self, shape, actGeom, rotation=0, within=0, ifScl=1,
         lateralScl=[1,1], lateralOffset=[0,0]):
      '''npix is the sampling for the DM surface
         actGeom represents the actuator geometry
         rotation is in degrees
         within is whether to let influence functions go over the edge
         ifScl is the scaling factor for the influence fn width
         lateralScl is actuator spacing along axes relative to the actuator
            grid, not the output axes.
         lateralOffset is a relative factor of actuator spacing for shifting the
            actuator grid.
      '''
      self.npix=shape
      self.actGeom=actGeom
      self.nacts=actGeom[0]*actGeom[1]
      self.rotation=rotation
      self.within=within
      self.ifScl=ifScl
      self.lateralScl=lateralScl
      self.lateralOffset=lateralOffset

      self.define()

   def define(self):
      '''Define projection geometry.
      '''
      self.coords()
      self.influenceFns()

   def coords(self):
      self.actCds=[]
#      offset=[ (
#            self.actGeom[i]+(self.within!=0)-1)/2. for i in (0,1)]
      self.actspacing=[ self.lateralScl[i]*self.npix[i]/float(
               self.actGeom[i]+(self.within!=0)) for i in (0,1) ]
      for i in range(self.nacts):
         (x,y)=( i//self.actGeom[1]-(self.actGeom[1]-1)/2.0-self.lateralOffset[1],
                i%self.actGeom[0]-(self.actGeom[0]-1)/2.0-self.lateralOffset[0] )
         self.actCds.append([ x*self.actspacing[0]+self.npix[0]/2.,
               y*self.actspacing[1]+self.npix[1]/2. ])
      if self.rotation!=0:
         for i in range(self.nacts):
            self.actCds[i]=self.rotator(self.actCds[i],self.rotation)

   def influenceFns(self):
      '''Influence functions, for input of 1'''

      rad=[ numpy.arange(self.npix[0]), numpy.arange(self.npix[1]) ]
      self.infFns=[]
      trad=lambda dirn : (rad[dirn]-self.actCds[i][dirn]+0.5)
      for i in range(self.nacts):
         self.infFns.append( numpy.exp(
           -numpy.add.outer(trad(0)**2/(self.ifScl*self.actspacing[0])**2.0,
                          trad(1)**2/(self.ifScl*self.actspacing[1])**2.0 )/2.)
#               numpy.multiply.outer(abs(trad(0))<=self.actspacing[0]/2.,
#                                    abs(trad(1))<=self.actspacing[1]/2.)
            )

   def returnInfFn(self,actnum):
      return self.infFns[actnum]

   def poke(self,actnum,ravel=1):
      '''Poke one actuator'''
      if ravel:
         return self.returnInfFn(actnum).ravel()
      else:
         return self.returnInfFn(actnum)

if __name__=="__main__":
   import Zernike
#   import projection
   import matplotlib.pyplot as pg
   import commonSeed
   import gradientOperator
   import phaseCovariance

   nfft=32

   #mask=(Zernike.anyZernike(1,nfft,nfft/2,ongrid=1)
   #   -Zernike.anyZernike(1,nfft,nfft/8,ongrid=1))
   mask=numpy.ones([nfft]*2)
   mask=mask.astype(numpy.int32)
   nMask=int(mask.sum())
  
   gO=gradientOperator.gradientOperatorType1( mask )
   gM=gO.returnOp()

   tdm=dm(gO.n_,[7]*2)

   pokeM=numpy.zeros([2*gO.numberSubaps,tdm.nacts],numpy.float64)
   for i in range(tdm.nacts):
      pokeM[:,i]=numpy.dot( gM, tdm.poke(i).take(gO.illuminatedCornersIdx) )

   reconM=numpy.linalg.pinv(pokeM)

   phaseMask=(gO.illuminatedCorners>0)

   # generate some test data
   r0=nfft/4
   L0=1e3
   rdm=numpy.random.normal(size=gO.numberPhases)
   
   directPCOne=phaseCovariance.covarianceDirectRegular( nfft+1, r0, L0 )
   directPC=phaseCovariance.covarianceMatrixFillInMasked(
      directPCOne, phaseMask )
   directcholesky=phaseCovariance.choleskyDecomp(directPC)
   directTestPhase=phaseCovariance.numpy.dot(directcholesky, rdm)
   testPhase2dI=numpy.zeros(gO.n_, numpy.float64)
   testPhase2dI.ravel()[gO.illuminatedCornersIdx]=directTestPhase
   testPhase2dI=numpy.ma.masked_array( testPhase2dI, [phaseMask==0] )

   slopes=numpy.dot( gM, directTestPhase )

   actuatorV=numpy.dot( reconM, slopes )
   reconPhaseV=numpy.zeros([gO.numberPhases],numpy.float64)
   for i in range(tdm.nacts):
      reconPhaseV+=( tdm.poke(i)*actuatorV[i] ).take(gO.illuminatedCornersIdx)

   recon2dI=numpy.zeros(gO.n_, numpy.float64)
   recon2dI.ravel()[gO.illuminatedCornersIdx]=reconPhaseV
   recon2dI=numpy.ma.masked_array( recon2dI, [phaseMask==0] )

   testPhase2dI-=testPhase2dI.mean()
   recon2dI-=recon2dI.mean()
   phsRge=[ (testPhase2dI.max()), (testPhase2dI.min()) ]
   pg.subplot(2,2,1) ; pg.imshow( testPhase2dI, vmin=phsRge[1],vmax=phsRge[0] ) ; pg.title("orig.")
   pg.subplot(2,2,3) ; pg.imshow( recon2dI, vmin=phsRge[1],vmax=phsRge[0] ) ; pg.title("recon.")
   pg.subplot(1,2,2) ; pg.imshow( recon2dI-testPhase2dI, vmin=phsRge[1],vmax=phsRge[0] ) ; pg.title("diff")

