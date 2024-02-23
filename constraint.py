"""ABBOT: Generate constraint operator for multi-layer reconstruction.

The concept is that if adjacent differences and sums between layer grid points,
with adjacent defined explicitly per usage and not here, are computed using
operator matrices <D> and <S> with the layer grid point values written as <w>
then,

  <w>T<D>T<D><w> is minimized if maximum correlated between the points,

and,

  <w>T<S>T<S><w> is minimized if maximum anti-correlated between points,

thus,
  
  <w>T(<D>T<D>+<S>T<S>)<w> is hypothesized to be minimized if reduced
  correlation exists between the points.

"""

from __future__ import print_function
from rounding import head,floor
import collections
import gradientOperator
import numpy
import version

class constraintOperatorCentreProjection(projection):
   '''Based on geometry, compute the per-layer indices of the centre-projected
   mask and build intra-layer adjacent-point sum and difference operator
   matrices.
   '''
   def __init__(self, layerHeights, zenAngles, azAngles, pupilMasks,
         starHeights=None, pixelScales=1, layerNpix=None, raiseWarnings=True
         sparse=False ):
      projection.__init__(self, layerHeights, zenAngles, azAngles, pupilMasks,
         starHeights, pixelScales, layerNpix, raiseWarnings)
      self.sparse=sparse

   def _operaterAdjacentMatrix(self, op):
      '''Define a matrix that operates on adjacent points per layer pairs.
      Centrally defined mask
      '''
      totalPts=(self.maskIdxs[-1])*(1-self.nLayers)
      summedPts=(self.maskIdxs[-1])
      if not self.sparse:
         opProjM=numpy.zeros( [ summedPts, totalPts ], numpy.float64 )
      else:
         import scipy.sparse, scipy.sparse.linalg
         opProjM={'dat':[],'col':[],'i':[0],'counter':0}
      lCPM=self.layerCentreProjectionMatrix()
      for i in range(self.nLayers-1):
         opProjM[i*summedPts:(i+1)*summedPts]=op(
               lCPM[i*summedPts:(i+1)*summedPts],
               lCPM[(i+1)*summedPts:(i+2)*summedPts] )
      return opProjM
   
   def _sumAdjacentMatrix(self):
      '''Define a matrix that sums adjacent points per layer pairs.
      Centrally defined mask
      '''
      return self._operatorAdjacentMatrix( lambda a,b: a+b )

   def _diffAdjacentMatrix(self):
      '''Define a matrix that differences adjacent points per layer pairs.
      Centrally defined mask
      '''
      return self._operatorAdjacentMatrix( lambda a,b: a-b )

   def returnOp(self):
      '''Return a constraint matrix for regularization.
      Centrally defined mask
      '''
      s,d=self._sumAdjacentMatrix(),self._diffAdjacentMatrix()
      return s.T.dot(s)+d.T.dot(d)


   def sumCentreProjectedMatrix(self):
      '''Define a matrix that sums the centre projected mask per layer-vector.
      '''
      totalPts=len(self.maskIdxs[-1])*self.nLayers
      summedPts=len(self.maskIdxs[-1])
      sumProjM=numpy.zeros( [ summedPts, totalPts ], numpy.float64 )
      # pretty straightforward, just ones for each layer's projection
      sumProjIdx=(
             numpy.arange(self.nLayers)*summedPts\
            +numpy.arange(summedPts).reshape([-1,1])*(1+totalPts)
         ).ravel()
      sumProjM.ravel()[ sumProjIdx ]=1
      return sumProjM

####
## class projectedModalBasis
## DISABLED:: it isn't clear that this class is utilized
####
##class projectedModalBasis(geometry):
##   modalBases=[]
##   radialPowers=None
##   angularPowers=None
##
##   def __init__(self, layerHeights, zenAngles, azAngles, pupilMask,
##         radialPowers, angularPowers, 
##         starHeights=None, pixelScale=1, layerNpix=None, sparse=False ):
##      geometry.__init__(self, layerHeights, zenAngles, azAngles, pupilMask,
##         starHeigh, pixelScale, layerNpix)
##      # for each layer, form an independent modal basis object
##      assert self.createLayerMasks()
##      modalBases=[ gradientOperator.modalBasis(
##            thisMask, radialPowers, angularPowers, sparse )
##            for thisMask in self.layerMasks ]

def edgeDetector(mask, clip=8):
   '''Using a convolution filter,
      / 1 1 1 \
      | 1 0 1 |
      \ 1 1 1 /  to detect neighbouring pixels, find the pixel indices
      corresponding to those that have less than 8, which are those
      with definitely one missing neighbour or less than 7, which are those
      with missing neighbours in the vertical/horizontal directions only.
      '''
   # do a bit of checking first
   mask=mask.astype(numpy.int16)
   if numpy.sum( (mask!=0)*(mask!=1) ):
      raise ValueError("The mask should have values of one and zero only.")
   import scipy.ndimage
   filter=[[1,1,1],[1,0,1],[1,1,1]]
   return (mask-(clip<=scipy.ndimage.filters.convolve(
                           mask,filter,mode='constant'))).ravel().nonzero()[0]

if __name__=='__main__':
   import Zernike
   import matplotlib.pyplot as pg
   import sys

   mask=Zernike.anyZernike(1,10,5,ongrid=1)-Zernike.anyZernike(1,10,2,ongrid=1)

#   \/ simplified geometry
   nAzi=4
   gsHeight=3
   thisProj=projection(
      numpy.array([0,1]),
      numpy.array([1]*nAzi),
      numpy.arange(nAzi)*2*numpy.pi*(nAzi**-1.0), mask,
      gsHeight )
   thisProj.define()
   okay=thisProj.createLayerMasks()
   if not okay:
      raise ValueError("Eek!")

   pg.figure()
   pg.gray()
   pg.subplot(2,2,1)
   pg.title("layer masks")
   pg.imshow( thisProj.layerMasks[0].sum(axis=0), interpolation='nearest' )
   pg.subplot(2,2,2)
   pg.imshow( thisProj.layerMasks[1].sum(axis=0), interpolation='nearest' )
   pg.subplot(2,2,3)
   pg.imshow( thisProj.layerMasks[0].sum(axis=0)>0, interpolation='nearest' )

   # fix a mask for the upper layer
   projectedMask=(thisProj.layerMasks[1].sum(axis=0)>0)*1.0
   edgeIdx=edgeDetector(projectedMask,clip=7)
   cds=lambda x : numpy.arange(x)-(x-1)/2.0
##   centreOffset = (
##      int(numpy.sign(thisProj.centreIdx[1])\
##         *(abs(thisProj.centreIdx[1])%thisProj.layerNpix[1,1])),
##      int(thisProj.centreIdx[1]/float(thisProj.layerNpix[1,1]))
##                  )
##   radius=numpy.add.outer(
##      centreOffset[0]+cds(thisProj.layerNpix[1,0])**2.0,
##      centreOffset[1]+cds(thisProj.layerNpix[1,1])**2.0 )**0.5
##   edgeRadii=radius.ravel()[edgeIdx]

   pg.subplot(2,2,4)
   pg.imshow( projectedMask, interpolation='nearest' )
##   circle=pg.Circle( [(thisProj.layerNpix[1,dirn]-1)/2.0 for dirn in (1,0)],
##      radius=edgeRadii.min(),lw=2,ec='w',fill=False)
##   patch=pg.gca().add_patch(circle)
##   # now, if there are small holes, want to find the smallest and the largest
##   # which is about 1/2 the layerNpix radius
##   if (edgeRadii.min()/thisProj.layerNpix[1].mean())<0.30:
##      newEdgeRadii=edgeRadii[numpy.flatnonzero( edgeRadii>edgeRadii.min() )]
##      circle2=pg.Circle( 
##         [centreOffset[dirn]+(thisProj.layerNpix[1,dirn]-1)/2.0
##            for dirn in (1,0)],
##         radius=newEdgeRadii.min(),lw=2,ls='dashed',ec='w',fill=False)
##      patch2=pg.gca().add_patch(circle2)
##      print("2ndary drawn")

   pg.draw()

   # try projection
   print("Projection matrix calcs...",end="")
   layerExM=thisProj.layerExtractionMatrix()
   sumPrM=thisProj.sumProjectedMatrix()
   trimIdx=thisProj.trimIdx()
   sumLayerExM=numpy.dot( sumPrM, layerExM.take(trimIdx,axis=1) )
   print("(done)")

      # \/ random values as a substitute dataset
   random=[ numpy.random.uniform(-1,1,size=tS) for tS in thisProj.layerNpix ]
   print("Input creation...",end="")
   randomA=[ numpy.ma.masked_array(random[i],
         thisProj.layerMasks[i].sum(axis=0)==0) for i in (0,1) ]
   randomV=(1*random[0].ravel()).tolist()+(1*random[1].ravel()).tolist()
   randomExV=numpy.take( randomV, trimIdx )
   randomProjV=numpy.dot( sumLayerExM, randomExV )
   print("(done)")
   
      # \/ create an imagable per-projection array of the random values
   projectedRdmVA=numpy.ma.masked_array(
      numpy.zeros([5]+list(mask.shape),numpy.float64),
      (mask*numpy.ones([5,1,1]))==0, astype=numpy.float64)
   projectedRdmVA.ravel()[
      (thisProj.maskIdxs[-1]+(numpy.arange(0,5)*mask.shape[0]*mask.shape[1]).reshape([-1,1])).ravel() ]\
         =randomProjV*numpy.ones(len(randomProjV))

   pg.figure()
   for i in range(nAzi):
      pg.subplot(3,2,i+1)
      pg.imshow( projectedRdmVA[i,:,:], interpolation='nearest' )
      pg.title("projection #{0:1d}".format(i+1))
   pg.xlabel("layer values")
   pg.draw()

   # correlate to be sure
   # do the first with the next four
   print("XC...",end="")
   nfft=128
   fpRVA=numpy.fft.fft2(projectedRdmVA,s=[nfft,nfft])
   fpRVA[:,0,0]=0
   smilT2= lambda x :\
      numpy.roll(numpy.roll( x,x.shape[-2]/2,axis=-2 ),x.shape[-1]/2,axis=-1)
   xc=numpy.array([ numpy.fft.ifft2(fpRVA[0].conjugate()*fpRVA[i])
      for i in range(0,5) ])
   xc=numpy.array([ 
      smilT2(txc)[nfft/2-mask.shape[0]/2:nfft/2+mask.shape[0]/2,
          nfft/2-mask.shape[1]/2:nfft/2+mask.shape[1]/2] for txc in xc ])
   pg.figure()
   for i in range(nAzi-1):
      pg.subplot(2,2,i+1)
      pg.imshow( abs(xc[i+1]-xc[0])**2.0, interpolation='nearest' )
      pg.title("xc (1,{0:1d})".format(i+1+1))
   print("(done)")
   pg.draw()

   # now, try straight inversion onto the illuminated portions of the layers 
   # with regularisation and via SVD
   print("Inversion...",end="")
   sTs=numpy.dot(sumLayerExM.transpose(),sumLayerExM)
   print(".",end="");sys.stdout.flush()
   sTs_invR=numpy.linalg.inv( sTs + 0.1*numpy.identity(len(trimIdx)) )
   print(".",end="");sys.stdout.flush()
   sTs_invSVD=numpy.linalg.pinv( sTs )
   print(".",end="");sys.stdout.flush()
   recoveryM=[ numpy.dot( thissTsI, sumLayerExM.transpose() )
      for thissTsI in (sTs_invR,sTs_invSVD) ]
   recoveryM.append( numpy.linalg.pinv( sumLayerExM ) ) # directly
   print("(done)")


   print("Recon...",end="")
   recoveredV=[ numpy.dot( thisrecoveryM, randomProjV )
      for thisrecoveryM in recoveryM ]
   recoveredLayersA=[[
      numpy.ma.masked_array(numpy.zeros(thisProj.layerNpix[i], numpy.float64),
         thisProj.layerMasks[i].sum(axis=0)==0) for i in (0,1)]
            for j in range(len(recoveryM)) ]
   layerInsertionIdx=thisProj.trimIdx(False)
   print("(done)")

   for j in range(len(recoveryM)):
      print("Type {0:d}".format(j+1))
      pg.figure()
      for i in range(thisProj.nLayers):
         recoveredLayersA[j][i].ravel()[layerInsertionIdx[i][1]]=\
            recoveredV[j][layerInsertionIdx[i][0]:layerInsertionIdx[i+1][0]]
         pg.title("layer 1, recon type="+str(j+1))
         pg.subplot(2,3,1+i*3)
         pg.imshow( recoveredLayersA[j][i]-recoveredLayersA[j][i].mean(),
            interpolation='nearest',vmin=-1,vmax=1 )
         pg.xlabel("recov'd")
         pg.subplot(2,3,2+i*3)
         pg.imshow( randomA[i]-randomA[i].mean(),
            interpolation='nearest',vmin=-1,vmax=1 )
         pg.xlabel("actual")
         pg.subplot(2,3,3+i*3)
         pg.imshow( recoveredLayersA[j][i]-randomA[i],
            interpolation='nearest',vmin=-1,vmax=1 )
         pg.xlabel("diff")
         print(" Layer#{0:d}".format(i+1))
         print("  Original RMS={0:5.3f}".format(randomA[i].var()))
         print("  Difference RMS={0:5.3f}".format(
            (randomA[i]-recoveredLayersA[j][i]).var()))

   pg.waitforbuttonpress()


