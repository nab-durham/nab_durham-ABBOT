from __future__ import print_function
# What is this?
# Define a projection matrix and geometry for AO
# Includes variable guidestar height (LGS) and non-one pixel scales

import numpy
from rounding import head,floor
import gradientOperator

def quadrantFractions( v,h,s ):
   '''For given central vertical & horizontal coordinate and the scale,
   what are the pixel weightings that contribute. Assume at least 2 in each
   direction, although there may be more.
   They are relative to int(v),int(h) and the fractions should add to s**2.
   Their relative offsets are
   (0,0),(+1,0),(0,+1),(+1,+1).
   '''
   v0,h0=int(floor( v-s/2.+0.5 )),int(floor( h-s/2.+0.5 ))
      # \/ note we round to 4 dp to ensure sensible fractions
      #  which does mean the addition isn't necessarily to s^2
   # pixel edges are {v/h}-s/2.,{v/h}+s/2.
   # check against {v0+n-0.5,v0+n+0.5 [n->{0,max(s+1,2)}] }
   # also ensure pixel hasn't slipped off the edge of this pixel
   fracs=[] ; rcs=[]
   for iv in range(numpy.where( s<=1, 2, head(s+1) )):
      vf=min( 0.5,max(v+s/2.-(v0+iv),-0.5) )-max( -0.5,v-s/2.-(v0+iv) )
      if (numpy.round(vf,4)):
         for ih in range(numpy.where( s<=1, 2, head(s+1) )):
            hf=min( 0.5,max(h+s/2.-(h0+ih),-0.5) )-max( -0.5,h-s/2.-(h0+ih) )
            if (numpy.round(hf*vf,4)>0): # have to check both
               rcs.append([iv,ih])
               fracs.append(vf*hf*s**-2.0) # aha! (& scale too)
   assert numpy.array(fracs).sum()>0.9999,\
         "Insufficient fractional subdivision"
   assert numpy.array(fracs).sum()<1.0001,\
         "Excess fractional subdivision"
   return (numpy.array(rcs).astype(numpy.int32),
         s**2.0*numpy.array(fracs).astype(numpy.float64),v0,h0)

class geometry(object):
   layerMasksFilledIn=False
   def _hs(self,x):
      if type(self.starHeight)==type(None):
         return self.pixelScale
      else:
         return self.pixelScale*(1-self.layerHeights[x]*(self.starHeight**-1.0))

   def __init__(self, layerHeights, zenAngles, azAngles, pupilMask,
         starHeight=None, pixelScale=1, layerNpix=None ):
      '''Layer heights in [metres], angles in [radians]
      '''
      self.nAzi=len(azAngles)
      self.nLayers=len(layerHeights)
      self.npix=numpy.array(pupilMask.shape)
      self.layerHeights=layerHeights
      self.zenAngles=zenAngles
      self.azAngles=azAngles
      self.pupilMask=pupilMask
      self.starHeight=starHeight # None is infinity, NGS
      # need at least 10% of the guidestar altitude between the final layer and
      # it \/
      starHeightMinDistance=0.1 
      if type(starHeight)!=type(None):
         if 1-self.layerHeights[-1]/self.starHeight<starHeightMinDistance:
            raise ValueError("Must have the guide star at a higher altitude")
      self.pixelScale=pixelScale
      self.layerNpix=layerNpix

      self.maskIdx=self.pupilMask.ravel().nonzero()[0]
      self.define()

   def define(self):
      '''Define projection geometry.
      '''
      # calculate the x,y offsets of the projected aperture on the layers
      self.offsets=numpy.zeros( [ self.nLayers,self.nAzi,2 ], numpy.float64 )
      for i in range(self.nLayers):
         for j in range(self.nAzi):
            for k in 0,1:
               if k:
                  tf=numpy.sin
               else:
                  tf=numpy.cos
               self.offsets[i,j,k]=float(self.layerHeights[i])*\
                       numpy.round(tf(self.azAngles[j]),6)*self.zenAngles[j]

      self.maxxydim=numpy.zeros([self.nLayers,2,2], numpy.int32)
      for i in range(self.nLayers):
         for k in (0,1):
            # \/ establish how much bigger the dimensions of the support become 
            if type(self.layerNpix)==type(None):
               for j in range(self.nAzi):
                  coordL=[
                     head( self._hs(i)*(self.npix[k]/2)+self.offsets[i,j,k]),
                     head( self._hs(i)*(-self.npix[k]/2)+self.offsets[i,j,k]) ]
                  if coordL[0]>self.maxxydim[i,k,0]: self.maxxydim[i,k,0]=coordL[0]
                  if coordL[1]<self.maxxydim[i,k,1]: self.maxxydim[i,k,1]=coordL[1]
            else: # just calculate maxxydim
               for l in (0,1): self.maxxydim[i,k,l]=(1-2*l)*self.layerNpix[i][k]//2
      self.layerNpix=self.maxxydim[:,:,0]-self.maxxydim[:,:,1]
 
      # index self.offsets, centred projection may not be (0,0)
      self.centreIdx=\
         (self.maxxydim.sum(axis=2)[:,1]/2.0).astype(numpy.int32)\
        +(self.maxxydim.sum(axis=2)[:,0]/2.0).astype(numpy.int32)\
            *self.layerNpix[:,1]
      self.centreOffsets=(self.layerNpix-1)/2.0\
            -(self.maxxydim[:,:,1]+self.maxxydim[:,:,0])/2.0
      
      self.maskCentre=[ (self.npix[i]-1)/2.0 for i in (0,1) ]
      self.layerMasks=[
         numpy.zeros([self.nAzi]+self.layerNpix[i].tolist(),numpy.float32)
            for i in range(self.nLayers) ]
      
   def maskLayerCentreIdx(self, layer, flat=0):
      '''For a layer, return the indices per mask pixel and their fractions
      for centre projection (zero zenith angle).
      '''
      return (self._maskLayerIdx(layer, [0,0], flat))
             
   def maskLayerIdx(self, layer, azi, flat=0):
      '''For a layer, return the indices per mask pixel and their fractions
      for that azimuth.
      '''
      return (self._maskLayerIdx(layer, self.offsets[layer,azi], flat))
   
   def _maskLayerIdx(self, layer, offsets, flat):
      '''Generic: return for a layer the offset mask position. Use
      maskLayerCentreIdx or maskLayerIdx instead of this.
      '''
      self.maskCoords=(numpy.array([ self.maskIdx//self.npix[0],
               self.maskIdx%self.npix[1]])-numpy.reshape(self.maskCentre,[2,1])
               )*self._hs(layer)+numpy.reshape(self.centreOffsets[layer],[2,1])

      indices=[] ; fractions=[]
      for i in range(self.maskIdx.shape[0]):
         rcs,fracs,v0,h0=quadrantFractions( self.maskCoords[0,i]+offsets[0],
               self.maskCoords[1,i]+offsets[1],self._hs(layer))
         thisidx=(rcs[:,0]+v0)*self.layerNpix[layer,1]+(h0+rcs[:,1])
         if not flat:
            indices.append(thisidx)
            fractions.append(fracs)
         else:
            indices+=list(thisidx)
            fractions+=list(fracs)
      if not flat:
         return (indices,fractions)
      else:
         return (numpy.array(indices), numpy.array(fractions))

   def createLayerMasks(self):
      '''Create the masks for each layer by imposing the projected pupil
      masks for the given azimuth. Only once.
      '''
      if self.layerMasksFilledIn: return True 
      self.layerMasksFilledIn=True # presume okay
      for nl in range(self.nLayers):
         for na in range(self.nAzi):
            indices,fractions=self.maskLayerIdx(nl,na,flat=1)
            valid=numpy.flatnonzero( (indices>-1)*
                     (indices<self.layerNpix[nl,0]*self.layerNpix[nl,1]) )
            if len(valid)!=len(indices):
               print("Eurgh. Something horid has happened;nl={0:d},na={1:d}".format(nl,na))
               self.layerMasksFilledIn=False
#            self.layerMasks[nl][na].ravel()[ indices[valid] ]+=fractions[valid]
# /\ doesn't work because indices has repeat values so have to do by hand
# \/ but this can be slow: probably need a C module for speed
            for i in valid:
               self.layerMasks[nl][na].ravel()[indices[i]]+=fractions[i]
       
      return self.layerMasksFilledIn

   def layerIdxOffsets(self):
      '''Indexing into the concatenated-layer vector, to extract each layer.
      '''
      return [0]\
         +(self.layerNpix[:,0]*self.layerNpix[:,1]).cumsum().tolist()

class projection(geometry):
   '''Based on geometry, calculate the tomography matrices for projection of
   values.
   '''
   def layerExtractionMatrix(self):
      '''Define a layer extraction matrix, that extracts each projected
      mask from the concatenated layer-vector.
      '''
      layerIdxOffsets=self.layerIdxOffsets()
            # \/ /\ only time [-1] used is for the total size
      extractionM=numpy.zeros( [ len(self.maskIdx)*self.nLayers*self.nAzi,
         layerIdxOffsets[-1] ], numpy.float64 )
      # matrix can be filled in by saying:
      # for each layer,
      #   for each azimuth angle,
      #     find the indices in the layer and the fraction for each
      #     these represent the entries in the matrix
      for nl in range(self.nLayers):
         for na in range(self.nAzi):
            projectedIdxOffset=len(self.maskIdx)*(na+nl*self.nAzi)
            indices,fractions=self.maskLayerIdx(nl,na)
            for i in range(len(self.maskIdx)):
               extractionM[projectedIdxOffset+i,layerIdxOffsets[nl]+indices[i]]\
                     +=fractions[i]
      return extractionM

   def layerCentreProjectionMatrix(self):
      '''Define a layer extraction matrix, that extracts a centrally
      projected mask through the concatenated layer-vector.
      '''
      layerIdxOffsets=self.layerIdxOffsets()
            # \/ /\ only time [-1] used is for the total size
      extractionM=numpy.zeros(
          [len(self.maskIdx)*self.nLayers, layerIdxOffsets[-1]], numpy.float64 )
      # matrix can be filled in by saying:
      # for each layer,
      #   for each azimuth angle,
      #     find the indices in the layer and the fraction for each
      #     these represent the entries in the matrix
      for nl in range(self.nLayers):
         projectedIdxOffset=len(self.maskIdx)*nl
         indices,fractions=self.maskLayerCentreIdx(nl)
         for i in range(len(self.maskIdx)):
            extractionM[projectedIdxOffset+i,layerIdxOffsets[nl]+indices[i]]\
                  +=fractions[i]
      return extractionM 

   def trimIdx(self, concatenated=True):
      '''Return a concatenated (or not) index that when applied to a
      concatenated layer-vector, returns the illuminated layer-vector (the
      points that contribute to the projected masks) or the per-layer index
      into the layer rectangles.
      '''
      self.createLayerMasks()
      layerIdxOffsets=self.layerIdxOffsets()
      self.trimmingIdx=[]
      thisOffset=0
      for nl in range(self.nLayers):
         illuminatedLayer=self.layerMasks[nl].sum(axis=0).ravel()
         thisIdx=numpy.flatnonzero(illuminatedLayer)
         if concatenated:
            self.trimmingIdx+=(thisIdx+layerIdxOffsets[nl]).tolist()
         else:
            self.trimmingIdx.append( (thisOffset, thisIdx) )
            thisOffset+=len(thisIdx)
      if not concatenated:
         self.trimmingIdx.append( (thisOffset, []) )
      return self.trimmingIdx

   def maskInLayerIdx(self, layer, thisMask):
      '''Return an concatenated index that when applied to a concatenated
      layer-vector, returns the portion of the layer-vector specified by
      the provided 2D mask.
      '''
      self.createLayerMasks()
      if thisMask.shape[0]!=self.layerNpix[layer][0]\
            or thisMask.shape[1]!=self.layerNpix[layer][1]:
         raise ValueError("Mask doesn't match the layer size")
      layerIdxOffset=self.layerIdxOffsets()[layer]
      return (thisMask!=0).ravel().nonzero()[0]+layerIdxOffset

   def trimLayerExtractionMatrix(self):
      '''Define a layer extraction matrix, that extracts each projected
      mask from the trimmed concatenated layer-vector.
      '''
      return numpy.take( self.layerExtractionMatrix(), self.trimIdx(), axis=1 )

   def sumProjectedMatrix(self):
      '''Define a matrix that sums the projected mask per layer-vector.
      '''
      totalPts=len(self.maskIdx)*self.nLayers*self.nAzi
      summedPts=len(self.maskIdx)*self.nAzi
      sumProjM=numpy.zeros( [ summedPts, totalPts ], numpy.float64 )
      # pretty straightforward, just ones for each layer's projection
      sumProjIdx=(
             numpy.arange(self.nLayers)*summedPts\
            +numpy.arange(summedPts).reshape([-1,1])*(1+totalPts)
         ).ravel()
      sumProjM.ravel()[ sumProjIdx ]=1
      return sumProjM

   def sumCentreProjectedMatrix(self):
      '''Define a matrix that sums the centre projected mask per layer-vector.
      '''
      totalPts=len(self.maskIdx)*self.nLayers
      summedPts=len(self.maskIdx)
      sumProjM=numpy.zeros( [ summedPts, totalPts ], numpy.float64 )
      # pretty straightforward, just ones for each layer's projection
      sumProjIdx=(
             numpy.arange(self.nLayers)*summedPts\
            +numpy.arange(summedPts).reshape([-1,1])*(1+totalPts)
         ).ravel()
      sumProjM.ravel()[ sumProjIdx ]=1
      return sumProjM

class projectedModalBasis(geometry):
   modalBases=[]
   radialPowers=None
   angularPowers=None

   def __init__(self, layerHeights, zenAngles, azAngles, pupilMask,
         radialPowers, angularPowers, 
         starHeight=None, pixelScale=1, layerNpix=None, sparse=False ):
      geometry.__init__(self, layerHeights, zenAngles, azAngles, pupilMask,
         starHeigh, pixelScale, layerNpix)
      # for each layer, form an independent modal basis object
      assert self.createLayerMasks()
      modalBases=[ gradientOperator.modalBasis(
            thisMask, radialPowers, angularPowers, sparse )
            for thisMask in self.layerMasks ]

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
   geometry=geometry(
      numpy.array([0,1]),
      numpy.array([1]*nAzi),
      numpy.arange(nAzi)*2*numpy.pi*(nAzi**-1.0), mask,
      gsHeight )
   geometry.define()
   okay=geometry.createLayerMasks()
   if not okay:
      raise ValueError("Eek!")

   pg.figure()
   pg.gray()
   pg.subplot(2,2,1)
   pg.title("layer masks")
   pg.imshow( geometry.layerMasks[0].sum(axis=0), interpolation='nearest' )
   pg.subplot(2,2,2)
   pg.imshow( geometry.layerMasks[1].sum(axis=0), interpolation='nearest' )
   pg.subplot(2,2,3)
   pg.imshow( geometry.layerMasks[0].sum(axis=0)>0, interpolation='nearest' )

   # fix a mask for the upper layer
   projectedMask=(geometry.layerMasks[1].sum(axis=0)>0)*1.0
   edgeIdx=edgeDetector(projectedMask,clip=7)
   cds=lambda x : numpy.arange(x)-(x-1)/2.0
   centreOffset = (
      int(numpy.sign(geometry.centreIdx[1])\
         *(abs(geometry.centreIdx[1])%geometry.layerNpix[1,1])),
      int(geometry.centreIdx[1]/float(geometry.layerNpix[1,1]))
                  )
   radius=numpy.add.outer(
      centreOffset[0]+cds(geometry.layerNpix[1,0])**2.0,
      centreOffset[1]+cds(geometry.layerNpix[1,1])**2.0 )**0.5
   edgeRadii=radius.ravel()[edgeIdx]

   pg.subplot(2,2,4)
   pg.imshow( projectedMask, interpolation='nearest' )
   circle=pg.Circle( [(geometry.layerNpix[1,dirn]-1)/2.0 for dirn in (1,0)],
      radius=edgeRadii.min(),lw=2,ec='w',fill=False)
   patch=pg.gca().add_patch(circle)
   # now, if there are small holes, want to find the smallest and the largest
   # which is about 1/2 the layerNpix radius
   if (edgeRadii.min()/geometry.layerNpix[1].mean())<0.30:
      newEdgeRadii=edgeRadii[numpy.flatnonzero( edgeRadii>edgeRadii.min() )]
      circle2=pg.Circle( 
         [centreOffset[dirn]+(geometry.layerNpix[1,dirn]-1)/2.0
            for dirn in (1,0)],
         radius=newEdgeRadii.min(),lw=2,ls='dashed',ec='w',fill=False)
      patch2=pg.gca().add_patch(circle2)
      print("2ndary drawn")

   pg.draw()

   # try projection
   print("Projection matrix calcs...",end="")
   layerExM=geometry.layerExtractionMatrix()
   sumPrM=geometry.sumProjectedMatrix()
   trimIdx=geometry.trimIdx()
   sumLayerExM=numpy.dot( sumPrM, layerExM.take(trimIdx,axis=1) )
   print("(done)")

      # \/ random values as a substitute dataset
   random=[ numpy.random.uniform(-1,1,size=tS) for tS in geometry.layerNpix ]
   print("Input creation...",end="")
   randomA=[ numpy.ma.masked_array(random[i],
         geometry.layerMasks[i].sum(axis=0)==0) for i in (0,1) ]
   randomV=(1*random[0].ravel()).tolist()+(1*random[1].ravel()).tolist()
   randomExV=numpy.take( randomV, trimIdx )
   randomProjV=numpy.dot( sumLayerExM, randomExV )
   print("(done)")
   
      # \/ create an imagable per-projection array of the random values
   projectedRdmVA=numpy.ma.masked_array(
      numpy.zeros([5]+list(mask.shape),numpy.float64),
      (mask*numpy.ones([5,1,1]))==0, astype=numpy.float64)
   projectedRdmVA.ravel()[
      (geometry.maskIdx+(numpy.arange(0,5)*mask.shape[0]*mask.shape[1]).reshape([-1,1])).ravel() ]\
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
      numpy.ma.masked_array(numpy.zeros(geometry.layerNpix[i], numpy.float64),
         geometry.layerMasks[i].sum(axis=0)==0) for i in (0,1)]
            for j in range(len(recoveryM)) ]
   layerInsertionIdx=geometry.trimIdx(False)
   print("(done)")

   for j in range(len(recoveryM)):
      print("Type {0:d}".format(j+1))
      pg.figure()
      for i in range(geometry.nLayers):
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


