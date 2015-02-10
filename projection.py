"""ABBOT : Define a projection matrix and geometry for AO

Includes variable guidestar height (LGS) and non-one pixel scales
"""
#TODO::
#(1) Generalize members of class projection, for centre projected and
#  other azimuth-angle dependent projections.

from __future__ import print_function
from rounding import head,floor
import collections
import gradientOperator
import numpy
import version

def quadrantFractions( (v,h),s, stopOnFailure=True ):
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
   opStr=None
   if numpy.array(fracs).sum()<0.9999:
      opStr="Insufficient fractional subdivision"
   elif numpy.array(fracs).sum()>1.0001:
      opStr="Excess fractional subdivision"
   if opStr:
      opStr="[{0:f},{1:f},{2:f}] {3:s}".format(v,h,s,opStr)
      if stopOnFailure: raise RuntimeError(opStr)
      print(opStr)

   return (numpy.array(rcs).astype(numpy.int32),
         s**2.0*numpy.array(fracs).astype(numpy.float64),v0,h0)

class geometry(object):
   layerMasksFilledIn=False
   def _hs(self,x,y):
      if x==None or type(self.starHeights[y])==type(None):
         return self.pixelScales[y]
      else:
         return self.pixelScales[y]*(1-self.layerHeights[x]*(self.starHeights[y]**-1.0))

   def __init__(self, layerHeights, zenAngles, azAngles, pupilMasks,
         starHeights=None, pixelScales=1, layerNpix=None, raiseWarnings=True ):
      '''Layer heights in [metres],
         angles in [radians],
         Equal no. of zenAngles and azAngles required.
         starHeights [metres] can either be None (equiv. Inf), a constant, or a
            list with length equal to no. of zenAngles
            and the central projection assumed to be at infinity
         pixelScales [] can either be a constant, or a
            list with length equal to no. of zenAngles or zenAngles+1
            with the last entry being the scale of the central projection
            (so it can be different, but need not be)
        
      '''
      self.raiseWarnings=raiseWarnings # whether to halt or try to carry on
      self.nAzi=len(azAngles)
      self.nLayers=len(layerHeights)
      self.layerHeights=layerHeights
      self.zenAngles=zenAngles
      self.azAngles=azAngles
      self.pupilMasks=([pupilMasks]*(self.nAzi+1) if (
            type(pupilMasks)==numpy.ndarray and len(pupilMasks.shape)==2)
         else pupilMasks)
      if type(self.pupilMasks)==numpy.ndarray:
         self.pupilMasks=self.pupilMasks.tolist()
      for i,tPM in enumerate(self.pupilMasks):
         # x-check the masks are 2D arrays or can be coerced as such
         tPM=numpy.array(tPM) # will almost always work
         if not tPM.dtype in (int,float):
            warningStr="Incompatible dtype for mask {0:d}".format(i+1)
            raise RuntimeError(warningStr)
         if len(tPM.shape)!=2:
            warningStr="Wrong number of dimensions for mask {0:d}".format(i+1)
            raise RuntimeError(warningStr)
         if 1 in tPM.shape:
            warningStr="Really one dimensional? for mask {0:d}".format(i+1)
            if self.raiseWarnings: raise RuntimeError(warningStr)
            #
            print(warningStr)
##      assert sum([ len(thisPM.shape)==3 for thisPM in pupilMasks ])==self.nAzi,\
##            "Pupil masks must be 2D"
      if len(self.pupilMasks)==self.nAzi:
         # have not specified a pupilMask for the centre so assume the
         # first, and warn
         warningStr="No pupilMask was specified for the centre projection"
         if self.raiseWarnings: raise RuntimeError(warningStr)
         print(warningStr)
         #
         self.pupilMasks=[tPM for tPM in self.pupilMasks] + [self.pupilMasks[0]]
      self.npixs=numpy.array([ thisPM.shape for thisPM in self.pupilMasks ])
      self.starHeights=[None]*self.nAzi if starHeights==None else (
         [starHeights]*self.nAzi if not
               isinstance( starHeights, collections.Iterable)
         else starHeights )
      if len(self.starHeights)==self.nAzi:
         # have not specified a height for the centre so assume a
         # sensible value
         self.starHeights=list(self.starHeights)+[None]
      elif len(self.starHeights)!=self.nAzi+1:
         raise ValueError("Wrong number of elements in starHeights, "+
               " got {0:d} but expected {1:d} (or less one)".format(
                  len(self.starHeights),self.nAzi+1 ))
      # need at least 10% of the guidestar altitude between the final layer and
      # it \/
      if max(self.starHeights[:-1])!=None:
         starHeightMinDistance=0.1 
         if 1-self.layerHeights[-1]/(
                  max(self.starHeights[:-1])/starHeightMinDistance
               )<starHeightMinDistance:
            raise ValueError("Must have the guide star at a higher altitude")
      self.pixelScales=[pixelScales]*self.nAzi if\
         not isinstance( pixelScales, collections.Iterable ) else pixelScales
      if len(self.pixelScales)==self.nAzi:
         # have not specified a pixel scale for the centre so assume a
         # sensible value
         self.pixelScales=list(self.pixelScales)+[1]
      elif len(self.pixelScales)!=self.nAzi+1:
         raise ValueError("Wrong number of elements in pixelScales, "+
               " got {0:d} but expected {1:d} (or less one)".format(
                  len(self.pixelScales),self.nAzi+1 ))

      self.layerNpix=(None if not isinstance( layerNpix, collections.Iterable)
                           else numpy.array(layerNpix) )
      self.maskIdxs=[ numpy.array(thisPM).ravel().nonzero()[0]
            for thisPM in self.pupilMasks ]
      self.define()

   def define(self):
      '''Define projection geometry.
      '''
      # calculate the x,y offsets of the projected aperture on the layers
      # including the central projection
      self.offsets=numpy.zeros( [ self.nLayers,self.nAzi+1,2 ], numpy.float64 )
      for i in range(self.nLayers):
         for j in range(self.nAzi+1):
            self.offsets[i,j]=[ float(self.layerHeights[i])*\
                  numpy.round(tf(0 if j==self.nAzi else self.azAngles[j]),6)*\
                  (0 if j==self.nAzi else self.zenAngles[j])
                     for tf in (numpy.sin,numpy.cos) ]

         # \/ for each mask projection, calculate the corner coordinate,
         #  for the minimum and maximum positions, relative to the centre
         #  of the array
      self.cornerCoordinates=numpy.array([ [
            [ dirn*(self._hs(i,j)*self.npixs[j])/2.0+self.offsets[i,j]
               for dirn in (+1,-1) ]
                  for j in range(self.nAzi+1) ]
                     for i in range(self.nLayers) ])
         # \/ the minimum and maximum array coordinates of the edges of
         #  the centred mask
      self.maxMinSizes=numpy.array([
            self.cornerCoordinates[:,:,0].max(axis=1),
            self.cornerCoordinates[:,:,1].min(axis=1) ])
         # To place a mask into the array, the vector from the corner of
         # a mask to the corner of the array is required.
         # Everything is known except the corner of the array to the
         # centre of the layer vector, so calculate this now.
         # Note that the variables named 'layer' represent the array and the
         # actual layer (boundless) isn't formally called anything.
      self.layerMaskCorners=self.cornerCoordinates[:,:,1]\
            -self.maxMinSizes[1].reshape([self.nLayers,1,2])
         # now the relative position with the layer, and possibly
         # also compute the size of the layer too
      expectedLayerNpix=numpy.ceil( (self.maxMinSizes[0]-self.maxMinSizes[1])
                                 ).astype(numpy.int32)
      if self.layerNpix==None:
         self.layerNpix=expectedLayerNpix
      elif self.layerNpix.shape!=(self.nLayers,2):
         raise ValueError("Wrong layerNpix shape")
      else:
         for i,tlNP in enumerate(self.layerNpix): 
            if False in [ tlNP[j]>=expectedLayerNpix[i][j] for j in (0,1) ]:
               raise ValueError("LayerNpix is not compatible with the "
                     "requirement:"+str(expectedLayerNpix))
            # \/ fix up the relative position within the layer
            self.layerMaskCorners[i]+=\
                  (self.layerNpix[i]-expectedLayerNpix[i])/2.
        
      self.layerMasks=[
         numpy.zeros([self.nAzi+1]+self.layerNpix[i].tolist(),numpy.float32)
            for i in range(self.nLayers) ]
      
   def maskLayerCentreIdx(self, layer, flat=0):
      '''For a layer, return the indices per mask pixel and their fractions
      for centre projection (zero zenith angle).
      '''
      return self.maskLayerIdx( layer, -1, flat )
             
   def maskLayerIdx(self, layer, azi, flat=0):
      '''For a layer, return the indices per mask pixel and their fractions
      for that azimuth.
      '''
      return (self._maskLayerIdx(layer, azi, self.offsets[layer,azi], flat))
   
   def _maskLayerIdx(self, layer, azi, offsets, flat):
      '''Generic: return for a layer the offset mask position. Use
      maskLayerCentreIdx or maskLayerIdx instead of this.
      '''
      self.maskCoords=numpy.array([
               self.maskIdxs[azi]//self.npixs[azi][0],
               self.maskIdxs[azi]%self.npixs[azi][0]
            ])*self._hs(layer,azi)\
            +( self._hs(layer,azi)-1 )/2.0\
            +self.layerMaskCorners[layer,azi].reshape([2,1])

      indices=[] ; fractions=[]
      for i in range(self.maskIdxs[azi].shape[0]):
         rcs,fracs,v0,h0=quadrantFractions( self.maskCoords.T[i],
               self._hs(layer,azi) )
         thisidx=(rcs[:,0]+v0)*self.layerNpix[layer,1]+(h0+rcs[:,1])
         if not flat:
            indices.append(thisidx.astype('i'))
            fractions.append(fracs)
         else:
            indices+=list(thisidx)
            fractions+=list(fracs)
      if not flat:
         return (indices,fractions)
      else:
         return (numpy.array(indices,'i'), numpy.array(fractions))

   def createLayerMasks(self):
      '''Create the masks for each layer by imposing the projected pupil
      masks for the given azimuth. Only once.
      '''
      # because this function can be slow, check if it can be avoided
      if self.layerMasksFilledIn: return True 
      for nl in range(self.nLayers):
         for na in range(self.nAzi+1):
            indices,fractions=self.maskLayerIdx(nl,na,flat=1)
            valid=numpy.flatnonzero( (indices>-1)*
                     (indices<self.layerNpix[nl,0]*self.layerNpix[nl,1]) )
            if len(valid)!=len(indices):
               warningStr="Eurgh. Something horid has happened"+\
                     ";nl={0:d},na={1:d}".format(nl,na)
               if self.raiseWarnings: raise RuntimeError(warningStr)
               print(warningStr)
               self.layerMasksFilledIn=False
#            self.layerMasks[nl][na].ravel()[ indices[valid] ]+=fractions[valid]
# /\ doesn't work because indices has repeat values so have to do by hand
# \/ but this can be slow: probably need a C module for speed
            blank=numpy.zeros(self.layerNpix[nl],numpy.float32)
            for i in valid:
               self.layerMasks[nl][na].ravel()[indices[i]]+=fractions[i]
       
      self.layerMasksFilledIn=True # all is okay
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
   def __init__(self, layerHeights, zenAngles, azAngles, pupilMasks,
         starHeights=None, pixelScales=1, layerNpix=None, raiseWarnings=True ):
      geometry.__init__(self, layerHeights, zenAngles, azAngles, pupilMasks,
         starHeights, pixelScales, layerNpix, raiseWarnings)
         # \/ skip the last one, the centre projected mask
      self.maskIdxCumSums=numpy.cumsum( 
            [ len(thisMI) for thisMI in self.maskIdxs[:-1] ])
   def layerExtractionMatrix(self):
      '''Define a layer extraction matrix, that extracts each projected
      mask from the concatenated layer-vector, ignoring the central
      projected mask.
      '''
      layerIdxOffsets=self.layerIdxOffsets()
            # \/ /\ only time [-1] used is for the total size
      extractionM=numpy.zeros(
            (self.maskIdxCumSums[-1]*self.nLayers, layerIdxOffsets[-1]),
            numpy.float64 )
      # matrix can be filled in by saying:
      # for each layer,
      #   for each azimuth angle,
      #     find the indices in the layer and the fraction for each
      #     these represent the entries in the matrix
      for nl in range(self.nLayers):
         for na in range(self.nAzi):
            projectedIdxOffset=( self.maskIdxCumSums[-1]*nl+(
                  0 if na==0 else self.maskIdxCumSums[na-1] ) )
            indices,fractions=self.maskLayerIdx(nl,na)
            for i in range(len(self.maskIdxs[na])):
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
          [len(self.maskIdxs[-1])*self.nLayers, layerIdxOffsets[-1]], numpy.float64 )
      # matrix can be filled in by saying:
      # for each layer,
      #   for each azimuth angle,
      #     find the indices in the layer and the fraction for each
      #     these represent the entries in the matrix
      for nl in range(self.nLayers):
         projectedIdxOffset=len(self.maskIdxs[-1])*nl
         indices,fractions=self.maskLayerCentreIdx(nl)
         for i in range(len(self.maskIdxs[-1])):
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
      totalPts=self.maskIdxCumSums[-1]*self.nLayers
      summedPts=self.maskIdxCumSums[-1]
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


