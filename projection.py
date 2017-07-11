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
         if not tPM.dtype in (
               int,float,numpy.int32,numpy.int16,numpy.int64,numpy.int,
               numpy.float32,numpy.float64):
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
         starHeights=None, pixelScales=1, layerNpix=None, raiseWarnings=True,
         sparse=False ):
      geometry.__init__(self, layerHeights, zenAngles, azAngles, pupilMasks,
         starHeights, pixelScales, layerNpix, raiseWarnings)
         # \/ skip the last one, the centre projected mask
      self.maskIdxCumSums=numpy.cumsum( 
            [ len(thisMI) for thisMI in self.maskIdxs[:-1] ])
      self.sparse=sparse

   def layerExtractionMatrix(self,trimmed=False):
      '''Define a layer extraction matrix, that extracts each projected
      mask from the concatenated layer-vector, ignoring the central
      projected mask.
      '''
      matrixshape = [ self.maskIdxCumSums[-1]*self.nLayers, None ]
      if trimmed:
         trimIdx=self.trimIdx()
      layerIdxOffsets = self.layerIdxOffsets()
      matrixshape[1] = layerIdxOffsets[-1] if not trimmed else len(trimIdx)
            # \/ /\ only time [-1] used is for the total size
      if not self.sparse:
         extractionM = numpy.zeros( matrixshape, numpy.float64 )
      else:
         import scipy.sparse, scipy.sparse.linalg
         extractionM = { 'ij':[[],[]], 'data':[] }
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
               ij0=[ projectedIdxOffset+i ]*len(fractions[i])
               ij1=( layerIdxOffsets[nl]+indices[i] ).tolist()
               if trimmed:
                  ij1=numpy.searchsorted(trimIdx,ij1).tolist()
               if not self.sparse:
                  extractionM[ ij0[0], ij1 ]+=fractions[i]
               else:
                  extractionM['ij'][0]+=ij0
                  extractionM['ij'][1]+=ij1
                  extractionM['data']+=list(fractions[i])
      if self.sparse:
         extractionM=scipy.sparse.csr_matrix(
               (extractionM['data'], extractionM['ij']),
               matrixshape )
      return extractionM

   def layerCentreProjectionMatrix(self,trimmed=False):
      '''Define a layer extraction matrix, that extracts a centrally
      projected mask through the concatenated layer-vector.
      '''
      matrixshape = [ len(self.maskIdxs[-1])*self.nLayers, None ]
      if trimmed:
         trimIdx=self.trimIdx()
      layerIdxOffsets = self.layerIdxOffsets()
      matrixshape[1] = layerIdxOffsets[-1] if not trimmed else len(trimIdx)
            # \/ /\ only time [-1] used is for the total size
      if not self.sparse:
         extractionM=numpy.zeros( matrixshape, numpy.float64 )
      else:
         import scipy.sparse, scipy.sparse.linalg
         extractionM = { 'ij':[[],[]], 'data':[] }
      # matrix can be filled in by saying:
      # for each layer,
      #   for each azimuth angle,
      #     find the indices in the layer and the fraction for each
      #     these represent the entries in the matrix
      for nl in range(self.nLayers):
         projectedIdxOffset=len(self.maskIdxs[-1])*nl
         indices,fractions=self.maskLayerCentreIdx(nl)
         for i in range(len(self.maskIdxs[-1])):
            ij0=[ projectedIdxOffset+i ]*len(fractions[i])
            ij1=( layerIdxOffsets[nl]+indices[i] ).tolist()
            if trimmed:
               ij1=numpy.searchsorted(trimIdx,ij1).tolist()
            if not self.sparse:
               extractionM[ ij0[0], ij1 ]+=fractions[i]
            else:
               extractionM['ij'][0]+=ij0
               extractionM['ij'][1]+=ij1
               extractionM['data']+=list(fractions[i])
      if self.sparse:
         extractionM=scipy.sparse.csr_matrix(
               (extractionM['data'], extractionM['ij']),
               matrixshape )
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

#(redundant?)   def trimLayerExtractionMatrix(self):
#(redundant?)      '''Define a layer extraction matrix, that extracts each projected
#(redundant?)      mask from the trimmed concatenated layer-vector.
#(redundant?)      '''
#(redundant?)      return numpy.take( self.layerExtractionMatrix(), self.trimIdx(), axis=1 )

   def _sumProjectedMatrix(self, totalPts, summedPts):
      '''Define a matrix that sums a specified set of points per layer-vector.
      '''
      matrixshape = [ summedPts, totalPts ]
      if not self.sparse:
         sumProjM=numpy.zeros( matrixshape, numpy.float64 )
      else:
         import scipy.sparse, scipy.sparse.linalg
         sumProjM = { 'ij':[[],[]], 'data':[] }
      # pretty straightforward, just ones for each layer's projection
      if self.sparse:
         for i in range(summedPts):
            sumProjM['ij'][0]+=[i]*self.nLayers
            sumProjM['ij'][1]+=range(i,self.nLayers*summedPts,summedPts)
         sumProjM=scipy.sparse.csr_matrix(
               ([1]*totalPts, sumProjM['ij']), matrixshape )
      else: 
         sumProjIdx=(
                numpy.arange(self.nLayers)*summedPts\
               +numpy.arange(summedPts).reshape([-1,1])*(1+totalPts)
            ).ravel()
         sumProjM.ravel()[ sumProjIdx ]=1
      return sumProjM

   def sumProjectedMatrix(self):
      '''Define a matrix that sums the centre projected mask per layer-vector.
      '''
      totalPts=self.maskIdxCumSums[-1]*self.nLayers
      summedPts=self.maskIdxCumSums[-1]
      return self._sumProjectedMatrix( totalPts, summedPts )

   def sumCentreProjectedMatrix(self):
      '''Define a matrix that sums the centre projected mask per layer-vector.
      '''
      totalPts=len(self.maskIdxs[-1])*self.nLayers
      summedPts=len(self.maskIdxs[-1])
      return self._sumProjectedMatrix( totalPts, summedPts )

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
   #
   # for future unittest compatibility, use assert statements in this test code
   #
   def test_geometry(ip,success,failure):
      global thisGeom,thisGeom2x
      (nAzi,gsHeight,mask)=ip
      thisGeom=geometry(
            numpy.array([0,1]),
            numpy.array([1]*nAzi),
            numpy.arange(nAzi)*2*numpy.pi*(nAzi**-1.0), mask,
            gsHeight )
      thisGeom.define()
      assert thisGeom.createLayerMasks(), "thisGeom createLayerMasks failed"
      success+=1
      thisGeom2x=projection(
            numpy.array([0,1]),
            numpy.array([1]*nAzi),
            numpy.arange(nAzi)*2*numpy.pi*(nAzi**-1.0), mask,
            gsHeight,
            pixelScales=2)
      thisGeom2x.define()
      assert thisGeom2x.createLayerMasks(), "thisGeom2x createLayerMasks failed"
      success+=1

      summedSfcLayerMasks=thisGeom.layerMasks[0][:-1].sum(axis=0)
      summedSfcLayerMasks2x=thisGeom2x.layerMasks[0][:-1].sum(axis=0)
      summedAltLayerMasks=thisGeom.layerMasks[1][:-1].sum(axis=0)
      # test: are there the right number (and placement) of surface layer masks
      assert ( summedSfcLayerMasks==(nAzi)*mask ).all(),\
            "total of sfc layer masks is wrong"
      success+=1
      assert ( summedSfcLayerMasks.sum()==summedSfcLayerMasks2x.sum()*0.25 
            ).all(), "scaling of sfc layer masks is wrong"
      success+=1
      # test: are there the right number (and scaling) of altitude layer masks
      altScaling=(gsHeight-1)**2.0*gsHeight**-2.0
      assert abs( summedAltLayerMasks.sum()-
          summedSfcLayerMasks.sum()*altScaling )/summedSfcLayerMasks.sum()<1e-5\
          , "total of upper layer masks is wrong"
      success+=1
      # test: are there the right placement of altitude layer masks
      cds=[ numpy.arange(tn)-(tn-1.)/2. for tn in thisGeom.layerNpix[1] ]
      cds[0]=cds[0].reshape([-1,1])
      centreCoordinate=[
            (thisGeom.layerMasks[1][-1]*tcds).sum()*mask.sum()**-1.
               for tcds in cds ]
      for i in range(nAzi):
         c=numpy.array([
               (thisGeom.layerMasks[1][i]*cds[j]).sum()\
                     /(mask.sum()*altScaling)-
               centreCoordinate[j] for j in (0,1) ])
         ec=thisGeom.cornerCoordinates[1][i].mean(axis=0)
         assert (abs(c-ec)<1e-2).all(),\
               "placement of upper layer mask {0:d} is wrong".format(i)
         success+=1
      return success,failure
   
   def test_projection(ip,success,failure):
      global thisProj,layerExM,sumPrM,sumLayerExM
      (nAzi,gsHeight,mask)=ip
      (thisProj,layerExM,layerExUTM,sumPrM,sumCentPrM,sumLayerExM,layerCentExM,
         sumLayerCentExM)={},{},{},{},{},{},{},{}
      for sparse in (0,1):
         thisProj[sparse]=projection(
               numpy.array([0,1]),
               numpy.array([1]*nAzi),
               numpy.arange(nAzi)*2*numpy.pi*(nAzi**-1.0), mask,
               gsHeight,
               sparse=sparse )
         thisProj[sparse].define()
         okay=thisProj[sparse].createLayerMasks()
         if not okay:
            raise ValueError("Eek!")
         # try projection
         print("{0:s}:Projection matrix calcs...".format(
               "sparse" if sparse else "dense"), end="")
         layerExUTM[sparse]=thisProj[sparse].layerExtractionMatrix(0)
         #
         layerExM[sparse]=thisProj[sparse].layerExtractionMatrix(1)
         sumPrM[sparse]=thisProj[sparse].sumProjectedMatrix()
         layerCentExM[sparse]=thisProj[sparse].layerCentreProjectionMatrix(1)
         sumCentPrM[sparse]=thisProj[sparse].sumCentreProjectedMatrix()
         #
         sumLayerExM[sparse]=sumPrM[sparse].dot( layerExM[sparse] )
         sumLayerCentExM[sparse]=sumCentPrM[sparse].dot( layerCentExM[sparse] )
         print("(done)")

      # TEST: basic matrices comparison between sparse and dense
      try:
         assert ( numpy.array( layerExM[1].todense() )-layerExM[0] ).var()==0,\
            "layerExM sparse!=dense"
      except:
         failure+=1
         print(sys.exc_info()[1])
      else:
         success+=1
      try:
         assert (numpy.array(layerCentExM[1].todense())-layerCentExM[0]
               ).var()==0, "layerExM sparse!=dense"
      except:
         failure+=1
         print(sys.exc_info()[1])
      else:
         success+=1
      try:
         assert ( numpy.array( sumPrM[1].todense() )-sumPrM[0] ).var()==0,\
            "sumPrM sparse!=dense"
      except:
         failure+=1
         print(sys.exc_info()[1])
      else:
         success+=1
      try:
         assert (numpy.array(sumCentPrM[1].todense())-sumCentPrM[0]).var()==0,\
            "sumCentPrM sparse!=dense"
      except:
         failure+=1
         print(sys.exc_info()[1])
      else:
         success+=1
      try:
         assert ( numpy.array( sumLayerExM[1].todense() )-sumLayerExM[0] 
               ).var()==0, "sumLayerExM sparse!=dense"
      except:
         failure+=1
         print(sys.exc_info()[1])
      else:
         success+=1
      try:
         assert ( numpy.array(sumLayerCentExM[1].todense())-sumLayerCentExM[0] 
               ).var()==0, "sumLayerCentExM sparse!=dense"
      except:
         failure+=1
         print(sys.exc_info()[1])
      else:
         success+=1
      try:
         assert ( layerExUTM[0].take( thisProj[0].trimIdx(), axis=1 )-
               layerExM[0] ).var()==0, "layerExM inbuilt trimming failed"
      except:
         failure+=1
         print(sys.exc_info()[1])
      else:
         success+=1
      # TEST: input means 
      tilts=lambda s : numpy.add.outer(
            numpy.arange(-s[0]/2,s[0]/2),
            numpy.arange(-s[1]/2,s[1]/2) )
      quadratic=lambda s : numpy.add.outer(
            numpy.arange(-s[0]/2,s[0]/2)**2.0,
            numpy.arange(-s[1]/2,s[1]/2)**2.0 )
      ip=[ quadratic(tS) for tS in thisProj[0].layerNpix ]
         # now, take each mask as projected and compute the sum of the mean
         # of the projected tilt and this is our comparison
      ipV=(ip[0].ravel()).tolist()+(ip[1].ravel()).tolist()
      #           
      ipExV=numpy.take( ipV, thisProj[0].trimIdx() )
      ipProjV={
               0: numpy.dot( sumLayerExM[0], ipExV ),
               1: numpy.array( sumLayerExM[1].dot( ipExV ))
            }
      ipCentProjV={
               0: numpy.dot( sumLayerCentExM[0], ipExV ),
               1: numpy.array( sumLayerCentExM[1].dot( ipExV ))
            }
      try:
         test=(ipProjV[0]-ipProjV[1]).var()
         assert test<1e-10, "ipProjV, var{sparse-dense}>1e-10:"+str(test)
      except:
         failure+=1
         print(sys.exc_info()[1])
      else:
         success+=1
      try:
         test=(ipCentProjV[0]-ipCentProjV[1]).var()
         assert test<1e-10, "ipCentProjV, var{sparse-dense}>1e-10"+str(test)
      except:
         failure+=1
         print(sys.exc_info()[1])
      else:
         success+=1
      lmIdx=[ thisProj[0].maskInLayerIdx(
            i,thisProj[0].layerMasks[i].sum(axis=0)) for i in (0,1) ]
      maskDerivedMean=[ numpy.take(ipV,lmIdx[i]).sum() for i in (0,1) ]
      expectedIpMean=numpy.array(
            [ (ip[j]*(thisProj[0].layerMasks[j][:].sum(axis=0)!=0)).sum()
                  for j in (0,1) ])
      try:
         assert ( maskDerivedMean==expectedIpMean ).all(),\
               "maskInLayerIdx discrepancy"
      except:
         failure+=1
         print(sys.exc_info()[1])
      else:
         success+=1
      #
      expectedIpPerMask=numpy.array([
            [ (ip[j]*thisProj[0].layerMasks[j][i]).sum()
                  for j in (0,1) ] for i in range(nAzi+1)])
      maskL=mask.sum()
      for i in range(nAzi):
         try:
            test=( expectedIpPerMask[i].sum()-
                  ipProjV[0][maskL*i:maskL*(i+1)].sum() )
            assert abs(test)<1e-3,\
                ("expectedIpPerMask[{0:d}]-that from projected"+
                 ", summed vector>1e-3/{1:f}").format(i,test)
         except:
            failure+=1
            print(sys.exc_info()[1])
         else:
            success+=1
      try:
         test=( expectedIpPerMask[-1].sum()-ipCentProjV[0].sum() )
         assert abs(test)<1e-3,\
             ("expectedIpPerMask[-1]-that from centre projected"+
              ", summed vector>1e-3/{1:f}").format(test)
      except:
         failure+=1
         print(sys.exc_info()[1])
      else:
         success+=1
      #
      return success,failure

   import datetime, sys
   titleStr="projection.py, automated testing"
   print("\n{0:s}\n{1:s}\n".format(titleStr,len(titleStr)*"^"))
   print("BEGINS:"+str(datetime.datetime.now()))
   #
   rad=10 # note, array pixels
   nAzi=5
   gsHeight=3 # note, unitless
   #
   circ = lambda b,r : (numpy.add.outer(
         (numpy.arange(b)-(b-1.0)/2.0)**2.0,
         (numpy.arange(b)-(b-1.0)/2.0)**2.0 )**0.5<=r).astype( numpy.int32 )
   mask=circ(rad,rad/2)-circ(rad,rad/2*0.25)
   #
   success,failure=test_geometry([nAzi,gsHeight,mask],0,0)
   total=success+failure
   succeeded,failed=success,failure
   print("SUMMARY:geometry: {0:d}->{1:d} successes and {2:d} failures".format(
         total, succeeded, failed))
   #
   success,failure=test_projection([nAzi,gsHeight,mask],0,0)
   total=success+failure
   succeeded,failed=success,failure
   print("SUMMARY:projection: {0:d}->{1:d} successes and {2:d} failures".format(
         total, succeeded, failed))
   print("ENDS:"+str(datetime.datetime.now()))
   sys.exit( failed>0 )
