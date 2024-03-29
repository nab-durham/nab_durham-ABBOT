"""ABBOT : General non-complete modal basis for arbitrary shaped apertures.

For modal filtering, create modal functions which are limited (so non-complete) 
which are not guaranteed to be either orthogonal or normalized.
Currently based on Zernike-like forms and sinusoidal-like forms.
"""
from __future__ import print_function
import gradientOperator
import numpy
import sys
if sys.version_info.major==2:
   from exceptions import NotImplementedError # Py2k
else:
   pass # Py3k

class modalBasis(gradientOperator.geometryType1):
   '''Calculate modal basis based on type 1 geometry.
   '''
   op=None
   numberSubaps=None
   numberPhases=None
   sparse=False

   def __init__( self, pupilMask,
         sparse=False, compute=True, verbose=False, orthonormalize=True,
         truncate=True ):
      raise NotImplementedError("This member must be overloaded")
      self._initialize( pupilMask, compute, orthonormalize, verbose )
   
   def _initialize( self, pupilMask, compute, orthonormalize, verbose ):
      gradientOperator.geometryType1.__init__(self,None,pupilMask)
      if compute:
         self.calculateFactors()
         self.calculatemodalFunctions(orthonormalize,verbose)

   def calculateFactors(self):
      raise NotImplementedError("This member must be overloaded")

   def calculatemodalFunctions(self, orthonormalize=True ,verbose=False):
      raise NotImplementedError("This member must be overloaded")
   
      
class polySinRadAziBasisType1(modalBasis):
   '''upto the specified
   order of radial terms such that they are orthogonal.
   NB: this code is only meant for the linear and quadratic terms,
   perhaps it ought be replaced with Zernike-like terms defined over the
   aperture to make it more generic.'''
   def __init__( self, pupilMask, radialPowers=[0], angularPowers=[0],
         sparse=False, compute=True, verbose=False, orthonormalize=True):
      self.radialPowers=radialPowers
      self.angularPowers=angularPowers
      self._initialize( pupilMask, compute, orthonormalize, verbose )
   
   def calculateFactors(self):
      self.cds=[ self.illuminatedCornersIdx//self.n_[0]-(self.n_[0]-1)/2.0
              ,self.illuminatedCornersIdx%self.n_[1]-(self.n_[1]-1)/2.0 ]
      self.cds=[ self.cds[i]*(self.n_[i]-1.0)**-1.0 for i in (0,1) ]
      self.r=(self.cds[0]**2+self.cds[1]**2)**0.5
      self.rpower=lambda p : numpy.power(self.r,p)
      self.ang=numpy.arctan2(self.cds[0],self.cds[1])
      self.angcos=lambda n : numpy.cos(n*self.ang)
      self.angsin=lambda n : numpy.sin(n*self.ang)


   def modalFunction(self,rPower,angPower,angType):
      '''rPower, int
         angPower, int
         angType, 0=sin,1=cos
      '''
      tmf=self.rpower(rPower)*(
         angType*self.angsin(angPower)+
         (1-angType)*self.angcos(angPower)
         )
      return tmf*(tmf**2.0).sum()**-0.5

   def calculatemodalFunctions(self, orthonormalize=True ,verbose=False):
      self.modalFunctions=[]
      for tr in self.radialPowers:
         for ta in range(tr%2,min(len(self.angularPowers),tr+1),2): # Py3k
            for tt in range(1+(ta>0)*1): # Py3k
               if verbose: print(tr,ta,tt)
               self.modalFunctions.append(self.modalFunction(tr,ta,tt))
      self.modalFunctions=numpy.array(self.modalFunctions)
      if orthonormalize and len(self.modalFunctions)>1:
         self.mfOrth=limitDP(4,self.modalFunctions.dot(self.modalFunctions.T))
            # \/ find orthogonal combination of basis
         self.s,self.v=numpy.linalg.svd(self.mfOrth,full_matrices=1)[1:]
         self.orthomodalFunctions=\
            self.v.dot(self.modalFunctions)*(self.s**-0.5).reshape([-1,1]) 
         self.omfOrth=limitDP(4,
               self.orthomodalFunctions.dot(self.orthomodalFunctions.T))

class FourierModalBasisType1(modalBasis):
   '''Based on type 1 geometry, calculate modal basis with complete
   frequency coverage in both orthogonal directions.
   Ignores piston and the highest frequency term.
   '''
   def __init__( self, pupilMask,
         sparse=False, compute=True, verbose=False, orthonormalize=True,
         truncate=True ):
      gradientOperator.geometryType1.__init__(self,None,pupilMask)
      if compute:
         self.calculateFactors()
         self.calculatemodalFunctions(orthonormalize,verbose,truncate)
   
   def calculateFactors(self):
      self.cds=[ numpy.arange(self.n_[ii])*self.n_[ii]**-1.0 for ii in (0,1) ]

   def calculatemodalFunctions(self, orthonormalize=True ,verbose=False,
         truncate=True):
      self.modalFunctions=[]
      for ty in range( self.n_[0]/2 ):
         for tx in range(-self.n_[1]/2, self.n_[1]/2 ):
            if ty==0 and tx==0:
               continue # skip piston
            if ty==self.n_[0]/2-1 and tx==self.n_[1]/2-1:
               continue # skip waffle
            for trigfn in (numpy.cos,numpy.sin):
               self.modalFunctions.append(
                  trigfn( 2*numpy.pi*numpy.add.outer(
                        ty*self.cds[0], tx*self.cds[1] )
                        ).ravel()[self.illuminatedCornersIdx] )
                  
      self.modalFunctions=numpy.array(self.modalFunctions)
      if orthonormalize and len(self.modalFunctions)>1:
         self.mfOrth=limitDP(4,self.modalFunctions.dot(self.modalFunctions.T))
            # \/ find orthogonal combination of basis
         self.s,self.v=numpy.linalg.svd(self.mfOrth,full_matrices=1)[1:]
         self.orthomodalFunctions=\
            self.v.dot(self.modalFunctions)*(self.s**-0.5).reshape([-1,1]) 
         self.omfOrth=limitDP(4,
               self.orthomodalFunctions.dot(self.orthomodalFunctions.T))
         validOmfIdx=self.omfOrth.diagonal().nonzero()[0]
         if truncate and len(validOmfIdx)>self.numberPhases:
            validOmfIdx=validOmfIdx[:self.numberPhases]
            if verbose: print("FourierModalBasisType1:truncating")
         self.omfOrth=self.omfOrth[:validOmfIdx[-1],:validOmfIdx[-1]]
         self.orthomodalFunctions=self.orthomodalFunctions[validOmfIdx]

class KosambiKarhunenLoeveModalBasisType1(modalBasis):
   '''Use a KKL decomposition via SVD to generate bi-orthogonal modal functions.
   Requires a covariance matrix.
   truncate : explicitly, remove piston and waffle.
   orthonormalize : explicitly, check the orthonormalization.
   verbose : output information to stdout.
   compute : create the modes upon instantiation.
   sparse : ignored.
   '''
   def __init__( self, covarianceMatrix,
         sparse=False, compute=True, verbose=False, orthonormalize=True,
         truncate=True, precision=1e-9 ):
      self.covarianceMatrix = covarianceMatrix
      gradientOperator.geometryType1.__init__(self,None,covarianceMatrix.mask)
      self.pistonMode = numpy.empty( self.numberPhases )
      self.pistonMode[:] = ( self.numberPhases )**-0.5
      self.waffleMode = numpy.empty( self.numberPhases )
      self.waffleMode = (
            self.illuminatedCornersIdx//self.n_[1]+\
            self.illuminatedCornersIdx%self.n_[1]
         )%2.0 # force as a f.p. vector
      self.waffleMode -= self.waffleMode.mean()
      self.waffleMode /= (self.waffleMode**2.0).sum()**0.5
      #
      self.precision = precision
      self.nFunctions = None
      if compute:
         self.calculatemodalFunctions(orthonormalize,verbose,truncate)

   def calculatemodalFunctions(self, orthonormalize=True ,verbose=False,
         truncate=True):
      u,s,vT = numpy.linalg.svd( self.covarianceMatrix, full_matrices=1 )
         # \/ since covarianceMatrix could be non-symmetrical, use the left-singular vectors
         #  however only those with sufficient amplitude need be considered, we may
         #  truncate the quantity
      self.nFunctions = sum( (s[0]*self.precision)<=s )
      if verbose and self.nFunctions!=self.numberPhases:
         print("# of functions = {:d}".format( self.nFunctions ))
      self.modalFunctions = numpy.empty(
            [self.nFunctions, self.numberPhases], self.covarianceMatrix.dtype )
      for i in range(self.nFunctions):
         thisV = u.T[i]
         self.modalFunctions[i] = thisV
      if truncate:
         # remove unwanted modes and then re-normalize
            # \/ remove
         for thisMode in self.pistonMode, self.waffleMode:
            self.modalFunctions -=\
                  self.modalFunctions.dot( thisMode ).reshape( [-1,1] )\
                  *thisMode
            # \/ renormalize
         self.modalFunctions /= ( 
               (self.modalFunctions**2.0).sum(axis=1)**0.5 ).reshape([-1,1])
      if orthonormalize:
            # this should be true by default so it is verified here
         if abs(
               self.modalFunctions.dot(self.modalFunctions.T).diagonal().mean()
               -1 )>1e-6:
            print("WARNING: diagonal is not sufficiently one")
         if (
               self.modalFunctions.dot(self.modalFunctions.T)-numpy.identity(self.nFunctions)
               ).std()>1e-6:
            print("WARNING: off-diagonal is not sufficiently zero")

# ------------------------------

if __name__=="__main__":
   print("What is this?")
   print("Test modal basis filtering and split modal reconstruction")
   print()

   import numpy
   import Zernike
   import gradientOperator
   import matplotlib.pyplot as pyp
   import commonSeed
   import sys
   import kolmogorov
   import time
   numpy.random.seed(int(time.time()%1234))

   # -- configuration begins -----
   #
   baseSize=20
   mBidxs=[(1, 1, 1), (1, 1, 0), (1, 2, 1), (1, 3, 1), (1, 3, 0)]
   mask=Zernike.anyZernike(1,baseSize,baseSize/2,ongrid=1)\
         -Zernike.anyZernike(1,baseSize,baseSize/2/7.0,ongrid=1)
##
##   mask[baseSize/2]=0   # /
##   mask[:,baseSize/2]=0 # \ crude spider
##
#(redundant?)  modalPowers={'r':[1,2],'ang':[1]}
   #
   # -- configuration ends -------

   
   mask=mask.astype(numpy.int32)
   nMask=int(mask.sum())

   gradOp=gradientOperator.gradientOperatorType1(pupilMask=mask)
   gradM=gradOp.returnOp()

   thisModalBasis=polySinRadAziBasisType1( mask, [],[], orthonormalize=0, verbose=1 )
   mBs=numpy.array([
      thisModalBasis.modalFunction( tmBidx[0],tmBidx[1],tmBidx[2] )
      for tmBidx in mBidxs ])
      
   #modalFiltering=[ 
   #      thismodalB.reshape([-1,1]).dot(thismodalB.reshape([1,-1]))
   #         for thismodalB in modalBasis.modalFunctions ]
   #modalFilteringSummed=(
   #      numpy.identity(nMask)-numpy.array(modalFiltering).sum(axis=0) )
   modalFilterM=mBs.T.dot(mBs)

   gradMplus=numpy.linalg.pinv(gradM,1e-6)
   #
   gradmM=gradM.dot( modalFilterM )
   gradmMplus=numpy.linalg.pinv(gradmM,1e-6)
   #
   gradzM=gradM-gradmM
   gradzMplus=gradMplus
   gradzMplus=(numpy.identity(nMask)-modalFilterM).dot(gradzMplus)

   print("Input data...",end="") ; sys.stdout.flush()
   thisData=kolmogorov.TwoScreens( baseSize*2, (nMask**0.5)/2.0,
           flattening=2*numpy.pi/baseSize/2.0*30.)[0][:baseSize+2,:baseSize]
   thisData=thisData[:-2] -thisData[2:] # represent shifted screen
   thisData-=thisData.mean()
   thisData/=(thisData.max()-thisData.min())
   # (O)    # \/ add a fake offset to the bottom left quadrant
   # (O) thisData[:baseSize/2,:baseSize/2]+=1
   inputDataV=thisData.ravel()[gradOp.illuminatedCornersIdx] 
   # (m),inputDataV=numpy.array(
   # (m),   [ numpy.random.normal()*tb for tb in modalBasis.modalFunctions ]).sum(axis=0)
   gradsV=numpy.dot( gradM, inputDataV ) # calculate input vector
   print("(done)") ; sys.stdout.flush()


   # prepare plots
   inputDataA=numpy.ma.masked_array(
      numpy.empty(gradOp.n_), gradOp.illuminatedCorners==0 )
   inputDataA.ravel()[gradOp.illuminatedCornersIdx]=inputDataV
   #
   reconFilteredV=gradzMplus.dot(gradsV)
   reconModalV=gradmMplus.dot(gradsV)
   reconJointV=(gradzMplus+gradmMplus).dot(gradsV)
   reconOrigV=(gradMplus).dot(gradsV)
   reconFilteredA=inputDataA.copy()*0
   reconModalA=inputDataA.copy()*0
   reconJointA=inputDataA.copy()*0
   reconOrigA=inputDataA.copy()*0
   reconFilteredA.ravel()[gradOp.illuminatedCornersIdx]=reconFilteredV
   reconModalA.ravel()[gradOp.illuminatedCornersIdx]=reconModalV
   reconJointA.ravel()[gradOp.illuminatedCornersIdx]=reconJointV
   reconOrigA.ravel()[gradOp.illuminatedCornersIdx]=reconOrigV

   for subplot,toshow,thistitle in (
         [ (4,3,1), inputDataA, "i/p" ],
         [ (4,3,2), reconFilteredA, "recon, w/o Z" ],
         [ (4,3,3), inputDataA-reconFilteredA, "(delta)" ],
         [ (4,3,2+3), reconModalA, "recon, only Z" ],
         [ (4,3,3+3), inputDataA-reconModalA, "(delta)" ],
         [ (4,3,2+6), reconJointA, "recon, inc Z" ],
         [ (4,3,3+6), inputDataA-reconJointA, "(delta)" ],
         [ (4,3,2+9), reconOrigA, "recon, no Z" ],
         [ (4,3,3+9), inputDataA-reconOrigA, "(delta)" ],
         ):
      pyp.subplot( subplot[0],subplot[1],subplot[2] )
      pyp.imshow( toshow, vmax=inputDataA.max(),vmin=inputDataA.min() )
      pyp.gca().xaxis.set_visible(0)
      pyp.gca().yaxis.set_visible(0)
      pyp.title(thistitle)
      if subplot[-1]==1: pyp.colorbar()

   print("(input-recon_{{standard/original}}).var()/input.var()="+
         "{0:5.3f}".format( (inputDataV-reconOrigV).var()/inputDataV.var() ))
   print("(input-recon_{{filtered}}).var()/input.var()={0:5.3f}".format(
        (inputDataV-reconFilteredV).var()/inputDataV.var() ))
   print("(input-recon_{{modal}}).var()/input.var()={0:5.3f}".format(
        (inputDataV-reconModalV).var()/inputDataV.var() ))
   print("(input-recon_{{joint}}).var()/input.var()={0:5.3f}".format(
        (inputDataV-reconJointV).var()/inputDataV.var() ))
