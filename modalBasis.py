# What is this?
# Generate modal bases

import gradientOperator
import numpy

class modalBasis(gradientOperator.geometryType1):
   '''Based on type 1 geometry, calculate modal basis upto the specified
   order of radial terms such that they are orthogonal.
   NB: this code is only meant for the linear and quadratic terms,
   perhaps it ought be replaced with Zernike-like terms defined over the
   aperture to make it more generic.
   '''
   op=None
   numberSubaps=None
   numberPhases=None
   sparse=False
   radialPowers=None
   angularPowers=None

   def __init__( self, pupilMask, radialPowers=[0], angularPowers=[0],
         sparse=False, compute=True, verbose=False, orthonormalize=True):
      self.radialPowers=radialPowers
      self.angularPowers=angularPowers
      gradientOperator.geometryType1.__init__(self,None,pupilMask)
      if compute:
         self.calculateFactors()
         self.calculatemodalFunctions(orthonormalize,verbose)

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
         for ta in xrange(tr%2,min(len(self.angularPowers),tr+1),2):
            for tt in xrange(1+(ta>0)*1):
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
      
class FourierModalBasisType1(gradientOperator.geometryType1):
   '''Based on type 1 geometry, calculate modal basis with complete
   frequency coverage in both orthogonal directions.
   Ignores piston and the highest frequency term.
   '''
   op=None
   numberSubaps=None
   numberPhases=None
   sparse=False

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


# ------------------------------
