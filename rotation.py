"""ABBOT : Generate slope rotation operator, based off gradientOperatorType1.
"""

from numpy import array, float64, floor, where, zeros, flatnonzero
from numpy import cos, sin, pi
from numpy.linalg import pinv
# can choose to use Type 1 without any consequences
from gradientOperator import geometryType1, gradientOperatorType1
        

rotationMatrix=lambda ang : array( (( cos(ang), -sin(ang) ),
                                    ( sin(ang),  cos(ang) )) )
# note that the rotation matrix is defined with an opposite handedness
# in scipy.ndimage.rotate
deg2rad=lambda ang : ang/180.0*pi


class __rotationOperator__(geometryType1):
    '''Base class, not to be used'''
    op=None
    sparse=False

    def __init__(self, angle=0, subapMask=None, pupilMask=None, sparse=False):
        self.sparse=sparse
        self.angle=angle
        self.last_angle=None
        geometryType1.__init__(self, subapMask, pupilMask)
        ##
        if self.sparse:
            self._calcOp=self._calcOp__scipyCSR
        else:
            self._calcOp=self._calcOp__NumpyArray
        ##
        self._geometryInit()

    def _geometryInit(self):
        raise RuntimeError("Not implemented")

    def returnOp(self, angle=None):
        if self.numberSubaps==None:
            return None
        if not angle is None:
            self.angle=angle
        if self.last_angle!=self.angle:
            self._calcOp()
            self.last_angle=self.angle
        return self.op

    def _calcOp__NumpyArray(self):
        indices=self._calcIndices()
        self.op=zeros(
                [self.numberPhases,self.numberPhases],float64)
        for i,idx,f in indices:
            self.op[i,idx]+=f

    def _calcOp__scipyCSR(self):
        indices=self._calcIndices()
        import scipy.sparse # only required if we reach this stage
        entries={}
        for i,idx,f in indices:
            entries[(i,idx)]=(entries[(i,idx)] if (i,idx) in entries else 0)+f
        ##
        row,col=[],[] ; data=[]
        for k in entries:
            row.append(k[0])
            col.append(k[1])
            data.append(entries[k])
        self.op=scipy.sparse.csr_matrix((data,(row,col)), dtype=float64)


class rotationExplicitOperator(__rotationOperator__):
    '''From the calculation of rotation of points explicitly,
    generate a wavefront rotation operator matrix.'''

    def __init__(self, angle=0, subapMask=None, pupilMask=None, sparse=False):
        __rotationOperator__.__init__(self,angle,subapMask,pupilMask,sparse)

    def _geometryInit(self):
        if self.numberSubaps==None:
            return
        self.origVs=array( # coordinates of original geometry
                [ self.illuminatedCornersIdx%self.n_[1],
                  self.illuminatedCornersIdx//self.n_[1] ])
        self.im={}
        for i,pair in enumerate(self.origVs.T):
            self.im[tuple(pair)]=i
        self.centreV=array(self.n_).reshape([2,1])/2.0-0.5

    def _calcIndices(self):
        indices=[]
        # rotate coordinates about centre and then iterate positions
        # note negative angle for consistency
        rM=rotationMatrix(deg2rad(-self.angle))
        for i,pair in enumerate(
                (rM.dot(self.origVs-self.centreV)+self.centreV).T ):
            pairs=[]
            ##
            frac=(1-pair[0]+floor(pair[0])) * (1-pair[1]+floor(pair[1]))
            pairs.append( [tuple([floor(pair[0]),floor(pair[1])]),frac] )
            pairs[-1].append(
                self.im[pairs[-1][0]] if pairs[-1][0] in self.im else None)
            frac=(1-pair[0]+floor(pair[0])) * (pair[1]-floor(pair[1]))
            pairs.append( [tuple([floor(pair[0]),floor(pair[1])+1]),frac] )
            pairs[-1].append(
                self.im[pairs[-1][0]] if pairs[-1][0] in self.im else None)
            frac=(pair[0]-floor(pair[0])) * (1-pair[1]+floor(pair[1]))
            pairs.append( [tuple([floor(pair[0])+1,floor(pair[1])]),frac] )
            pairs[-1].append(
                self.im[pairs[-1][0]] if pairs[-1][0] in self.im else None)
            frac=(pair[0]-floor(pair[0])) * (pair[1]-floor(pair[1]))
            pairs.append( [tuple([floor(pair[0])+1,floor(pair[1])+1]),frac] )
            pairs[-1].append(
                self.im[pairs[-1][0]] if pairs[-1][0] in self.im else None)
            ##
            for p,f,idx in pairs:
                if f==0:
                    continue
                if (p[0]>=self.n_[1] or p[1]>=self.n_[0]
                        or p[0]<0 or p[1]<0) or idx is None:
                    pass
                else:
#                    c=tuple(array(p).astype('i').tolist())
                    # c for 2D rotation +=f
                    # i for matrix +=f
                    indices.append([i,idx,f])
        return(indices)

class __slopeRotationOperator__(gradientOperatorType1):
    '''Base class, needs to be inherited & overloaded to be useful.'''
    op=None
    sparse=False

    def __init__(self, angle=None, subapMask=None, pupilMask=None, sparse=False):
        gradientOperatorType1.__init__(self, subapMask, pupilMask)
        if self.sparse:
            # critically, the need to invert suggests the best that can be done is to supply
            # matrices and let the use use e.g. a CG algorithm
            self.gOp_calcOp=gradientOperatorType1.calcOp_scipyCSR
        else:
            self.gOp_calcOp=gradientOperatorType1.calcOp_NumpyArray

    def returnOp(self, angle=None):
        if self.numberSubaps==None: return None
        self.gOp_calcOp(self) # have to do it this way
        gM=self.op
        if not self.sparse:
            gM_i=pinv(gM) # this causes the sparse heartache
            self.op=gM.dot(self.rotM).dot(gM_i)
        else:
            self.op=(self.gM,self.rotM) # return tuple
        return( self.op )

class slopeRotationExplicitOperator(__slopeRotationOperator__,rotationExplicitOperator):
    '''Using Type 1 geometry, generate a slope rotation operator matrix from a
    wavefront rotation operator using the explicit operator form.'''
    def __init__(self, angle=None, subapMask=None, pupilMask=None, sparse=False):
        __slopeRotationOperator__.__init__(self, angle, subapMask, pupilMask, sparse)
        rotationExplicitOperator.__init__(self, angle, subapMask, pupilMask, sparse)
        self.rotM=rotationExplicitOperator.returnOp(self, angle)

class rotationScipyOperator(__rotationOperator__):
    '''From the rotation of points via scipy.ndimage.rotate,
    generate a wavefront rotation operator matrix.
    Note that the scipy rotation direction is opposite hence a negative
    angle must be considered.'''

    def __init__(self, angle=0, subapMask=None, pupilMask=None, sparse=False, hah=False):
        __rotationOperator__.__init__(self,angle,subapMask,pupilMask,sparse)
        self.hah=hah # if True then rotate back by angle/2 then forwards again by angle
        from scipy.ndimage import rotate
        self.rotateFn=rotate

    def _geometryInit(self):
        if self.numberSubaps==None:
            return
        blank=zeros(self.n_)
        self.pokes=[]
        for i in self.illuminatedCornersIdx:
            blank.ravel()[i]=1
            self.pokes.append(blank.copy())
            blank.ravel()[i]=0
        self.pokes=array(self.pokes)

    def _calcIndices(self):
        indices=[]
        if self.hah:
            rot_pokes_half=self.rotateFn(
                self.pokes,
                -self.angle/2,
                (1,2),
                False,
                order=1,
                mode='constant',
                prefilter=False) # forwards from 0 to +self.angle/2
            rot_pokes_full=self.rotateFn(
                rot_pokes_half,
                self.angle,
                (1,2),
                False,
                order=1,
                mode='constant',
                prefilter=False) # backwards from +self.angle/2 to -self.angle/2
        else:
            rot_pokes_full=self.rotateFn(
                self.pokes,
                self.angle,
                (1,2),
                False,
                order=1,
                mode='constant',
                prefilter=False) # backwards from 0 to -self.angle
        ##
        for i,ci in enumerate(self.illuminatedCornersIdx):
            pairs=[]
            this_rot_poke=rot_pokes_full[i].ravel().take(self.illuminatedCornersIdx)
            for j in flatnonzero(this_rot_poke):
                indices.append([i,j,this_rot_poke[j]])
        return(indices)

class slopeRotationScipyOperator(__slopeRotationOperator__,rotationScipyOperator):
    '''Using Type 1 geometry, generate a slope rotation operator matrix from a
    wavefront rotation operator using the scipy.ndimage.rotation function.'''
    def __init__(self, angle=None, subapMask=None, pupilMask=None, sparse=False):
        __slopeRotationOperator__.__init__(self, angle, subapMask, pupilMask, sparse)
        rotationExplicitOperator.__init__(self, angle, subapMask, pupilMask, sparse)
        rotationScipyOperator.__init__(self, angle, subapMask, pupilMask, sparse)
        self.rotM=rotationScipyOperator.returnOp(self, angle)