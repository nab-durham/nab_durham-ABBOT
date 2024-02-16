# -*- coding: utf-8 -*-
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
        rotationScipyOperator.__init__(self, angle, subapMask, pupilMask, sparse)
        self.rotM=rotationScipyOperator.returnOp(self, angle)

class affineScipyOperator(rotationScipyOperator):
    '''From the affine transform of points via scipy.ndimage.affine_transform,
    generate a wavefront affine transform operator matrix.
    The function notes state:
       Given an output image pixel index vector o, the pixel value is determined from the input image at position np.dot(matrix, o) + offset.

This does ‘pull’ (or ‘backward’) resampling, transforming the output space to the input to locate data. Affine transformations are often described in the ‘push’ (or ‘forward’) direction, transforming input to output. If you have a matrix for the ‘push’ transformation, use its inverse (numpy.linalg.inv) in this function.
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html)
    '''

    def __init__(self, translate=[0,0], scale=[1,1], shear=[0,0], angle=0, subapMask=None, pupilMask=None, sparse=False, order=1, prefilter=False):
        '''The parameters for the affine transform are,
        translate=[0,0], [x,y], [sample resolution],
        scale=[1,1], [x,y] [sample resolution],
        shear=[0,0], [x,y] [sample resolution],
        angle=0, [degrees counter-clockwise],
        and,
        order=1, spline interpolation (1=linear),
        prefilter=False, pre-filtering prior to spine interpolatioon (not reqd. for order=1)
        '''
        __rotationOperator__.__init__(self,angle,subapMask,pupilMask,sparse)
        from scipy.ndimage import affine_transform
        # build the matrix which operates as follows,
        # 1,rotation,
        # 2,scale,
        # 3,translate
        # unsure about shear but think it is between 2 and 3.
        self.affine_mx=zeros([2,2],float64)
        self.affine_mx[0,0]=(scale[1])*cos( deg2rad(angle))
        self.affine_mx[0,1]=(scale[0])*sin( deg2rad(angle))+(-shear[0])
        self.affine_mx[1,0]=(scale[1])*sin(-deg2rad(angle))+(-shear[1])
        self.affine_mx[1,1]=(scale[0])*cos( deg2rad(angle))
        self.affine_offset=array(self.n_-self.affine_mx.dot(self.n_))/2.0 # rotate about centre
        self.affine_offset-=array(translate)[::-1] # X,Y -> Y,X
        self.affine=lambda ip : affine_transform(
                ip, self.affine_mx, self.affine_offset, order=order, mode='constant',
                cval=0, prefilter=prefilter 
            )

    def _calcIndices(self):
        indices=[]
        tfmd_pokes_full=[ self.affine(self.pokes[i]) for i in range(self.numberPhases) ]
        ##
        for i,ci in enumerate(self.illuminatedCornersIdx):
            pairs=[]
            this_tfm_poke=tfmd_pokes_full[i].ravel().take(self.illuminatedCornersIdx)
            for j in flatnonzero(this_tfm_poke):
                indices.append([i,j,this_tfm_poke[j]])
        return(indices)

class slopeAffineScipyOperator(__slopeRotationOperator__,affineScipyOperator):
    '''Using Type 1 geometry, generate an affine transform matrix from a
    wavefront rotation operator using the scipy.ndimage.affine_transform function.'''
    def __init__(self, translate=[0,0], scale=[1,1], shear=[0,0], angle=0, subapMask=None, pupilMask=None, sparse=False, order=1, prefilter=False):
        __slopeRotationOperator__.__init__(self, angle, subapMask, pupilMask, sparse)
        affineScipyOperator.__init__(self, translate, scale, shear, angle, subapMask, pupilMask, sparse, order, prefilter)
        self.rotM=affineScipyOperator.returnOp(self)


if __name__=='__main__':
    #
    # for future unittest compatibility, use assert statements in this test code
    #
    def test_rotation(ip,success,failure):
# RE-WRITE ME...         global thisProj,layerExM,sumPrM,sumLayerExM
# RE-WRITE ME...         (nAzi,gsHeight,mask)=ip
# RE-WRITE ME...         (thisProj,layerExM,layerExUTM,sumPrM,sumCentPrM,sumLayerExM,layerCentExM,
# RE-WRITE ME...             sumLayerCentExM)={},{},{},{},{},{},{},{}
# RE-WRITE ME...         ...
# RE-WRITE ME...         # TEST: basic matrices comparison between sparse and dense
# RE-WRITE ME...         try:
# RE-WRITE ME...             assert ( numpy.array( layerExM[1].todense() )-layerExM[0] ).var()==0,\
# RE-WRITE ME...                 "layerExM sparse!=dense"
# RE-WRITE ME...         except:
# RE-WRITE ME...             failure+=1
# RE-WRITE ME...             print(sys.exc_info()[1])
# RE-WRITE ME...         else:
# RE-WRITE ME...             success+=1
# RE-WRITE ME...         try:
# RE-WRITE ME...             assert (numpy.array(layerCentExM[1].todense())-layerCentExM[0]
# RE-WRITE ME...                     ).var()==0, "layerExM sparse!=dense"
# RE-WRITE ME...         except:
# RE-WRITE ME...             failure+=1
# RE-WRITE ME...             print(sys.exc_info()[1])
# RE-WRITE ME...         else:
# RE-WRITE ME...             success+=1
# RE-WRITE ME...         # TEST: input means 
# RE-WRITE ME...         tilts=lambda s : numpy.add.outer(
# RE-WRITE ME...                 numpy.arange(-s[0]/2,s[0]/2),
# RE-WRITE ME...                 numpy.arange(-s[1]/2,s[1]/2) )
# RE-WRITE ME...         quadratic=lambda s : numpy.add.outer(
# RE-WRITE ME...                 numpy.arange(-s[0]/2,s[0]/2)**2.0,
# RE-WRITE ME...                 numpy.arange(-s[1]/2,s[1]/2)**2.0 )
# RE-WRITE ME...         ...
# RE-WRITE ME...         try:
# RE-WRITE ME...             test=(ipProjV[0]-ipProjV[1]).var()
# RE-WRITE ME...             assert test<1e-10, "ipProjV, var{sparse-dense}>1e-10:"+str(test)
# RE-WRITE ME...         except:
# RE-WRITE ME...             failure+=1
# RE-WRITE ME...             print(sys.exc_info()[1])
# RE-WRITE ME...         else:
# RE-WRITE ME...             success+=1
        #
        return success,failure

    import datetime, sys, numpy
    success,failure=0,0
    titleStr="rotation.py, automated testing"
    print("\n{0:s}\n{1:s}\n".format(titleStr,len(titleStr)*"^"))
    print("BEGINS:"+str(datetime.datetime.now()))
    #
    rad=10 # note, array pixels
#     nAzi=5
#     gsHeight=3 # note, unitless
    #
    circ = lambda b,r : (numpy.add.outer(
            (numpy.arange(b)-(b-1.0)/2.0)**2.0,
            (numpy.arange(b)-(b-1.0)/2.0)**2.0 )**0.5<=r).astype( numpy.int32 )
    mask=circ(rad,rad/2)-circ(rad,rad/2*0.25)
    #
    success,failure=test_rotation([None],success,failure)
    total=success+failure
    succeeded,failed=success,failure
    print("SUMMARY:rotation: {0:d}->{1:d} successes and {2:d} failures".format(
            total, succeeded, failed))
    print("ENDS:"+str(datetime.datetime.now()))
    sys.exit( failed>0 )