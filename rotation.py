"""ABBOT : Generate slope rotation operator, based off gradientOperatorType1.
"""

from numpy import array, float64, floor, where, zeros
from numpy import cos, sin, pi
from numpy.linalg import pinv
# can choose to use Type 1 without any consequences
from gradientOperator import geometryType1, gradientOperatorType1

class rotationOperator(geometryType1):
    '''Using Type 1 geometry, generate a wavefront rotation operator matrix.'''
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
        self.numberSubaps=None
        self._geometryInit(self)

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

    def returnOp(self, angle=None):
        if self.numberSubaps==None:
            return None
        if not angle is None:
            self.angle=angle
        if self.last_angle!=self.angle:
            indices=self._calcIndices()
            self._calcOp(indices)
            self.last_angle=self.angle
        return self.op

    def _calcIndices(self):
        indices=[]
        rM=array( (( cos(self.angle/180.0*pi), -sin(self.angle/180.0*pi) ),
                   ( sin(self.angle/180.0*pi),  cos(self.angle/180.0*pi) )) )
        # rotate coordinates about centre and then iterate positions
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

    def _calcOp__NumpyArray(self, indices):
        self.op=zeros(
                [self.numberPhases,self.numberPhases],float64)
        for i,idx,f in indices:
            self.op[i,idx]+=f

    def _calcOp__scipyCSR(self, indices):
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


class slopeRotationOperator(rotationOperator, gradientOperatorType1):
    '''Using Type 1 geometry, generate a wavefront rotation operator matrix.'''
    op=None
    sparse=False

    def __init__(self, angle=None, subapMask=None, pupilMask=None, sparse=False):
        gradientOperatorType1.__init__(self, subapMask, pupilMask)
        rotationOperator.__init__(self, angle, subapMask, pupilMask, sparse)
        if self.sparse:
            raise NotImplementedError("Cannot produce a sparse version")
            self.gOp_calcOp=gradientOperatorType1._calcOp__scipyCSR
        else:
            self.gOp_calcOp=gradientOperatorType1._calcOp__NumpyArray

    def returnOp(self, angle=None):
        if self.numberSubaps==None: return None
        rotM=rotationOperator.returnOp(self, angle)
        self.gOp_calcOp(self) # have to do it this way
        gM=self.op
        gM_i=pinv(gM)
        self.op=gM.dot(rotM).dot(gM_i)
        return( self.op )
