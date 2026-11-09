"""Analytic point-to-nearest-surface residual Jacobian, using RGB-D only."""
import numpy as np

class SurfaceObjective:
    def __init__(self,points,tree,marker=None,marker_local=None):
        self.points=np.asarray(points);self.tree=tree
        self.marker=marker;self.marker_local=marker_local;self.last=None
    def evaluate(self,x):
        if self.last is not None and np.array_equal(x,self.last):return
        self.last=np.array(x).copy();c,s=np.cos(x[3]),np.sin(x[3])
        r=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        dr=np.array([[-s,-c,0.],[c,-s,0.],[0.,0.,0.]])
        delta=self.points-x[:3];local=delta@r;_,indices=self.tree.query(local)
        residual=(local-self.tree.data[indices]).ravel()
        jac=np.zeros((len(local),3,4));jac[:,:,:3]=-r.T;jac[:,:,3]=delta@dr
        jac=jac.reshape(-1,4)
        if self.marker is not None:
            residual=np.r_[residual,4*(r@self.marker_local+x[:3]-self.marker)]
            jac=np.vstack([jac,4*np.column_stack([np.eye(3),dr@self.marker_local])])
        self.residual=residual;self.gradient=jac
    def fun(self,x):self.evaluate(x);return self.residual
    def jac(self,x):self.evaluate(x);return self.gradient
