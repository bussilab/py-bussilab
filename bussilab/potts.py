"""
Module containing a tool to solve Potts models by enumeration

See `bussilab.potts.Model()`.

"""

import numpy as np
from scipy.optimize import minimize
from typing import Optional, Callable
import warnings
from .coretools import import_numba_jit
from . import coretools

numba_jit=import_numba_jit()

@numba_jit
def _make_lists(size: int,
                colors: int = 1,
                start: int = 0,
                stop: int = -1,
                shifted: bool = False):
    N=(colors+1)**size
    if(stop==-1):
        stop=N
    ret=np.zeros((stop-start,size*colors))
    for i in range(stop-start):
        l=i+start
        for j in range(size):
            m=l%(colors+1)
            if m>0:
                ret[i,(size-j-1)*colors+m-1]=1.0
            l=l//(colors+1)
    if shifted:
        ret-=0.5
    return ret

class InferResult(coretools.Result):
    """Result of a `bussilab.potts.Model.infer` calculation."""
    def __init__(self,
                 *,
                 h: np.ndarray,
                 J: np.ndarray,
                 averages: np.ndarray,
                 loglike: float,
                 success: bool,
                 message: str,
                 nfev: int,
                 nit: int):
        super().__init__()
        self.h = h
        """`np.ndarray`, optimized h."""
        self.J = J
        """`np.ndarray`, optimized h."""
        self.averages = averages
        """`np.ndarray`, resulting averages."""
        self.loglike = loglike
        """`float` containing the resulting likelihood Gamma."""
        self.success = success
        """`bool` reporting the success of the minimizer."""
        self.message = message
        """`str` reporting the possible reason of failuer of the minimizer."""
        self.nfev = nfev
        """`int` reporting the number of function evaluations."""
        self.nit = nit
        """`int` reporting the number of iterations in the minimization procedure."""

class Model:
    def __init__(self,
                 size: int,
                 colors: int = 1,
                 shifted: bool = False,
                 fullmatrix: bool = True):
        """Init model.
           size: number of spins
           colors: number of colors
           fullmatrix: set to False to use less memory (slower)
        """
        if colors != 1:
            warnings.warn("number of colors different from 1 is not tested")
        if shifted:
            warnings.warn("shifted spins not tested")
        self.size=size
        self.colors=colors
        self.nstates=(1+colors)**self.size
        # list of all possible sequences
        self.allseq=_make_lists(self.size,colors,shifted=shifted)
        # outer products of all possible sequences
        # used to compute the energy
        # takes a lot of memory but allow to write operations with numpy
        if fullmatrix:
            self.allseq_matrix=np.einsum('ki,kj->kij',self.allseq,self.allseq)
        else:
            self.allseq_matrix=None
    def compute(self,
                h: np.ndarray,
                J: np.ndarray):
        """Compute averages <sigma_i,sigma_j> for a coupling matrix J.
           Returns (a,b) with a=free energy and b=averages
        """
        if not self.allseq_matrix is None:
            all_ene=np.tensordot(self.allseq_matrix,self.fixJ(J),((1,2),(0,1)))
        else:
            all_ene=np.einsum("ki,kj,ij->k",self.allseq,self.allseq,self.fixJ(J))
        if h is not None:
            all_ene+=np.matmul(self.allseq,h.T)
        shift=all_ene.min()
        all_ene-=shift
        prob=np.exp(-all_ene)
        Z=np.sum(prob)
        if self.allseq_matrix is not None:
            average=np.tensordot(prob,self.allseq_matrix,(0,0))/Z
        else:
            average=np.einsum("ki,kj,k->ij",self.allseq,self.allseq,prob)/Z
        return (-np.log(Z)+shift,average)
    def loglike(self,
                h: np.ndarray,
                J: np.ndarray,
                ave: np.ndarray):
        """Compute -log likelihood for a coupling matrix J with averages ave.
           Returns (a,b) with a=-log likelihood and b=derivatives
        """
        c=self.compute(h,self.fixJ(J))
        l=np.sum(self.fixJ(J)*ave)
        if h is not None:
            l+=np.sum(h*np.diag(ave))
        der=-c[1]+ave
        return (l-c[0],der)
    def draw(self,
             h: np.ndarray,
             J: np.ndarray,
             n: int):
        """Compute averages for a coupling matrix J sampling n states.
        """
        if self.allseq_matrix is not None:
            all_ene=np.tensordot(self.allseq_matrix,self.fixJ(J),((1,2),(0,1)))
        else:
            all_ene=np.einsum("ki,kj,ij->k",self.allseq,self.allseq,self.fixJ(J))
        if h is not None:
            all_ene+=np.matmul(self.allseq,h.T)
        shift=all_ene.min()
        all_ene-=shift
        prob=np.exp(-all_ene)
        Z=np.sum(prob)
        prob/=Z
        ret=np.zeros((self.size*self.colors,self.size*self.colors))
        for i in range(n):
            j=np.random.choice(len(self.allseq),p=prob)
            if self.allseq_matrix is not None:
                ret+=self.allseq_matrix[j]
            else:
                ret+=np.outer(self.allseq[j],self.allseq[j])
        return ret/n
    def fixJ(self,
             J: np.ndarray):
        newJ=0.5*(J+J.T)
        for i in range(self.size):
            newJ[self.colors*i:self.colors*(i+1),self.colors*i:self.colors*(i+1)]=0.0
        return newJ
    def infer(self,
              averages: np.ndarray,
              nseq: int = 1,
              reg: Optional[Callable] = None):
        x0=np.zeros(self.size*self.size*self.colors*self.colors)
        def function(par,m,a):
            J=m.fixJ(par.reshape((self.size*self.colors,self.size*self.colors)))
            h=np.diag(par.reshape((self.size*self.colors,self.size*self.colors)))
            c=m.loglike(h,J,a)
            c=(nseq*c[0],nseq*c[1])
            if reg is not None:
                r=reg(J)
                c=(c[0]+r[0],c[1]+r[1])
            return (c[0],c[1].flatten())
        res = minimize(function, x0, args=(self,averages),
              method="L-BFGS-B",jac=True,tol=1e-10)
        h=np.diag(res.x.reshape((self.size*self.colors,self.size*self.colors)))
        J=self.fixJ(res.x.reshape((self.size*self.colors,self.size*self.colors)))
        return InferResult(
            h=h,
            J=J,
            averages=averages,
            loglike=res.fun,
            success=res.success,
            message=res.message,
            nfev=res.nfev,
            nit=res.nit
            )
    def random_couplings(self,
                         seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        J=np.triu(np.random.normal(0,1.0,(self.size*self.colors,self.size*self.colors)))
        return self.fixJ(J)
