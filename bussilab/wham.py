"""
Module containing a WHAM implementation.

See `bussilab.wham.wham()`.

"""
import sys
from typing import Optional, Union, cast
import numpy as np
from . import coretools

class WhamResult(coretools.Result):
    """Result of a `bussilab.wham.wham` calculation."""
    def __init__(self,
                 *,
                 logW: np.ndarray,
                 logZ: np.ndarray,
                 nit: int,
                 nfev: int,
                 eps: float):
        super().__init__()
        self.logW = logW
        """`numpy.ndarray` containing the logarithm of the weight of the frames."""
        self.logZ = logZ
        """`numpy.ndarray` containing the logarithm of the partition function of each state."""
        self.nit = nit
        """The number of performed iterations."""
        self.nfev = nfev
        """The number of function evalutations (might differ from nit when using method='minimize')."""
        self.eps = eps
        """The final error in the iterative solution."""

def wham(bias,
        *,
        frame_weight=None,
        traj_weight=None,
        T: float = 1.0,
        maxiter: int = 1000,
        threshold: float = 1e-20,
        verbose: bool = False,
        logZ: Optional[np.ndarray] = None,
        logW: Optional[np.ndarray] = None,
        normalize: Union[bool, str] = "log",
        method: str = "minimize",
        minimize_opt: Optional[dict] = None):
    """Compute weights according to binless WHAM.

       The main input for this calculation is in the 2D array `bias`.
       Element `bias[i, j]` should contain the energy of the i-th frame computed
       according to the j-th Hamiltonian. Trajectories should be concatenated first,
       so that the total number of frames should be equal to the number of frames
       of each trajectory multiplied by the number of trajectories.
       However, it is also possible to contatenate simulations of different lengths.
       It is crucial however to compute the potential according to each of the employed
       Hamiltonian on **all the frames**, not only on those simulated using that Hamiltonian.

       Notice that by default weights are normalized. It is possible to override this behavior with
       normalize=False. However, starting with v0.0.40, this should not be necessary. The new
       implementation of normalization should be numerically stable in all cases.

       Bugs
       ----

       Up to version 0.042, method="minimize" does not work correctly when setting traj_weights.
       As a consequence, results produced with v0.0.41, where this method is the default,
       might be incorrect. In v0.0.42 this is temporarily fixed by reverting to method="substitute"
       when using traj_weights. In v0.0.43 this should be finally fixed: both methods should
       equally work in all cases, and the default "minimize" method should require less iterations.

       Combining trajectories of different length
       ------------------------------------------

       Let's imagine three frames obtained from three Hamiltonians. Let's assume that the
       energy of frame i in Hamiltonian j is given by `bias[i, j]` defined as
       ```python
       import numpy as np
       bias = np.array([[1, 10, 7],
                       [2, 9, 6],
                       [3, 8, 5]])
       ```
       We can compute the weights with the following command:
       ```python
       np.exp(wham.wham(bias).logW)

       array([0.41728571, 0.39684866, 0.18586563])
       ```
       We now notice that the second and third columns of this matrix are equal except for
       a rigid shift.  They thus correspond to Hamiltonians that are equivalent.
       We should have been able to obtain the same result saying that these frames were
       coming from two simulations. If we only pass the first two columns however
       we obtain different weights
       ```python
       np.exp(wham.wham(bias[:, 0:2]).logW)

       array([0.28224026, 0.43551948, 0.28224026])
       ```
       In order to correcly analyze these frames we should pass the information that
       the second Hamiltonian was used for twice the time:
       ```python
       np.exp(wham.wham(bias[:, 0:2], traj_weight=(1, 2)).logW)

       array([0.41728571, 0.39684866, 0.18586563])
       ```
       Notice that now the weights are identical to those computed in the first example.

       Trusting more a trajectory than another
       ---------------------------------------

       When you concatenate trajectories, you might explicitly want to trust more a trajectory
       than another.  For instance, two trajectories might have been accumulated with a different
       stride, and the reliability of each frame of the one with smaller stride should be lower.
       We thus would like to assign a weight to each frame that accounts for its reliability.

       Consider the following command:
       ```python
       import numpy as np
       np.exp(wham.wham([[3, 5],
                         [4, 4],
                         [4, 4],
                         [4, 4],
                         [4, 4],
                         [4, 4],
                         [4, 4],
                         [4, 4],
                         [4, 4],
                         [4, 4],
                         [4, 4]]).logW)
       ```
       that results in the following weights
       ```python
       array([0.06421006, 0.09357899, 0.09357899, 0.09357899, 0.09357899,
              0.09357899, 0.09357899, 0.09357899, 0.09357899, 0.09357899,
              0.09357899])
       ```
       Notice that all the frames except for the first one are identical. An equivalent result
       would have been obtained using
       ```python
       import numpy as np
       np.exp(wham.wham([[3, 5],
                         [4, 4]], frame_weight=(1, 10)).logW)

       array([0.06421006, 0.93578994])
       ```
       Clearly, the weight of the second frame in this example is equal to ten times the
       weights of the ten corresponding frames in the previous example.


       Parameters
       ----------

       bias: np.ndarray
           An array with shape (nframes, ntraj) containing the bias potential applied to
           each frame according to each of the Hamiltonians.

       frame_weight: np.ndarray, optional
           An array with nframes elements. These elements should contain the reliability weight
           of the frames.  By default, these weights are set to one.

       traj_weight: np.ndarray, optional
           An array with ntraj elements. These elements should contain the total weight of each of
           the Hamiltonians. Should be used when combining trajectories of different lengths.

       T: float, optional
           The temperature of the system. This number is just used to divide the `bias` array in
           order to make it adimensional. In case replicas are simulated at different temperatures,
           it is possible to pass an array here, with ntraj elements.

       maxiter: int, optional
           Maximum number of iterations in the minimization procedure.

       threshold: float, optional
           Threshold for the minimization procedure.

       verbose: bool, optional
           If True, print information as the minimization proceeds.

       logZ: np.ndarray, optional
           Array with ntraj elements.
           Initial value for the logarithm of the partition functions. If not provided, it is
           computed from the bias.  Providing an initial guess that is close to the converged
           value (e.g. as obtained from a calculation with a limited number of frames) can speed up
           significantly the convergence.

       logW: np.ndarray, optional
           Array with nframes elements.
           Initial value for the logarithm of the weights. If not provided, they are computed from
           the bias.  Providing an initial guess that is close to the converged value can speed up
           significantly the convergence. *If logW is provided, logZ is ignored*.

       normalize: bool or str, optional
           By default, "log", which properly normalizes weights in all cases.
           normalize=True or False is enabled for backward compatibility.

       method: str, optional
           If "substitute", solve self-consistent equations by substitution.
           If "minimize", use a minimization as in J Chem Phys 136, 144102 (2012).
           Prior to version 0.0.40, the default was "substitute".
           Starting with version 0.0.41, the default is "minimize".

       minimize_opt: dict, optional
           If method=="minimize", this dict can be used to pass options to scipy.minimize.
           Notice that by default the minimization is performed using 'L-BFGS-B'.
    """

    # allow tuples or lists
    bias = coretools.ensure_np_array(bias)
    frame_weight = coretools.ensure_np_array(frame_weight)
    traj_weight = coretools.ensure_np_array(traj_weight)

    nframes = bias.shape[0]
    ntraj = bias.shape[1]

    if isinstance(T,float):
        T=T*np.ones(ntraj)
    T = coretools.ensure_np_array(T)

    # default values
    if frame_weight is None:
        frame_weight = np.ones(nframes)
    if traj_weight is None:
        traj_weight = np.ones(ntraj)

    assert len(traj_weight) == ntraj
    assert len(frame_weight) == nframes

    # divide by T once for all
    shifted_bias = bias/T[np.newaxis,:]
    # track shifts
    shifts1 = np.min(shifted_bias, axis=1)
    shifted_bias -= shifts1[:,np.newaxis]
    shifts0 = np.min(shifted_bias, axis=0)
    shifted_bias -= shifts0[np.newaxis,:]

    # do exponentials only once
    expv = np.exp(-shifted_bias)

    if logW is not None:
        Z = np.matmul(np.exp(logW-shifts1), expv)
        Z /= np.sum(Z*traj_weight)
    elif logZ is not None:
        Z = np.exp(logZ+shifts0)
    else:
        Z = np.ones(ntraj)

    Zold = Z

    if verbose:
        sys.stderr.write("WHAM: start\n")
    if method == "substitute":
        for nit in range(maxiter):
            # find unnormalized weights
            weight = 1.0/np.matmul(expv, traj_weight/Z)*frame_weight
            # update partition functions
            Z = np.matmul(weight, expv)
            # normalize the partition functions
            Z /= np.sum(Z*traj_weight)
            # monitor change in partition functions
            eps = np.sum(np.log(Z/Zold)**2)
            Zold = Z
            if verbose:
                sys.stderr.write("WHAM: iteration "+str(nit)+" eps "+str(eps)+"\n")
            if eps < threshold:
                break
        nfev=nit
    elif method == "minimize":
        from scipy.optimize import minimize
        # in the equation below, traj_weight should be exactly scaled to the number of frames
        traj_weight_scaled=traj_weight/np.sum(traj_weight)*np.sum(frame_weight)
        def func(x):
            Zm1=np.exp(-x)
            tmp=expv*(traj_weight_scaled*Zm1)[np.newaxis,:]
            tmp1=np.sum(tmp,axis=1)
            C=np.sum(frame_weight*np.log(tmp1))+np.sum(traj_weight_scaled*x)
            tmp/=tmp1[:,np.newaxis]
            grad=-np.matmul(frame_weight,tmp)+traj_weight_scaled
            return C,grad
        if minimize_opt is not None:
            if "method" not in minimize_opt:
                minimize_opt["method"]="L-BFGS-B"
            if "jac" in minimize_opt:
                if not minimize_opt["jac"]:
                    raise ValueError("minimize_opt['jac'] must be True")
            else:
                minimize_opt["jac"]=True
        else:
            minimize_opt={}
            minimize_opt["method"]="L-BFGS-B"
            minimize_opt["jac"]=True
        x=np.log(Z)
        res = minimize(func, x, **minimize_opt)
        Z=np.exp(res.x)
        Z/=np.sum(Z*traj_weight)
        weight = 1.0/np.matmul(expv, traj_weight/Z)*frame_weight
        Zold=Z.copy()
        Z = np.matmul(weight, expv)
        Z /= np.sum(Z*traj_weight)
        nit=res.nit
        nfev=res.nfev
        eps=np.sum(np.log(Z/Zold)**2)
    else:
        raise ValueError("method should be 'minimize' or 'substitute'")

    # this is the traditional implementation, either normalized or not
    if isinstance(normalize,bool):
        if normalize:
            weight *= np.exp((shifts1-np.max(shifts1)))
            # normalized weights
            weight /= np.sum(weight)
            with np.errstate(divide = 'ignore'):
                logW = np.log(weight)
        else:
            logW = np.log(weight) + shifts1
    # this is the new default, which allows normalizing also non-normalizable weights
    else:
        if normalize == "log":
            logW = np.log(weight) + shifts1
            logW -= np.max(logW)
            logW -= np.log(np.sum(np.exp(logW)))
        else:
            raise ValueError("normalize should be True, False, or 'log'")

    if verbose:
        sys.stderr.write("WHAM: end")

    logW = cast(np.ndarray, logW)  # to avoid mypy error

    return WhamResult(logW=logW, logZ=np.log(Z)-shifts0, nit=nit, nfev=nfev, eps=eps)
