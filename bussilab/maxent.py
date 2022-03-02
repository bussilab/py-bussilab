"""
Tools to perform reweighting using MaxEnt.
"""
import sys
from typing import Optional, Callable
import numpy as np
from scipy.optimize import minimize
from . import coretools

try:
    import cudamat as cm  # pylint: disable=import-error
    _HAS_CUDAMAT=True
except ModuleNotFoundError:
    _HAS_CUDAMAT=False


class MaxentResult(coretools.Result):
    """Result of a `bussilab.maxent.maxent` calculation."""
    def __init__(self,
                 *,
                 logW_ME: np.ndarray,
                 lambdas: np.ndarray,
                 averages: np.ndarray,
                 gamma: float,
                 success: bool,
                 message: str,
                 nfev: int,
                 nit: int):
        super().__init__()
        self.logW_ME = logW_ME
        """`np.ndarray` with `traj.shape[0]` elements, logarithms of the optimized weights."""
        self.lambdas = lambdas
        """`np.ndarray` with `len(reference)` elements, optimized Lagrangian multipliers."""
        self.averages = averages
        """`np.ndarray` with `len(reference)` elements, resulting averages."""
        self.gamma = gamma
        """`float` containing the resulting likelihood Gamma."""
        self.success = success
        """`bool` reporting the success of the minimizer."""
        self.message = message
        """`str` reporting the possible reason of failuer of the minimizer."""
        self.nfev = nfev
        """`int` reporting the number of function evaluations."""
        self.nit = nit
        """`int` reporting the number of iterations in the minimization procedure."""

# Internal tool
# should be called before sum() calls in cudamat
def _ensure_ones(min_size):
    if cm.CUDAMatrix.ones.shape[0]<min_size:
        cm.CUDAMatrix.ones = cm.empty((min_size, 1)).assign(1.0)

def _ensure_cm_init():
    if not hasattr(cm.CUDAMatrix, 'ones'):
        cm.cublas_init()
    elif not isinstance(cm.CUDAMatrix.ones,cm.CUDAMatrix):
        cm.cublas_init()

# Internal tool to compute averages over trajectory.
# Does not access external data.
# Might be optimized on GPU or to access traj from disk.
def _heavy_part(logW: np.ndarray,
                traj: np.ndarray,
                l: np.ndarray,
                weights: bool = False,
                cuda: bool = False):
    if cuda:
        cu_minus_l=cm.CUDAMatrix(np.reshape(-l,(-1,1)))
        logW_ME = cm.dot(traj,cu_minus_l)
        logW_ME.add(logW)
        shift_ME = logW_ME.max(0).asarray()[0,0]
        logW_ME.subtract(float(shift_ME))
        if weights:
            save_logW_ME=logW_ME.copy()
        cm.exp(logW_ME)
        _ensure_ones(logW_ME.shape[0])
        Z=logW_ME.sum(0).asarray()[0,0]
        averages = cm.dot(logW_ME.transpose(),traj).asarray()[0,:]
        averages=np.array(averages,dtype="float64")/Z
        logZ = np.log(Z) + shift_ME
        if not weights:
            return (logZ, averages)
        save_logW_ME.subtract(float(np.log(Z)))
        return (logZ, averages, save_logW_ME.asarray()[:,0])
    else:
        logW_ME = logW-np.dot(traj, l)  # maxent correction
        shift_ME = np.max(logW_ME)  # shift to avoid overflow
        W_ME = np.exp(logW_ME - shift_ME)
        # Partition function:
        Z = np.sum(W_ME)
        # Averages:
        averages = np.dot(W_ME, traj) / Z
        logZ = np.log(Z) + shift_ME
        if not weights:
            return (logZ, averages)
        # only return weights if requested:
        logW_ME -= np.log(Z)+shift_ME
        return (logZ, averages, logW_ME)


def maxent(
        traj,
        reference,
        *,
        logW=None,
        maxiter: int = 1000,
        verbose: bool =False,
        lambdas=None,
        l2=None,
        l1=None,
        method: str = "L-BFGS-B",
        regularization: Optional[Callable] = None,
        tol: Optional[float] = None,
        options=None,
        cuda=False):
    """Tool that computes new weights to enforce reference values.

       This tools process a an array containing the observables computed along a trajectory and
       returns new weights that satisfy the maximum entropy principle and so that weighted averages
       agree with reference values.

       Parameters
       ----------

       traj : array_like
           A 2D array (lists or tuples are internally converted to numpy arrays).
           `traj[i,j]` is j-th observable computed in the i-th frame.
           If traj is a CUDAMatrix object, then cudamat is used irrespectively of the
           bool parameter `cuda`.

       reference : array_like

           A 1D array (lists or tuples are internally converted to numpy arrays)
           containing the reference values to be enforced. If the i-th element is a tuple
           or an array with 2 elements, they are interpreted as boundaries. For instance,
           `reference=[1.0,(2.0,3.0)]` will make sure the first observable has value 1 and
           the second observable is within the range (2,3). Boundaries equal to `+np.inf`
           or `-np.inf` can be used to imply no boundary. Notice that boundaries in the
           form (A,B) where both A and B are finite are implemented by adding fictitious
           variables in a way that is transparent to the user.  Boundaries in the form
           (A,B) where one of A or B is finite and the other is infinite are implemented
           as boundaries on lambdas.  Boundaries in the form (A,A) are interpreted as
           constraints.

       logW : array_like

           A 1D array (lists or tuples are internally converted to numpy arrays)
           containing the logarithm of the a priori weight of the provided frames.

       lamdbas : array_like

           A 1D array with initial values of lambda. A good guess will minimize faster. A
           typical case would be recycling the lambdas obtained with slighlty different
           regularization parameters.

       l2 : None, float, or array_like

           Prefactor for L2 regularization. If None, no regularization is applied. If
           float, the same factor is used on all the lambdas.  If it is an array, it
           should have length equal to `len(reference)`.

       l1 : None, float, or array_like

           Prefactor for L1 regularization. If None, no regularization is applied. If
           float, the same factor is used on all the lambdas.  If it is an array, it
           should have length equal to `len(reference)`.

       regularization : callable or None

           A function that takes as argument the current lambdas and return an tuple
           containing the regularization function and its derivatives. For instance,
           passing a function defined as
           `def reg(x): return (0.0001*0.5*np.sum(x**2),0.0001*x)`
           is equivalent to passing `l2=0.0001`.

       verbose : bool
           If True, progress informations are written on stdout.

       method : str
           Minimization method. See documentation of `scipy.optimize.minimize`.

       maxiter : int
           Maximum number of iterations

       tol : float or None
           Tolerance for minimization. See documentation of scipy.optimize.minimize.

       options : dict
           Arbitrary options passed to `scipy.optimize.minimize`.

       cuda : bool or None (default False)
           Use cuda. If None, chosen based on the availability of the cudamat library.

       Notes on using CUDA
       -------------------

       Note that for normal datasets the cost of transfering the traj object to
       the GPU dominates. It it however possible to transfer the traj object first to the GPU
       with `cu_traj=cm.CUDAMatrix(traj)` and then reuse it for multiple calls
       (e.g. for a hyper parameter scan).

    """

    if cuda is None:
        cuda = _HAS_CUDAMAT

    # when a cudamatrix is passed, cuda is enabled by default
    if _HAS_CUDAMAT:
        if isinstance(traj,cm.CUDAMatrix):
            cuda=True

    if cuda:
        if not _HAS_CUDAMAT:
            raise ValueError("Cudamat not available, can only run ANN with numpy")
        _ensure_cm_init()
        if isinstance(traj,cm.CUDAMatrix):
            cu_traj=traj
        else:
            traj = coretools.ensure_np_array(traj)
            cu_traj=cm.CUDAMatrix(traj)
    else:
        traj = coretools.ensure_np_array(traj)

    lambdas = coretools.ensure_np_array(lambdas)

    nframes = traj.shape[0]
    nobs = traj.shape[1]

    # accepts a scalar as l2 regularization term
    if isinstance(l2, float):
        l2 = np.ones(nobs)*l2

    l2 = coretools.ensure_np_array(l2)

    if isinstance(l1, float):
        l1 = np.ones(nobs)*l1

    l1 = coretools.ensure_np_array(l1)
    # default values
    if logW is None:
        logW = np.zeros(nframes)
    if lambdas is None:
        lambdas = np.zeros(nobs)

    # checks
    assert len(reference) == nobs
    assert len(logW) == nframes
    assert len(lambdas) == nobs

    fullreference = []
    bounds = []
    box_const = []
    for i in range(nobs):
        if hasattr(reference[i], "__len__"):
            if len(reference[i]) > 1:
                if len(reference[i]) > 2:
                    raise TypeError("")
                if reference[i][0] > reference[i][1]:
                    raise TypeError("")
                if reference[i][0] == reference[i][1]:
                    fullreference.append(reference[i][0])
                    bounds.append((-np.inf, +np.inf))
                    box_const.append(False)
                elif (np.isinf(reference[i][1]) and reference[i][1] > 0.0
                      and not np.isinf(reference[i][0])):
                    fullreference.append(reference[i][0])
                    bounds.append((-np.inf, 0.0))
                    box_const.append(False)
                elif (np.isinf(reference[i][0]) and reference[i][0] < 0.0
                      and not np.isinf(reference[i][1])):
                    fullreference.append(reference[i][1])
                    bounds.append((0.0, +np.inf))
                    box_const.append(False)
                elif not np.isinf(reference[i][0]) and not np.isinf(reference[i][1]):
                    fullreference.append(reference[i][0])
                    fullreference.append(reference[i][1])
                    bounds.append((-np.inf, 0.0))
                    bounds.append((0.0, +np.inf))
                    box_const.append(True)
                elif ((np.isinf(reference[i][0]) and reference[i][0] < 0.0)
                      and (np.isinf(reference[i][1]) and reference[i][1] > 0.0)):
                    fullreference.append(0.0)
                    bounds.append((0.0, 0.0))
                    box_const.append(False)
                else:
                    raise TypeError("")
            else:
                fullreference.append(reference[i][0])
                bounds.append((-np.inf, +np.inf))
                box_const.append(False)
        else:
            fullreference.append(reference[i])
            bounds.append((-np.inf, +np.inf))
            box_const.append(False)


    # to fix these, use np.asarray, only available in numpy 1.20.0
    fullreference = np.array(fullreference)  # type: ignore
    bounds = np.array(bounds)  # type: ignore
    box_const = np.array(box_const)  # type: ignore

    nit = 0
    def _callback(par):
        nonlocal nit  # needed to access outer scope
        nit += 1
        if verbose:
            sys.stderr.write("MAXENT: iteration "+str(nit)+"\n")

    callback: Optional[Callable] = None
    # verbose logging
    if verbose:
        sys.stderr.write("MAXENT: start\n")
        callback = _callback

    # logZ0 is not changing during minimization and is computed once.
    # it is only needed to compute Gamma
    shift0 = np.max(logW)  # shift to avoid overflow
    W0 = np.exp(logW - shift0)
    logZ0 = np.log(np.sum(W0)) + shift0

    if cuda:
        cu_logW=cm.CUDAMatrix(np.reshape(logW,(-1,1)))

    # function to be minimized
    def func(l):

        l = np.array(l)  # ensure array

        assert len(l) == len(fullreference)

        # takes care of >< constraints
        # vector ll only contains the Lagrangian multipliers to be applied on the trajectory
        if len(fullreference) != nobs:
            ll = np.zeros(nobs)
            s = 0
            for i in range(nobs):
                if box_const[i]:
                    # >< multipliers are summed
                    ll[i] = l[i+s] + l[i+s+1]
                    s += 1
                else:
                    ll[i] = l[i+s]
        else:
            ll = l


        if cuda:
            logZ, averages = _heavy_part(cu_logW, cu_traj, ll, cuda=cuda)
        else:
            logZ, averages = _heavy_part(logW, traj, ll, cuda=cuda)


        f = logZ - logZ0
        der = -averages


        if regularization is not None:
            reg = regularization(ll)
            f += reg[0]
            der += reg[1]

        if l2 is not None:
            f += 0.5*np.sum(l2*ll**2)
            der += l2*ll

        if l1 is not None:
            eee = 1e-50
            f += np.sum(l1*np.sqrt(ll**2+eee**2))
            der += l1*ll/np.sqrt(ll**2+eee**2)

        # takes care of >< constraints
        # vector der only contains the nobs elements
        # it is here extended
        if len(fullreference) != nobs:
            newder = np.zeros(len(fullreference))
            s = 0
            for i in range(nobs):
                if box_const[i]:
                    newder[i+s] = der[i]
                    newder[i+s+1] = der[i]
                    s += 1
                else:
                    newder[i+s] = der[i]
            der = newder

        # fullreference contains already nobs+nshift elements
        f += np.dot(l, fullreference)
        der += fullreference

        return(f, der)


    # With >< constraints the initial lambdas should be fixed
    if len(fullreference) != nobs:
        ll = np.zeros(len(fullreference))
        s = 0
        for i in range(nobs):
            if box_const[i]:
                if lambdas[i] >= 0:
                    ll[i+s+1] = lambdas[i]
                else:
                    ll[i+s] = lambdas[i]
                s += 1
            else:
                ll[i+s] = lambdas[i]
        lambdas = ll

    if maxiter is not None:
        if options is None:
            options = {}
        options["maxiter"] = maxiter

    res = minimize(
        func, lambdas, method=method, jac=True, callback=callback, bounds=bounds,
        tol=tol, options=options)

    # With >< constraints the final lambdas should be fixed
    if len(fullreference) != nobs:
        lambdas = np.zeros(nobs)
        s = 0
        for i in range(nobs):
            if box_const[i]:
                lambdas[i] = res.x[i+s]+res.x[i+s+1]
                s += 1
            else:
                lambdas[i] = res.x[i+s]
    else:
        lambdas = res.x

    # recompute weights at the end
    if cuda:
        logZ, averages, logW_ME = _heavy_part(cu_logW, cu_traj, lambdas, weights=True, cuda=cuda)
    else:
        logZ, averages, logW_ME = _heavy_part(logW, traj, lambdas, weights=True)

    if verbose:
        sys.stderr.write("MAXENT: end\n")

    return MaxentResult(
        logW_ME=logW_ME,
        lambdas=lambdas,
        averages=averages,
        gamma=res.fun,
        success=res.success,
        message=res.message,
        nfev=res.nfev,
        nit=res.nit
    )
