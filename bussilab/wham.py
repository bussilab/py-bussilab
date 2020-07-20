"""
Module containing a WHAM implementation.

See `bussilab.wham.wham()`.

"""
import sys
from typing import Optional
import numpy as np
from . import coretools

class WhamResult(coretools.Result):
    """Result of a `bussilab.wham.wham` calculation."""
    def __init__(self,
                 *,
                 logW: np.ndarray,
                 logZ: np.ndarray,
                 nit: int,
                 eps: float):
        super().__init__()
        self.logW = logW
        """`numpy.ndarray` containing the logarithm of the weight of the frames."""
        self.logZ = logZ
        """`numpy.ndarray` containing the logarithm of the partition function of each state."""
        self.nit = nit
        """The number of performed iterations."""
        self.eps = eps
        """The final error in the iterative solution."""

def wham(bias,
        *,
        frame_weight=None,
        traj_weight=None,
        T: float = 1.0,
        maxiter: int = 1000,
        threshold: float = 1e-40,
        verbose: bool = False,
        logZ: Optional[np.ndarray] = None,
        logW: Optional[np.ndarray] = None):
    """Compute weights according to binless WHAM.

       The main input for this calculation is in the 2D array `bias`.
       Element `bias[i, j]` should contain the energy of the i-th frame computed
       according to the j-th Hamiltonian. Trajectories should be concatenated first,
       so that the total number of frames should be equal to the number of frames
       of each trajectory multiplied by the number of trajectories.
       However, it is also possible to contatenate simulations of different lengths.
       It is crucial however to compute the potential according to each of the employed
       Hamiltonian on **all the frames**, not only on those simulated using that Hamiltonian.

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
           order to make it adimensional.

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

    """

    # allow tuples or lists
    bias = coretools.ensure_np_array(bias)
    frame_weight = coretools.ensure_np_array(frame_weight)
    traj_weight = coretools.ensure_np_array(traj_weight)

    nframes = bias.shape[0]
    ntraj = bias.shape[1]

    # default values
    if frame_weight is None:
        frame_weight = np.ones(nframes)
    if traj_weight is None:
        traj_weight = np.ones(ntraj)

    assert len(traj_weight) == ntraj
    assert len(frame_weight) == nframes

    # do exponentials only once
    expv = np.exp((-bias+np.min(bias))/T)

    if logW is not None:
        Z = np.matmul(np.exp(logW), expv)
        Z /= np.sum(Z*traj_weight)
    elif logZ is not None:
        Z = np.exp(logZ)
    else:
        Z = np.ones(ntraj)

    Zold = np.copy(Z)

    if verbose:
        sys.stderr.write("WHAM: start\n")
    for nit in range(maxiter):
        # find unnormalized weights
        weight = 1.0/np.matmul(expv, traj_weight/Z)*frame_weight
        # normalized them
        weight /= np.sum(weight)
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
    if verbose:
        sys.stderr.write("WHAM: end")

    return WhamResult(logW=np.log(weight), logZ=np.log(Z), nit=nit, eps=eps)
