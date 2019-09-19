"""
Module implementing Lohman model for helicases.
"""

from collections.abc import Sequence
from typing import Union, Tuple, overload
import numpy as np

@overload
def lohman(t: float, ku: float, kd: float, n: int, boundaries: Tuple[float, float]) -> float:
    pass
@overload
def lohman(t: Sequence, ku: float, kd: float, n: int, boundaries: Tuple[float, float]) -> np.array:
    pass
@overload
def lohman(t: np.array, ku: float, kd: float, n: int, boundaries: Tuple[float, float]) -> np.array:
    pass

# actual implementation:
def lohman(t,
           ku: float,
           kd: float,
           n: int = 1,
           boundaries: Tuple[float, float] = (0.0, 1.0)
          ) -> Union[float, np.array]:
    """Lohman model for helicases.

       Compute the fraction of unwound helices as a function of time.
       See [Lucius et al, Biophys J 2003](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1303449/).

       Parameters
       ----------
       t: float or sequence or np.ndarray
           Time. If a sequence or np.ndarray is provided, the function is computed
           for all values and an array is returned.

       ku: float
           Unwinding rate.

       kd: float
           Dissociation rate.

       n: int, optional
           Step size

       boundaries: tuple with 2 elements
           Result is mapped to this range.

       Returns
       -------

       float or np.ndarray
           The fraction of unfolded helices at a given time `t`. If an array is provided for `t`,
           an array is returned containing the fractions at all the times.
           If `boundaries` is provided, the fraction is linearly mapped into the
           `boundaries[0], boundaries[1]` range.

    """
    if isinstance(t, Sequence):
        t = np.array(t, dtype=float)
    kobs = ku+kd
    # processivity
    P = ku/kobs
    f = 0*t
    fact = 1
    for r in range(n):
        f += (kobs*t)**r/fact
        fact *= r+1
    f *= np.exp(-kobs*t)
    f = P**n*(1-f)
    return boundaries[0] + (boundaries[1]-boundaries[0]) * f
