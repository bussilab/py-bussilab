"""
Module containing an RNA secondary-structure model with continuous pairing penalties.

See `bussilab.rna2d.Molecule()`.

"""

import numpy as np
import threading
import math
import warnings
import sys

try:
    import RNA
except ImportError:
    # we make sure the module can be imported even if ViennaRNA is not installed
    RNA = None

def _require_viennarna():
    if RNA is None:
        raise ImportError(
            "The ViennaRNA Python package is required "
            "to use bussilab.rna2d."
        )


# Rounding factor in viennaRNA
# Might be increased to mimic more severe rounding.
# However, it should be a multiple of the true internal rounding (0.01)
_ROUNDING_FACTOR=0.01

# Boltzmann constant, as obtained from vienna source code
_KB = 1.98717/1000

# Shift from Celsius to Kelvin, as obtained from vienna source code
_CELSIUS_TO_KELVIN = 273.15

# params_load_RNA_* are not thread safe and require a global lock
_THERMODYNAMIC_PARAMETERS_LOCK = threading.Lock()

# Fraction of the floating-point overflow threshold allowed for
# accumulated negative unpaired pseudoenergies.
# This choice is very conservative for double precision (standard)
# vienna builds, and should also enable single precision (custom)
# vienna builds

_PARTITION_OVERFLOW_SAFETY_FRACTION = 0.1

# Can be used to make wider searches in mfe and subopt calculations
# with residuals. Should never be necessary, it is here for debugging purposes
# Integer, using the internal vienna units (0.01 kcal/mol)
_DEBUG_WIDEN_SEARCH = 0

if RNA is not None:
# Dictionary of available parameters
    _THERMODYNAMIC_PARAMETERS = {
        "turner1999" : RNA.params_load_RNA_Turner1999,
        "turner2004" : RNA.params_load_RNA_Turner2004,
        "andronescu2007" : RNA.params_load_RNA_Andronescu2007,
        "langdon2018" : RNA.params_load_RNA_Langdon2018
    }
else:
    _THERMODYNAMIC_PARAMETERS = {}

def _test_native_continuous_support():
    """
    Return whether static ViennaRNA soft constraints preserve fractional
    energies in partition-function calculations.
    """
    _require_viennarna()
    seq="GCGCAAAAGCGC"
    with _THERMODYNAMIC_PARAMETERS_LOCK:
        # make sure we use turner2004 parameters
        RNA.params_load_RNA_Turner2004()
        fc=RNA.fold_compound(seq)

    F0=fc.pf()[1]
    for i in range(5,9):
        fc.sc_add_up(i+1,+0.004)

    F1=fc.pf()[1]

    for i in range(4):
        fc.sc_add_bp(i+1,len(seq)-i,0.004)
    F2=fc.pf()[1]

    return F1-F0>1e-3 and F2-F1>1e-3

# Variable storing the support for continuous lambdas
# It can be hard modified for testing (forced to True or False)
if RNA is not None:
    _SUPPORTS_NATIVE_CONTINUOUS=_test_native_continuous_support()
    # A warning is issued upon import
    if not _SUPPORTS_NATIVE_CONTINUOUS:
        warnings.warn(
            "ViennaRNA does not support continuous soft constraints; "
            "using Python callback. "
            "Performance with soft constraints that are not multiple of "
            "0.01 kcal/mol can be significanly improved using a ViennaRNA "
            "patch available at https://github.com/bussilab/ViennaRNA-patches.",
            RuntimeWarning,
    )
else:
    _SUPPORTS_NATIVE_CONTINUOUS = False


# internal variable according to Vienna rules
if RNA is not None:
    _PAIR_DECOMPOSITIONS = {
        RNA.DECOMP_PAIR_HP,
        RNA.DECOMP_PAIR_IL,
        RNA.DECOMP_PAIR_ML,
    }
else:
    _PAIR_DECOMPOSITIONS = set()

def _apply_residual_callback(fc, residuals, kT):
    """
    Add continuous exact-minus-rounded pairing penalties to PF recursions.

    Returns the callback object, which must be kept alive while `fc` is used.
    """
    residuals = np.asarray(residuals, dtype=float)

    if np.max(np.abs(residuals)) <= 10 * sys.float_info.epsilon:
        return None

    def callback(i, j, k, l, decomposition, data):
        del k, l, data

        if decomposition not in _PAIR_DECOMPOSITIONS:
            return 1.0

        delta_energy = residuals[i - 1] + residuals[j - 1]
        return math.exp(-delta_energy / kT)

    fc.sc_add_exp_f(callback)
    return callback

def _apply_constraint(fc,lambdas,kT):
    """
    Apply per-nucleotide pairing penalties using a hybrid 1D/2D scheme.
    Returns the structure-independent energy shift that must be added to reported
    energies and free energies.
    """

    n = len(lambdas)

    # The dangerous factor is approximately
    #
    #     exp(sum(lambda_positive_1d) / kT).
    #
    # Stay well below the largest representable double.
    budget = _PARTITION_OVERFLOW_SAFETY_FRACTION * kT * math.log(sys.float_info.max)

    # All negative lambdas use the safe 1D representation.
    use_1d = lambdas < 0.0

    # Among positive lambdas, use as many as possible in 1D.
    # Sorting ascending maximizes their number under the budget.
    positive_indices = np.flatnonzero(lambdas > 0.0)
    positive_indices = positive_indices[
        np.argsort(lambdas[positive_indices])
    ]

    cumulative = np.cumsum(lambdas[positive_indices])
    positive_1d = positive_indices[cumulative <= budget]
    use_1d[positive_1d] = True

    # Everything else positive is represented directly as pair energies.
    use_2d = (lambdas > 0.0) & ~use_1d

    shift = 0.0

    # 1D representation:
    #
    # lambda * I_paired = lambda - lambda * I_unpaired

    n_1d = 0
    for i in np.flatnonzero(use_1d):
        n_1d += 1
        value = float(lambdas[i])
        fc.sc_add_up(int(i) + 1, -value)
        shift += value

    # 2D representation. Add each pair constraint only once, combining
    # contributions from both endpoints.
    lambda_2d = np.where(use_2d, lambdas, 0.0)

    n_2d = 0
    for i in range(n):
        for j in range(i + 1, n):
            value = float(lambda_2d[i] + lambda_2d[j])

            if value != 0.0:
                n_2d += 1
                fc.sc_add_bp(i + 1, j + 1, value)

    return shift, n_1d, n_2d

def _correct_rounding_energy(structure, dlambdas):
    """
    Compute the energy correction associated with residuals for
    a dot-bracket structure.
    """
    correction = 0.0
    for c, d in zip(structure, dlambdas):
        if c != ".":
            correction += d
    return float(correction)

class Molecule:
    """
    RNA secondary-structure model with continuous pairing penalties.

    The class wraps ViennaRNA and supports continuous per-nucleotide pairing
    penalties while preserving compatibility with ViennaRNA's dynamic programming
    algorithms.

    A penalty λᵢ is added whenever nucleotide *i* is paired. Internally, these
    penalties are automatically represented as an equivalent hybrid combination of
    unpaired and pair soft constraints. This representation avoids numerical
    overflows in partition-function calculations while preserving the requested
    thermodynamic model.

    Partition-function calculations use the exact continuous penalties. Minimum-
    free-energy and suboptimal structure prediction use ViennaRNA's rounded soft
    constraints to generate candidate structures, which are then rescored using the
    exact continuous penalties.

    Parameters
    ----------
    seq : str
        RNA sequence.

    lambdas1d : array-like, optional
        Per-nucleotide pairing penalties (kcal/mol). Positive values penalize
        pairing, whereas negative values favor pairing. If omitted, all penalties
        are zero.

    T : float, default=310.15
        Temperature in kelvin.

    NaCl : float or None, default=None
        Sodium concentration (M). If None, ViennaRNA's default value is used.

    parameters : {"turner1999", "turner2004", "andronescu2007", "langdon2018"}
        Thermodynamic parameter set.

    Notes
    -----
    Default ViennaRNA builds round soft constraints to the nearest 0.01 kcal/mol in
    partition-function calculations. When continuous soft constraints are not
    natively supported, this class automatically applies a lightweight Python
    callback to recover the exact continuous model.
    """

    def _make_md_params(self):
        """
        Internal utility to generate an md params object.
        """
        md = RNA.md()
        md.uniq_ML = 1
        md.temperature = self._temperature - _CELSIUS_TO_KELVIN
        if self._salt is not None:
            md.salt = self._salt
        return md

    def _make_fold_compound(self):
        """
        Internal utility to create a fold compound.
        """
        with _THERMODYNAMIC_PARAMETERS_LOCK:
            if not _THERMODYNAMIC_PARAMETERS[self._parameters]():
                raise RuntimeError(
                    f"Could not load thermodynamic parameters "
                    f"{self._parameters!r}"
                )
            return RNA.fold_compound(self._seq, self._make_md_params())

    def _ensure_fc_rounded(self):
        """
        Internal utility to ensure that the fold compound using rounded lambdas
        has been initialized.
        """
        if self._fc_rounded is None:
            self._fc_rounded = self._make_fold_compound()
            (self._fc_rounded_shift,
             self._fc_rounded_n_1d_constraints,
             self._fc_rounded_n_2d_constraints) = _apply_constraint(self._fc_rounded, self._lambdas1d_rounded, _KB * self._temperature)

    def _ensure_fc(self):
        """
        Internal utility to ensure that the fold compound using continuous lambdas
        has been initialized.
        """
        if self._fc is None:
            self._fc = self._make_fold_compound()

            if _SUPPORTS_NATIVE_CONTINUOUS:
                use_lambdas = self._lambdas1d
            else:
                # when using standard vienna builds without support for continuous lambdas
                # this fold compound is constructed using rounded lambdas
                # and residuals are added with a (slow) callback function
                use_lambdas = self._lambdas1d_rounded

            (self._fc_shift,
            self._fc_n_1d_constraints,
            self._fc_n_2d_constraints) = _apply_constraint(self._fc, use_lambdas, _KB * self._temperature)

            if _SUPPORTS_NATIVE_CONTINUOUS:
                self._pf_callback = None
            else:
                # note that the callback only applies the residuals
                self._pf_callback = _apply_residual_callback(self._fc, self._lambdas1d_residuals, _KB * self._temperature)

    def _ensure_pf(self):
        """
        Internal utility to ensure that the partition function (energy and bpp)
        have been calculated.
        """
        if self._base_pairing_probability is None:
            self._ensure_fc()

            # here we use the native mfe for two reasons:
            # - our self.mfe() is shifted due to constraints
            # - its calculation might be slow because of the internal use of subopt
            # in any case, an approximate mfe calculation is sufficient for this purpose
            mfe = self._fc.mfe()[1]
            self._fc.exp_params_rescale(mfe)

            self._total_free_energy = self._fc.pf()[1]
            # correction for using bp instead of up
            self._total_free_energy += self._fc_shift

            bpp = np.array(self._fc.bpp())[1:,1:]
            # matrix is made symmetric
            self._base_pairing_probability = bpp + bpp.T

    def _ensure_rounded_pf(self):
        """
        Internal utility to ensure that the partition function calculation
        with rounded lambdas has been done. Note that this is needed only for
        backtracking (see sample()).
        """
        if not self._fc_rounded_pf:
            self._ensure_fc_rounded()
            native_mfe = self._fc_rounded.mfe()[1]
            self._fc_rounded.exp_params_rescale(native_mfe)
            self._fc_rounded.pf()
            self._fc_rounded_pf=True

    def __init__(
        self,
        seq: str,
        *,
        lambdas1d = None,
        temperature = 37 + _CELSIUS_TO_KELVIN,
        NaCl = None,
        parameters = "turner2004"):

        _require_viennarna()

        self._parameters = str(parameters).lower()
        if not self._parameters in _THERMODYNAMIC_PARAMETERS:
            raise ValueError(f"Thermodynamic parameters {parameters} not known")

        self._seq = str(seq).upper()

        if not self._seq:
            raise ValueError("seq cannot be empty")

        if any(base not in "ACGU" for base in self._seq):
            raise ValueError("seq must contain only A, C, G, and U")

        self._temperature = temperature

        if not self._temperature >= 0.0:
            raise ValueError(f"Temperature {self._temperature} should be positive")

        self._salt = NaCl

        if self._salt is not None and not self._salt >= 0.0:
            raise ValueError(f"Salt concentration {self._salt} should be positive")

        if lambdas1d is None:
            lambdas1d = np.zeros(len(seq))
        self._lambdas1d = np.asarray(lambdas1d, dtype=float).copy()
        self._lambdas1d_rounded = _ROUNDING_FACTOR*np.rint(self._lambdas1d/_ROUNDING_FACTOR)
        self._lambdas1d_residuals = self._lambdas1d - self._lambdas1d_rounded

        if self._lambdas1d.ndim != 1:
            raise ValueError("lambdas1d must be one-dimensional")

        if len(self._lambdas1d) != len(self._seq):
            raise ValueError(
                "lambdas1d must contain one value per nucleotide"
            )

        if not np.all(np.isfinite(self._lambdas1d)):
            raise ValueError("lambdas1d must contain only finite values")

        self._lambdas1d_residuals_range=np.sum(np.abs(self._lambdas1d_residuals))

        self._fc_rounded = None
        self._fc_rounded_shift = 0.0
        self._fc_rounded_pf = False
        self._fc = None
        self._fc_shift = 0.0

        self._mfe_energy = None
        self._mfe_structure = None

        self._base_pairing_probability = None
        self._total_free_energy = None

        self._pf_callback = None


    def mfe(self):
        """
        Return the minimum-free-energy structure.

        The returned energy always corresponds to the exact continuous pairing
        penalties, even when ViennaRNA internally rounds soft constraints.

        Returns
        -------
        structure : str
            Dot-bracket representation of the MFE structure.

        energy : float
            Exact free energy (kcal/mol).
        """
        if self._mfe_energy is None:
            self._ensure_fc_rounded()

            subopt_range = int(2*self._lambdas1d_residuals_range / _ROUNDING_FACTOR  + 0.5 ) + _DEBUG_WIDEN_SEARCH

            if subopt_range == 0:
                self._mfe_structure , self._mfe_energy = self._fc_rounded.mfe()
                # correction for rounding
                self._mfe_energy += _correct_rounding_energy(self._mfe_structure,self._lambdas1d_residuals)
            else:
                subopt = self._fc_rounded.subopt(subopt_range)
                energies=[
                    _correct_rounding_energy(s.structure, self._lambdas1d_residuals) + s.energy for s in subopt
                ]
                index = np.argmin(energies)
                self._mfe_structure , self._mfe_energy = subopt[index].structure , energies[index]



            # correction for using bp instead of up
            self._mfe_energy += self._fc_rounded_shift

        return self._mfe_structure, float(self._mfe_energy)

    def base_pairing_probability(self):
        """
        Return the base-pairing probability matrix.

        Returns
        -------
        ndarray
            Symmetric NxN matrix whose element (i,j) is the equilibrium
            probability that nucleotides i and j form a base pair.
        """
        self._ensure_pf()
        return self._base_pairing_probability.copy()

    def total_free_energy(self):
        """
        Return the ensemble free energy.

        Returns
        -------
        float
            Ensemble free energy (kcal/mol) corresponding to the exact continuous
            pairing penalties.
        """
        self._ensure_pf()
        return float(self._total_free_energy)

    def suboptimal_structures(self,delta):
        """
        Enumerate suboptimal secondary structures.

        Candidate structures are generated using ViennaRNA's rounded soft
        constraints, rescored with the exact continuous penalties, and returned
        sorted by exact energy.

        Parameters
        ----------
        delta : float
            Maximum energy difference (kcal/mol) above the exact MFE.

        Returns
        -------
        list of (str, float)
            List of (structure, energy) pairs sorted by increasing exact energy.
        """

        delta = float(delta)
        if not np.isfinite(delta) or delta < 0.0:
            raise ValueError("delta must be finite and non-negative")

        self._ensure_fc_rounded()

        subopt_range = int(2*self._lambdas1d_residuals_range / _ROUNDING_FACTOR + 0.5) + _DEBUG_WIDEN_SEARCH

        subopt = self._fc_rounded.subopt(int(delta / _ROUNDING_FACTOR +0.5) +subopt_range)

        energies=[_correct_rounding_energy(s.structure, self._lambdas1d_residuals) + s.energy for s in subopt]
        index = np.argsort(energies)
        return [
            (subopt[i].structure , float(energies[i] + self._fc_rounded_shift))
            for i in index
            if energies[i]-energies[index[0]] <= delta
        ]

    def sample(self, number):
        """
        Generate Boltzmann-distributed secondary structures.

        Structures are sampled from the rounded ViennaRNA model. For each sampled
        structure, the returned log-weight corrects the rounded distribution to the
        exact continuous-lambda distribution by importance sampling.

        The returned log-weights are intentionally left unnormalized so that
        independent samples can be concatenated and optionally deduplicated before
        normalization.

        Parameters
        ----------
        number : int
            Number of structures to sample.

        Returns
        -------
        list of (str, float)
            Each element contains a dot-bracket structure and its unnormalized
            log-weight correction

                log(w) = -(E_exact - E_rounded) / (k_B T).

            When all lambdas are multiples of 0.01 kcal/mol, every returned
            log-weight is zero.
        """

        number = int(number)
        if number <= 0:
            raise ValueError("number must be a positive integer")
        self._ensure_rounded_pf()

        structures = self._fc_rounded.pbacktrack(number)
        inverse_kT = 1.0 / (_KB * self._temperature)

        return [
            (structure, -float(_correct_rounding_energy(structure, self._lambdas1d_residuals) * inverse_kT))
            for structure in structures
        ]


