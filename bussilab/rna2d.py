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

def _test_viennarna_sc_add_up_subopt_is_patched():
    seq = "CGACGUACCGUUUUGCAAAGGCGUGGCGGCCCCCAUGAACAUUGACCGUCACUGUUUCCACGUAUGUUCU"
    baseline = RNA.fold_compound(seq).subopt(130)
    fc = RNA.fold_compound(seq)
    fc.sc_add_up(12, -0.01)

    # sc_add_up() contributes to unpaired nucleotides.  This is equivalent to
    # the wrapper's paired-nucleotide convention up to a structure-independent
    # energy shift, which does not affect the suboptimal range.
    expected_energies = {
        s.structure: s.energy - (0.01 if s.structure[11] == "." else 0.0)
        for s in baseline
    }
    best = min(expected_energies.values())
    expected = {
        structure
        for structure, energy in expected_energies.items()
        if energy - best <= 1.0001
    }

    solution = fc.subopt(100)
    got = [s.structure for s in solution]
    return (
        len(got) == len(set(got))
        and set(got) == expected
        and all(
            "%6.2f" % s.energy == "%6.2f" % fc.eval_structure(s.structure)
            for s in solution
        )
    )

# Variable storing the support for continuous lambdas
# It can be hard modified for testing (forced to True or False)
if RNA is not None:
    _SUPPORTS_NATIVE_CONTINUOUS=_test_native_continuous_support()
    _SUPPORTS_SUBOPT_SOFT_CONSTRAINTS=_test_viennarna_sc_add_up_subopt_is_patched()
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
    _SUPPORTS_SUBOPT_SOFT_CONSTRAINTS = False

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

    # Vanilla ViennaRNA has a bug that makes subopt with negative sc_add_up problematic.
    # Avoid negative sc_add_up values by excluding positive lambdas from the 1D scheme.

    if not _SUPPORTS_SUBOPT_SOFT_CONSTRAINTS:
        budget = 0.0

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

def _apply_hard_constraint(fc, paired, unpaired):

    for p in paired:
        fc.hc_add_bp_nonspecific(int(p+1), 0, RNA.CONSTRAINT_CONTEXT_ENFORCE | RNA.CONSTRAINT_CONTEXT_ALL_LOOPS)

    for p in unpaired:
        fc.hc_add_up(int(p+1), RNA.CONSTRAINT_CONTEXT_ALL_LOOPS)

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

class _DPMolecule:
    """
    Internal dynamic-programming implementation of a single RNA ensemble.
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
            _apply_hard_constraint(self._fc_rounded, paired=self._force_paired, unpaired=self._force_unpaired)

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

            _apply_hard_constraint(self._fc, paired=self._force_paired, unpaired=self._force_unpaired)

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
            self._rounded_total_free_energy = self._fc_rounded.pf()[1]
            self._rounded_total_free_energy += self._fc_rounded_shift
            self._fc_rounded_pf=True

    def __init__(
        self,
        seq: str,
        *,
        lambdas1d = None,
        temperature = 37 + _CELSIUS_TO_KELVIN,
        force_paired = None,
        force_unpaired = None,
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

        if force_paired is None:
           force_paired = []
        self._force_paired = force_paired

        if force_unpaired is None:
           force_unpaired = []
        self._force_unpaired = force_unpaired

        self._fc_rounded = None
        self._fc_rounded_shift = 0.0
        self._fc_rounded_pf = False
        self._rounded_total_free_energy = None
        self._fc = None
        self._fc_shift = 0.0

        self._mfe_energy = None
        self._mfe_structure = None

        self._base_pairing_probability = None
        self._total_free_energy = None

        self._pf_callback = None

    def _satisfies_hard_constraints(self, structure):
        """
        Return whether a structure satisfies this component's hard constraints.
        """
        return (
            all(structure[int(i)] != "." for i in self._force_paired)
            and all(structure[int(i)] == "." for i in self._force_unpaired)
        )

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

    def sample_rounding_correction(self):
        """
        Return the component normalization correction for sampled log weights.

        This dimensionless correction accounts for the difference between the
        exact and rounded partition functions. It is structure-independent within
        one ensemble, but generally differs between components and is therefore
        required when their samples are combined.
        """
        self._ensure_pf()
        self._ensure_rounded_pf()
        inverse_kT = 1.0 / (_KB * self._temperature)
        return float(
            (
                self._total_free_energy
                - self._rounded_total_free_energy
            ) * inverse_kT
        )

class Molecule:
    """
    RNA secondary-structure model with continuous pairing penalties.

    The class wraps one or more ViennaRNA dynamic-programming ensembles and
    supports continuous per-nucleotide pairing penalties. Multiple ensembles may
    be used to assign arbitrary energy biases to the paired/unpaired states of
    selected nucleotides.

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

    force_paired : array-like of int, optional
        Zero-based indices of nucleotides that are required to be paired, without
        specifying their pairing partners.

    force_unpaired : array-like of int, optional
        Zero-based indices of nucleotides that are required to be unpaired.

    state_positions : array-like of int, optional
        Zero-based indices defining binary paired/unpaired states. When provided,
        `state_biases` must contain one energy bias for every state.

    state_biases : array-like, optional
        Energy biases (kcal/mol) for the states defined by `state_positions`.
        Its shape must be `(2,) * len(state_positions)`, with index zero denoting
        an unpaired nucleotide and index one denoting a paired nucleotide.

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

    def __init__(
        self,
        seq: str,
        *,
        lambdas1d=None,
        temperature=37 + _CELSIUS_TO_KELVIN,
        force_paired = None,
        force_unpaired = None,
        state_positions=None,
        state_biases=None,
        NaCl=None,
        parameters="turner2004",
    ):
        if (state_positions is None) != (state_biases is None):
            raise ValueError(
                "state_positions and state_biases must be provided together"
            )

        base_force_paired = (
            [] if force_paired is None else list(force_paired)
        )
        base_force_unpaired = (
            [] if force_unpaired is None else list(force_unpaired)
        )

        if state_positions is None:
            self._state_positions = ()
            self._state_biases = np.zeros((), dtype=float)
        else:
            positions = np.asarray(state_positions)
            if positions.ndim != 1:
                raise ValueError("state_positions must be one-dimensional")
            if (
                positions.size
                and not np.issubdtype(positions.dtype, np.integer)
            ):
                raise ValueError("state_positions must contain integers")

            self._state_positions = tuple(int(i) for i in positions)
            if len(set(self._state_positions)) != len(self._state_positions):
                raise ValueError("state_positions must not contain duplicates")
            if any(
                i < 0 or i >= len(str(seq))
                for i in self._state_positions
            ):
                raise ValueError(
                    "state_positions must contain valid nucleotide indices"
                )

            fixed_positions = (
                set(base_force_paired) | set(base_force_unpaired)
            )
            if fixed_positions.intersection(self._state_positions):
                raise ValueError(
                    "state_positions must not overlap force_paired or "
                    "force_unpaired"
                )

            expected_shape = (2,) * len(self._state_positions)
            self._state_biases = np.asarray(
                state_biases,
                dtype=float,
            ).copy()
            if self._state_biases.shape != expected_shape:
                raise ValueError(
                    f"state_biases must have shape {expected_shape}"
                )
            if not np.all(np.isfinite(self._state_biases)):
                raise ValueError(
                    "state_biases must contain only finite values"
                )

        self._states = list(np.ndindex(self._state_biases.shape))
        self._dp_molecules = []

        for state in self._states:
            state_paired = [
                position
                for position, value in zip(self._state_positions, state)
                if value
            ]
            state_unpaired = [
                position
                for position, value in zip(self._state_positions, state)
                if not value
            ]
            self._dp_molecules.append(
                _DPMolecule(
                    seq,
                    lambdas1d=lambdas1d,
                    temperature=temperature,
                    NaCl=NaCl,
                    force_paired=base_force_paired + state_paired,
                    force_unpaired=base_force_unpaired + state_unpaired,
                    parameters=parameters,
                )
            )

        first_molecule = self._dp_molecules[0]
        self._seq = first_molecule._seq
        self._lambdas1d = first_molecule._lambdas1d.copy()
        self._temperature = first_molecule._temperature
        self._salt = first_molecule._salt
        self._parameters = first_molecule._parameters
        self._force_paired = tuple(base_force_paired)
        self._force_unpaired = tuple(base_force_unpaired)

    def _condition_unpaired(self, position):
        """
        Return an equivalent molecule conditioned on one nucleotide being
        unpaired.
        """
        if position in self._force_paired:
            return None
        if position in self._force_unpaired:
            return self

        force_unpaired = self._force_unpaired + (position,)
        state_positions = self._state_positions
        state_biases = self._state_biases

        if position in state_positions:
            axis = state_positions.index(position)
            state_positions = (
                state_positions[:axis] + state_positions[axis + 1:]
            )
            state_biases = np.take(state_biases, 0, axis=axis)

        return Molecule(
            self._seq,
            lambdas1d=self._lambdas1d,
            temperature=self._temperature,
            force_paired=self._force_paired,
            force_unpaired=force_unpaired,
            state_positions=state_positions,
            state_biases=state_biases,
            NaCl=self._salt,
            parameters=self._parameters,
        )

    def _component_probabilities(self):
        """
        Return the total free energy and normalized component probabilities.
        """
        free_energies = np.array([
            molecule.total_free_energy()
            for molecule in self._dp_molecules
        ])
        biased_free_energies = (
            free_energies + self._state_biases.ravel()
        )

        finite = np.isfinite(biased_free_energies)
        if not np.any(finite):
            raise RuntimeError(
                "No state has a finite partition function"
            )

        reference = np.min(biased_free_energies[finite])
        kT = _KB * self._dp_molecules[0]._temperature
        relative_weights = np.zeros(len(self._dp_molecules))
        relative_weights[finite] = np.exp(
            -(biased_free_energies[finite] - reference) / kT
        )
        normalization = math.fsum(relative_weights)
        probabilities = relative_weights / normalization
        total_free_energy = reference - kT * math.log(normalization)

        return float(total_free_energy), probabilities

    def _component_mfes(self):
        """
        Return component MFE structures and their biased energies.
        """
        structures = []
        energies = []
        for molecule, bias in zip(
            self._dp_molecules,
            self._state_biases.ravel(),
        ):
            structure, energy = molecule.mfe()
            structures.append(structure)
            if molecule._satisfies_hard_constraints(structure):
                energies.append(energy + bias)
            else:
                energies.append(math.inf)

        energies = np.asarray(energies)
        if not np.any(np.isfinite(energies)):
            raise RuntimeError(
                "No state has a feasible MFE structure"
            )

        return structures, energies

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
        # Preserve the original direct MFE path when no state mixture is
        # present.
        if len(self._dp_molecules) == 1:
            structure, energy = self._dp_molecules[0].mfe()
            return (
                structure,
                float(energy + self._state_biases.item()),
            )

        structures, energies = self._component_mfes()
        index = int(np.argmin(energies))
        return structures[index], float(energies[index])

    def base_pairing_probability(self):
        """
        Return the base-pairing probability matrix.

        Returns
        -------
        ndarray
            Symmetric NxN matrix whose element (i,j) is the equilibrium
            probability that nucleotides i and j form a base pair.
        """
        _, probabilities = self._component_probabilities()
        matrix = np.zeros_like(
            self._dp_molecules[0].base_pairing_probability()
        )
        for probability, molecule in zip(
            probabilities,
            self._dp_molecules,
        ):
            if probability == 0.0:
                continue
            matrix += (
                probability * molecule.base_pairing_probability()
            )
        return matrix

    def total_free_energy(self):
        """
        Return the ensemble free energy.

        Returns
        -------
        float
            Ensemble free energy (kcal/mol) corresponding to the exact continuous
            pairing penalties.
        """
        total_free_energy, _ = self._component_probabilities()
        return total_free_energy

    def suboptimal_structures(self, delta):
        """
        Enumerate suboptimal secondary structures.

        Candidate structures are generated using ViennaRNA's rounded soft
        constraints, rescored with the exact continuous penalties, and returned
        sorted by exact energy.

        For state mixtures, the MFE of every feasible state is calculated first.
        Each component is then enumerated with the local energy window required
        by the common, globally biased cutoff.

        Parameters
        ----------
        delta : float
            Maximum energy difference (kcal/mol) above the exact MFE.

        Returns
        -------
        list of (str, float)
            List of (structure, energy) pairs sorted by increasing exact energy.
        """
        # Preserve the original direct enumeration path when no state mixture is
        # present.
        if len(self._dp_molecules) == 1:
            suboptimal = self._dp_molecules[0].suboptimal_structures(
                delta
            )
            bias = float(self._state_biases.item())
            if bias == 0.0:
                return suboptimal
            return [
                (structure, float(energy + bias))
                for structure, energy in suboptimal
            ]

        delta = float(delta)
        if not np.isfinite(delta) or delta < 0.0:
            raise ValueError("delta must be finite and non-negative")

        _, component_mfe_energies = self._component_mfes()
        global_mfe_energy = float(np.min(component_mfe_energies))
        cutoff = global_mfe_energy + delta
        merged = []

        for molecule, bias, component_mfe_energy in zip(
            self._dp_molecules,
            self._state_biases.ravel(),
            component_mfe_energies,
        ):
            if not np.isfinite(component_mfe_energy):
                continue

            local_delta = cutoff - component_mfe_energy
            if local_delta < 0.0:
                continue

            for structure, energy in molecule.suboptimal_structures(
                local_delta
            ):
                biased_energy = float(energy + bias)
                if biased_energy <= cutoff:
                    merged.append((structure, biased_energy))

        merged.sort(key=lambda item: item[1])
        return merged

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

            For state mixtures, the log-weight also includes the
            component-specific exact/rounded normalization correction. This
            correction is caused only by lambda rounding, not by the state bias.

            When all lambdas are multiples of 0.01 kcal/mol, every returned
            log-weight is zero.
        """
        number = int(number)
        if number <= 0:
            raise ValueError("number must be a positive integer")

        # Preserve the original direct sampling path when no state mixture is
        # present.
        if len(self._dp_molecules) == 1:
            return self._dp_molecules[0].sample(number)

        _, probabilities = self._component_probabilities()
        component_indices = np.random.choice(
            len(self._dp_molecules),
            size=number,
            p=probabilities,
        )
        result = [None] * number
        for component_index, molecule in enumerate(self._dp_molecules):
            output_indices = np.flatnonzero(
                component_indices == component_index
            )
            if not len(output_indices):
                continue

            samples = molecule.sample(len(output_indices))
            rounding_correction = (
                molecule.sample_rounding_correction()
            )

            for output_index, (structure, log_weight) in zip(
                output_indices,
                samples,
            ):
                result[output_index] = (
                    structure,
                    float(log_weight + rounding_correction),
                )

        return result

    def suboptimal_coverage(self, delta):
        """
        Return the fraction of the partition function represented by the
        suboptimal ensemble.

        The suboptimal ensemble contains all structures whose exact energy is
        within ``delta`` kcal/mol of the minimum-free-energy structure.

        Parameters
        ----------
        delta : float
            Maximum energy difference (kcal/mol) above the exact MFE.

        Returns
        -------
        float
            Fraction of the total partition function represented by the
            enumerated structures. The result lies between zero and one, apart
            from possible small numerical errors.
        """
        suboptimal = self.suboptimal_structures(delta)

        reference_energy = suboptimal[0][1]
        inverse_kT = 1.0 / (_KB * self._temperature)

        relative_partition_function = math.fsum(
            math.exp(-(energy - reference_energy) * inverse_kT)
            for _, energy in suboptimal
        )

        log_coverage = (
            math.log(relative_partition_function)
            + (self.total_free_energy() - reference_energy) * inverse_kT
        )

        coverage = float(math.exp(log_coverage))

        if coverage > 1.0:
            if coverage <= 1.0 + 1e-5:
                coverage = 1.0
            else:
                raise RuntimeError(
                    f"Suboptimal coverage is unexpectedly larger than one: "
                    f"{coverage}"
                )

        return coverage

    def _pairing_correlation_matrix_pf(self):
        """
        Return the exact joint pairing-probability matrix.
        """
        p = np.sum(self.base_pairing_probability(), axis=1)
        p = np.clip(p, 0.0, 1.0)
        p[np.asarray(self._force_paired, dtype=int)] = 1.0
        p[np.asarray(self._force_unpaired, dtype=int)] = 0.0

        n = len(self._seq)
        conditional = np.empty((n, n))

        for i in range(n):
            if p[i] == 0.0:
                conditional[i, :] = p
                continue
            if p[i] == 1.0:
                conditional[i, :] = 0.0
                continue

            conditioned = self._condition_unpaired(i)
            conditional[i, :] = np.sum(
                conditioned.base_pairing_probability(),
                axis=1,
            )

        matrix = np.empty((n, n))
        for i in range(n):
            matrix[i, :] = p - (1.0 - p[i]) * conditional[i, :]

        # Since s_i**2 = s_i, the diagonal must equal P(s_i = 1).
        np.fill_diagonal(matrix, p)

        # Remove small numerical asymmetries from independent constrained PF runs.
        return 0.5 * (matrix + matrix.T)

    def pairing_correlation_matrix(self):
        """
        Return the joint pairing-probability matrix.

        Element ``(i, j)`` is the probability that nucleotides ``i`` and ``j``
        are simultaneously paired, irrespective of their pairing partners.

        The matrix is computed exactly using constrained partition-function
        calculations.

        Returns
        -------
        ndarray
            Symmetric NxN matrix whose diagonal contains the pairing
            probabilities.
        """
        return self._pairing_correlation_matrix_pf()
