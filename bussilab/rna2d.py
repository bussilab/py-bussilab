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

# Maximum assumed candidates-per-accepted-sample ratio used when choosing the
# next rejection-sampling batch.
_MAX_ASSUMED_INFLATION = 1.1

# Canonical and wobble pairs, including both sequence orientations.
_ALLOWED_PAIRS = frozenset(("AU", "UA", "CG", "GC", "GU", "UG"))

# Boltzmann constant, as obtained from vienna source code
_KB = 1.98717/1000

# Shift from Celsius to Kelvin, as obtained from vienna source code
_CELSIUS_TO_KELVIN = 273.15

def _dotbracket_to_pairtable(structure):
    """Convert one dot-bracket structure to a zero-based pair table."""
    if not isinstance(structure, str):
        raise ValueError("structures must be strings")
    if len(structure) > np.iinfo(np.int16).max + 1:
        raise ValueError("structure is too long for an int16 pair table")

    table = np.full(len(structure), -1, dtype=np.int16)
    stack = []
    for i, character in enumerate(structure):
        if character == "(":
            stack.append(i)
        elif character == ")":
            if not stack:
                raise ValueError("structure contains unmatched parentheses")
            j = stack.pop()
            table[i] = j
            table[j] = i
        elif character != ".":
            raise ValueError(
                "structures must use '.', '(', and ')' dot-bracket symbols"
            )

    if stack:
        raise ValueError("structure contains unmatched parentheses")
    return table

def _structures_to_pairtables(structures):
    """Convert equally sized dot-bracket structures to pair-table rows."""
    tables = [_dotbracket_to_pairtable(structure) for structure in structures]
    length = len(tables[0])
    if any(len(table) != length for table in tables[1:]):
        raise ValueError("all structures must have the same length")
    return np.asarray(tables, dtype=np.int16)

def _normalize_logweights(logweights):
    """Return finite log weights normalized to unit exponential sum."""
    logweights = np.asarray(logweights, dtype=float)
    if not np.all(np.isfinite(logweights)):
        raise ValueError("log weights must be finite")
    normalization = np.logaddexp.reduce(logweights)
    return logweights - normalization

def sample_to_numpy(samples, *, deduplicate=True):
    """
    Convert sampled dot-bracket structures to NumPy arrays.

    The input may be either a sequence of structure strings, as returned by
    ``Molecule.sample(weights=False)``, or a sequence of ``(structure,
    log_weight)`` pairs, as returned by ``Molecule.sample(weights=True)``.
    Mixed inputs are rejected.

    Structures are represented by zero-based pair tables: ``states[k, i]`` is
    the index paired with nucleotide ``i`` in structure ``k``, or ``-1`` when
    it is unpaired. This is a NumPy-oriented adaptation of ViennaRNA's pair
    table format. A row ``table`` can be converted to ViennaRNA's convention
    with ``np.concatenate(([len(table)], table + 1))``: ViennaRNA prepends the
    sequence length, uses one-based partner indices, and uses zero for an
    unpaired nucleotide.

    Parameters
    ----------
    samples : sequence of str or sequence of (str, float)
        Unweighted structures or structures with unnormalized log weights.

    deduplicate : bool, default=True
        If True, merge identical structures. Unweighted occurrences are
        combined through their counts; supplied log weights are combined using
        logarithmic addition. If False, retain every occurrence.

    Returns
    -------
    states : ndarray of int16, shape (n_structures, sequence_length)
        Zero-based pair tables, with ``-1`` denoting an unpaired nucleotide.

    logweights : ndarray of float, shape (n_structures,)
        Normalized log weights, satisfying ``sum(exp(logweights)) == 1`` up to
        floating-point precision.
    """
    if not isinstance(deduplicate, (bool, np.bool_)):
        raise ValueError("deduplicate must be a boolean")

    samples = list(samples)
    if not samples:
        raise ValueError("samples cannot be empty")

    unweighted = all(isinstance(item, str) for item in samples)
    weighted = all(
        not isinstance(item, str)
        and hasattr(item, "__len__")
        and len(item) == 2
        and isinstance(item[0], str)
        for item in samples
    )
    if not (unweighted or weighted):
        raise ValueError(
            "samples must contain either strings or (structure, log_weight) "
            "pairs, without mixing the two forms"
        )

    if unweighted:
        structures = samples
        input_logweights = np.zeros(len(samples), dtype=float)
    else:
        structures = [item[0] for item in samples]
        try:
            input_logweights = np.asarray(
                [item[1] for item in samples],
                dtype=float,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("log weights must be real numbers") from error
        if input_logweights.ndim != 1:
            raise ValueError("log weights must be scalar")
        if not np.all(np.isfinite(input_logweights)):
            raise ValueError("log weights must be finite")

    if deduplicate:
        unique_structures = []
        unique_logweights = []
        indices = {}
        for structure, logweight in zip(structures, input_logweights):
            if structure in indices:
                index = indices[structure]
                unique_logweights[index] = np.logaddexp(
                    unique_logweights[index],
                    logweight,
                )
            else:
                indices[structure] = len(unique_structures)
                unique_structures.append(structure)
                unique_logweights.append(float(logweight))
        structures = unique_structures
        input_logweights = np.asarray(unique_logweights, dtype=float)

    states = _structures_to_pairtables(structures)
    return states, _normalize_logweights(input_logweights)

def suboptimal_to_numpy(suboptimal, temperature):
    """
    Convert suboptimal dot-bracket structures and energies to NumPy arrays.

    Structures use the same zero-based pair-table representation documented by
    :func:`sample_to_numpy`. Energies are converted to Boltzmann log weights at
    the explicitly supplied temperature.

    Parameters
    ----------
    suboptimal : sequence of (str, float)
        Structure and energy pairs, in kcal/mol, as returned by
        ``Molecule.suboptimal_structures()``.

    temperature : float
        Temperature in kelvin.

    Returns
    -------
    states : ndarray of int16, shape (n_structures, sequence_length)
        Zero-based pair tables, with ``-1`` denoting an unpaired nucleotide.

    logweights : ndarray of float, shape (n_structures,)
        Normalized Boltzmann log weights, satisfying
        ``sum(exp(logweights)) == 1`` up to floating-point precision.
    """
    suboptimal = list(suboptimal)
    if not suboptimal:
        raise ValueError("suboptimal cannot be empty")
    if not all(
        not isinstance(item, str)
        and hasattr(item, "__len__")
        and len(item) == 2
        and isinstance(item[0], str)
        for item in suboptimal
    ):
        raise ValueError(
            "suboptimal must contain (structure, energy) pairs"
        )

    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")

    structures = [item[0] for item in suboptimal]
    try:
        energies = np.asarray([item[1] for item in suboptimal], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("energies must be real numbers") from error
    if energies.ndim != 1:
        raise ValueError("energies must be scalar")
    if not np.all(np.isfinite(energies)):
        raise ValueError("energies must be finite")

    states = _structures_to_pairtables(structures)
    logweights = -energies / (_KB * temperature)
    if not np.all(np.isfinite(logweights)):
        raise ValueError("energies are too large to convert to log weights")
    return states, _normalize_logweights(logweights)

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

# Last parameter set loaded here and the marker subsequently reported by
# ViennaRNA. The latter lets us notice parameter changes made directly through
# RNA without depending on ViennaRNA's particular names for built-in sets.
_LAST_THERMODYNAMIC_PARAMETERS = None
_LAST_THERMODYNAMIC_PARAMETER_FILE = None

_INITIAL_DEFAULT_PARAMETERS = {
    "temperature": 37 + _CELSIUS_TO_KELVIN,
    "no_lonely_pair": False,
    "NaCl": None,
    "parameters": "turner2004",
}

# Module defaults used by subsequently constructed molecules. Molecules copy
# these values at construction time and are unaffected by later changes.
_default_parameters = _INITIAL_DEFAULT_PARAMETERS.copy()
_DEFAULT_PARAMETERS_LOCK = threading.Lock()

def set_default_parameters(
    *,
    temperature=None,
    no_lonely_pair=None,
    NaCl=None,
    parameters=None,
):
    """
    Update the default thermodynamic parameters for new molecules.

    Parameters set to ``None`` are left unchanged. Existing molecules are not
    affected. Use :func:`reset_default_parameters` to restore all original
    defaults, including ViennaRNA's default salt concentration.
    """
    updates = {}

    if temperature is not None:
        if not temperature >= 0.0:
            raise ValueError(
                f"Temperature {temperature} should be positive"
            )
        updates["temperature"] = temperature

    if no_lonely_pair is not None:
        if not isinstance(no_lonely_pair, (bool, np.bool_)):
            raise ValueError("no_lonely_pair must be a boolean")
        updates["no_lonely_pair"] = bool(no_lonely_pair)

    if NaCl is not None:
        if not NaCl >= 0.0:
            raise ValueError(
                f"Salt concentration {NaCl} should be positive"
            )
        updates["NaCl"] = NaCl

    if parameters is not None:
        parameters = str(parameters).lower()
        if parameters not in _THERMODYNAMIC_PARAMETERS:
            raise ValueError(
                f"Thermodynamic parameters {parameters} not known"
            )
        updates["parameters"] = parameters

    with _DEFAULT_PARAMETERS_LOCK:
        _default_parameters.update(updates)

def reset_default_parameters():
    """Restore the original thermodynamic defaults for new molecules."""
    with _DEFAULT_PARAMETERS_LOCK:
        _default_parameters.clear()
        _default_parameters.update(_INITIAL_DEFAULT_PARAMETERS)

def _resolve_default_parameters(
    temperature,
    no_lonely_pair,
    NaCl,
    parameters,
):
    """Resolve ``None`` values against one consistent defaults snapshot."""
    with _DEFAULT_PARAMETERS_LOCK:
        defaults = _default_parameters.copy()
    return (
        defaults["temperature"] if temperature is None else temperature,
        defaults["no_lonely_pair"]
        if no_lonely_pair is None else no_lonely_pair,
        defaults["NaCl"] if NaCl is None else NaCl,
        defaults["parameters"] if parameters is None else parameters,
    )

def _workaround_vienna_272_params_cache():
    """
    Invalidate ViennaRNA 2.7.2's stale energy-parameter cache.

    ViennaRNA 2.7.2 may reuse the parameter object created after the first
    ``params_load`` call even after another parameter set is loaded. Creating
    partition-function parameters with a slightly different model temperature
    invalidates that cache. Call this while holding
    ``_THERMODYNAMIC_PARAMETERS_LOCK``, immediately after loading parameters.

    See https://github.com/ViennaRNA/ViennaRNA/issues/284.
    """
    if RNA is None or str(getattr(RNA, "__version__", "")) != "2.7.2":
        return
    md = RNA.md()
    md.temperature += 0.001
    RNA.fold_compound("AAAA", md).pf()

def _ensure_thermodynamic_parameters(parameters):
    """Load a thermodynamic parameter set only when it is not active."""
    global _LAST_THERMODYNAMIC_PARAMETERS
    global _LAST_THERMODYNAMIC_PARAMETER_FILE

    current_parameter_file = (
        RNA.last_parameter_file()
        if hasattr(RNA, "last_parameter_file")
        else None
    )
    if (
        parameters == _LAST_THERMODYNAMIC_PARAMETERS
        and current_parameter_file == _LAST_THERMODYNAMIC_PARAMETER_FILE
    ):
        return

    if not _THERMODYNAMIC_PARAMETERS[parameters]():
        raise RuntimeError(
            f"Could not load thermodynamic parameters {parameters!r}"
        )
    _workaround_vienna_272_params_cache()
    _LAST_THERMODYNAMIC_PARAMETERS = parameters
    _LAST_THERMODYNAMIC_PARAMETER_FILE = (
        RNA.last_parameter_file()
        if hasattr(RNA, "last_parameter_file")
        else None
    )

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
        _workaround_vienna_272_params_cache()
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

    # do not add callback if all residuals are negligible
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

def _apply_constraint(fc, sequence, lambdas, kT):
    """
    Apply per-nucleotide pairing penalties using a hybrid 1D/2D scheme.
    Returns the structure-independent energy shift that must be added to reported
    energies and free energies.
    """

    # This is by far the common case, and avoids the pairwise O(n**2) scan
    # below for an unconstrained molecule.
    if not np.any(lambdas):
        return 0.0, 0, 0

    n = len(lambdas)

    # The dangerous factor is approximately
    #
    #     exp(sum(lambda_positive_1d) / kT).
    #
    # Stay well below the largest representable double.
    budget = _PARTITION_OVERFLOW_SAFETY_FRACTION * kT * math.log(sys.float_info.max)

    # Vanilla ViennaRNA has a bug that makes subopt with negative sc_add_up problematic.
    # Avoid negative sc_add_up values by excluding positive lambdas from the 1D scheme.
    # Technically, this is only needed for using subopt. However, to keep the code simpler
    # we apply the rule in general for build fold compounds.

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

    if not np.any(use_2d):
        return shift, n_1d, 0

    # 2D representation. Add each pair constraint only once, combining
    # contributions from both endpoints.
    lambda_2d = np.where(use_2d, lambdas, 0.0)

    n_2d = 0
    for i in range(n):
        for j in range(i + 1, n):
            if sequence[i] + sequence[j] not in _ALLOWED_PAIRS:
                continue

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
    # NumPy has a fixed setup cost but avoids a Python loop for longer RNAs,
    # where evaluation otherwise becomes dominated by the correction rather
    # than ViennaRNA's evaluator.
    if len(structure) >= 32:
        paired = (
            np.frombuffer(structure.encode("ascii"), dtype=np.uint8)
            != ord(".")
        )
        return float(np.dot(dlambdas, paired))

    correction = 0.0
    for symbol, residual in zip(structure, dlambdas):
        if symbol != ".":
            correction += residual
    return float(correction)

def _pf_with_mfe_rescaling_fallback(fc):
    """Compute a PF, retrying with MFE-based scaling after numeric failure."""
    free_energy = fc.pf()[1]
    if not math.isfinite(free_energy) or free_energy >= RNA.INF / 100.0:
        mfe = fc.mfe()[1]
        fc.exp_params_rescale(mfe)
        free_energy = fc.pf()[1]
    return free_energy

class _DPMolecule:
    """
    Internal dynamic-programming implementation of a single RNA ensemble.
    """

    def _make_md_params(self, *, compute_bpp=True):
        """
        Internal utility to generate an md params object.
        """
        md = RNA.md()
        md.uniq_ML = 1
        md.compute_bpp = int(compute_bpp)
        md.noLP = int(self._no_lonely_pair)
        md.temperature = self._temperature - _CELSIUS_TO_KELVIN
        if self._salt is not None:
            md.salt = self._salt
        return md

    def _make_fold_compound(self, *, compute_bpp=True):
        """
        Internal utility to create a fold compound.
        """
        with _THERMODYNAMIC_PARAMETERS_LOCK:
            _ensure_thermodynamic_parameters(self._parameters)
            return RNA.fold_compound(
                self._seq,
                self._make_md_params(compute_bpp=compute_bpp),
            )

    def _ensure_fc_rounded(self):
        """
        Internal utility to ensure that the fold compound using rounded lambdas
        has been initialized.
        """
        if self._fc_rounded is None:
            self._fc_rounded = self._make_fold_compound()
            (self._fc_rounded_shift,
             self._fc_rounded_n_1d_constraints,
             self._fc_rounded_n_2d_constraints) = _apply_constraint(self._fc_rounded, self._seq, self._lambdas1d_rounded, _KB * self._temperature)
            _apply_hard_constraint(self._fc_rounded, paired=self._force_paired, unpaired=self._force_unpaired)

    def _ensure_fc(self, *, compute_bpp=True):
        """
        Internal utility to ensure that the fold compound using continuous lambdas
        has been initialized.
        """
        if self._fc is None:
            self._fc = self._make_fold_compound(
                compute_bpp=compute_bpp
            )
            self._fc_compute_bpp = compute_bpp

            if _SUPPORTS_NATIVE_CONTINUOUS:
                use_lambdas = self._lambdas1d
            else:
                # when using standard vienna builds without support for continuous lambdas
                # this fold compound is constructed using rounded lambdas
                # and residuals are added with a (slow) callback function
                use_lambdas = self._lambdas1d_rounded

            (self._fc_shift,
            self._fc_n_1d_constraints,
            self._fc_n_2d_constraints) = _apply_constraint(self._fc, self._seq, use_lambdas, _KB * self._temperature)

            if _SUPPORTS_NATIVE_CONTINUOUS:
                self._pf_callback = None
            else:
                # note that the callback only applies the residuals
                self._pf_callback = _apply_residual_callback(self._fc, self._lambdas1d_residuals, _KB * self._temperature)

            _apply_hard_constraint(self._fc, paired=self._force_paired, unpaired=self._force_unpaired)

    def _ensure_pf(self, *, compute_bpp=True):
        """
        Ensure that the partition-function energy and, optionally, BPPs exist.
        """
        if (
            self._total_free_energy is None
            or compute_bpp and self._base_pairing_probability is None
        ):
            # A fold compound built for a scalar free-energy request omits
            # ViennaRNA's probability backtracking. Rebuild it if probabilities
            # are requested later.
            if (
                compute_bpp
                and self._fc is not None
                and not self._fc_compute_bpp
            ):
                self._fc = None
                self._pf_callback = None

            self._ensure_fc(compute_bpp=compute_bpp)

            # ViennaRNA automatically estimates a PF scaling factor. Usually
            # this is sufficient and avoids a separate MFE calculation. If it
            # reports numerical failure, retry with the more robust MFE-based
            # scaling factor.
            self._total_free_energy = _pf_with_mfe_rescaling_fallback(
                self._fc
            )
            # correction for using bp instead of up
            self._total_free_energy += self._fc_shift

            if compute_bpp:
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
            self._rounded_total_free_energy = (
                _pf_with_mfe_rescaling_fallback(self._fc_rounded)
            )
            self._rounded_total_free_energy += self._fc_rounded_shift
            self._fc_rounded_pf=True

    def __init__(
        self,
        seq: str,
        *,
        lambdas1d,
        temperature,
        force_paired,
        force_unpaired,
        no_lonely_pair,
        NaCl,
        parameters,
    ):

        _require_viennarna()

        self._parameters = str(parameters).lower()
        if not self._parameters in _THERMODYNAMIC_PARAMETERS:
            raise ValueError(f"Thermodynamic parameters {parameters} not known")

        self._seq = str(seq).upper()

        if not self._seq:
            raise ValueError("seq cannot be empty")

        if any(base not in "ACGU" for base in self._seq):
            raise ValueError("seq must contain only A, C, G, and U")

        if not isinstance(no_lonely_pair, (bool, np.bool_)):
            raise ValueError("no_lonely_pair must be a boolean")
        self._no_lonely_pair = bool(no_lonely_pair)

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
        self._fc_compute_bpp = None
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
            len(structure) == len(self._seq)
            and
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
                if self._lambdas1d_residuals_range != 0.0:
                    self._mfe_energy += _correct_rounding_energy(
                        self._mfe_structure,
                        self._lambdas1d_residuals,
                    )
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

    def evaluate(self, structure):
        """Return the exact free energy of a secondary structure."""
        self._ensure_fc_rounded()
        energy = self._fc_rounded.eval_structure(structure)
        if self._lambdas1d_residuals_range != 0.0:
            energy += _correct_rounding_energy(
                structure,
                self._lambdas1d_residuals,
            )
        return float(energy + self._fc_rounded_shift)

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
        self._ensure_pf(compute_bpp=False)
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

        if self._lambdas1d_residuals_range == 0.0:
            energies = [s.energy for s in subopt]
        else:
            energies = [
                _correct_rounding_energy(
                    s.structure,
                    self._lambdas1d_residuals,
                ) + s.energy
                for s in subopt
            ]
        index = np.argsort(energies)
        return [
            (subopt[i].structure , float(energies[i] + self._fc_rounded_shift))
            for i in index
            if energies[i]-energies[index[0]] <= delta
        ]

    def _sample_weighted(self, number):
        """
        Sample from the rounded model and return exact-model log weights.
        """
        self._ensure_rounded_pf()

        structures = self._fc_rounded.pbacktrack(number)
        if self._lambdas1d_residuals_range == 0.0:
            return [(structure, 0.0) for structure in structures]

        inverse_kT = 1.0 / (_KB * self._temperature)

        return [
            (
                structure,
                -float(
                    _correct_rounding_energy(
                        structure,
                        self._lambdas1d_residuals,
                    ) * inverse_kT
                ),
            )
            for structure in structures
        ]

    def sample(self, number, weights=False):
        """
        Generate Boltzmann-distributed secondary structures.

        By default, rejection sampling corrects samples from ViennaRNA's rounded
        model to the exact continuous-lambda distribution. Alternatively, the
        rounded samples and their importance weights can be returned directly.

        Parameters
        ----------
        number : int
            Number of structures to sample.

        weights : bool, default=False
            If False, return unweighted structures sampled from the exact model.
            If True, return samples from the rounded model together with
            unnormalized log-weight corrections.

        Returns
        -------
        list of str or list of (str, float)
            With ``weights=False``, a list of dot-bracket structures. With
            ``weights=True``, each element contains a structure and its
            unnormalized log-weight correction

                log(w) = -(E_exact - E_rounded) / (k_B T).

            When all lambdas are multiples of 0.01 kcal/mol, every returned
            log-weight is zero.
        """

        number = int(number)
        if number <= 0:
            raise ValueError("number must be a positive integer")
        if not isinstance(weights, (bool, np.bool_)):
            raise ValueError("weights must be a boolean")

        if weights:
            return self._sample_weighted(number)

        inverse_kT = 1.0 / (_KB * self._temperature)
        max_log_weight = -float(np.sum(
            np.minimum(self._lambdas1d_residuals, 0.0)
        )) * inverse_kT

        # Initialize the assumed candidates-per-accepted-sample ratio from the
        # worst-case log-weight range.
        log_weight_span = (
            self._lambdas1d_residuals_range * inverse_kT
        )
        if log_weight_span < math.log(_MAX_ASSUMED_INFLATION):
            assumed_inflation = math.exp(log_weight_span)
        else:
            assumed_inflation = _MAX_ASSUMED_INFLATION

        accepted = []
        while len(accepted) < number:
            remaining = number - len(accepted)
            batch_size = int(remaining * assumed_inflation)
            candidates = self._sample_weighted(batch_size)
            accepted_in_batch = 0

            for structure, log_weight in candidates:
                acceptance_probability = math.exp(
                    log_weight - max_log_weight
                )
                if np.random.random() < acceptance_probability:
                    accepted.append(structure)
                    accepted_in_batch += 1
                    if len(accepted) == number:
                        break

            if len(accepted) < number:
                # Replace the initial worst-case estimate, or the estimate from
                # the previous batch, with the observed rejection rate.
                if accepted_in_batch:
                    assumed_inflation = min(
                        _MAX_ASSUMED_INFLATION,
                        batch_size / accepted_in_batch
                    )
                else:
                    assumed_inflation = _MAX_ASSUMED_INFLATION

        return accepted

    def sample_rounding_correction(self):
        """
        Return the component normalization correction for sampled log weights.

        This dimensionless correction accounts for the difference between the
        exact and rounded partition functions. It is structure-independent within
        one ensemble, but generally differs between components and is therefore
        required when their samples are combined.
        """
        self._ensure_pf(compute_bpp=False)
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

    state_positions : array-like of int or sequence of array-like, optional
        Zero-based indices defining binary paired/unpaired states. A flat
        sequence defines one state set. A sequence of sequences defines
        multiple disjoint state sets whose energy biases are additive. If
        `state_biases` is omitted, every state is assigned zero bias.

    state_biases : array-like or sequence of array-like, optional
        Energy biases (kcal/mol) for the states defined by `state_positions`.
        For one state set, its shape must be `(2,) * len(state_positions)`.
        For multiple sets, provide one tensor per set, with shape
        `(2,) * len(positions)`. Index zero denotes an unpaired nucleotide and
        index one a paired nucleotide. It cannot be provided without
        `state_positions`.

    reduce_state_space : bool, default=True
        If True, represent the final selected state position as an equivalent
        1D pairing penalty, reducing the number of dynamic-programming ensembles
        from 2**N to 2**(N-1). If False, use one hard-conditioned ensemble for
        every state.

    no_lonely_pair : bool or None, default=None
        If True, exclude structures containing isolated base pairs using
        ViennaRNA's `noLP` model option. If None, use the current module
        default.

    temperature : float or None, default=None
        Temperature in kelvin. If None, use the current module default, which
        is initially 310.15 K.

    NaCl : float or None, default=None
        Sodium concentration (M). If None, use the current module default,
        which initially selects ViennaRNA's default value.

    parameters : {"turner1999", "turner2004", "andronescu2007", "langdon2018"} or None
        Thermodynamic parameter set. If None, use the current module default.

    Notes
    -----
    By default, a state set containing N positions uses 2**(N-1)
    dynamic-programming ensembles. For multiple sets of sizes N_k, the number
    is 2**(sum(N_k)-K). The final position of each set is represented within
    each ensemble by an equivalent constant energy shift and 1D pairing
    penalty. Set `reduce_state_space=False` to explicitly condition every
    selected position.

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
        temperature=None,
        force_paired = None,
        force_unpaired = None,
        state_positions=None,
        state_biases=None,
        reduce_state_space=True,
        no_lonely_pair=None,
        NaCl=None,
        parameters=None,
    ):
        (
            temperature,
            no_lonely_pair,
            NaCl,
            parameters,
        ) = _resolve_default_parameters(
            temperature,
            no_lonely_pair,
            NaCl,
            parameters,
        )

        self._has_state_biases = state_positions is not None

        if not isinstance(reduce_state_space, (bool, np.bool_)):
            raise ValueError("reduce_state_space must be a boolean")
        self._reduce_state_space = bool(reduce_state_space)

        if not isinstance(no_lonely_pair, (bool, np.bool_)):
            raise ValueError("no_lonely_pair must be a boolean")
        self._no_lonely_pair = bool(no_lonely_pair)

        if state_positions is None and state_biases is not None:
            raise ValueError(
                "state_biases cannot be provided without state_positions"
            )

        base_force_paired = (
            [] if force_paired is None else list(force_paired)
        )
        base_force_unpaired = (
            [] if force_unpaired is None else list(force_unpaired)
        )

        if state_positions is None:
            self._state_sets_were_nested = False
            self._state_position_sets = ()
            self._state_bias_sets = ()
        else:
            try:
                raw_positions = list(state_positions)
            except TypeError as error:
                raise ValueError(
                    "state_positions must be a sequence"
                ) from error

            # A flat sequence retains the original single-state-set syntax.
            # Nesting is detected from state_positions rather than state_biases,
            # since a two-element bias vector is inherently ambiguous.
            if all(np.isscalar(value) for value in raw_positions):
                self._state_sets_were_nested = False
                raw_position_sets = [raw_positions]
            else:
                self._state_sets_were_nested = True
                raw_position_sets = raw_positions

            position_sets = []
            for positions in raw_position_sets:
                positions = np.asarray(positions)
                if positions.ndim != 1:
                    raise ValueError(
                        "each state_positions set must be one-dimensional"
                    )
                if (
                    positions.size
                    and not np.issubdtype(positions.dtype, np.integer)
                ):
                    raise ValueError(
                        "state_positions must contain integers"
                    )
                position_set = tuple(int(i) for i in positions)
                if len(set(position_set)) != len(position_set):
                    raise ValueError(
                        "state_positions sets must not contain duplicates"
                    )
                if any(
                    i < 0 or i >= len(str(seq))
                    for i in position_set
                ):
                    raise ValueError(
                        "state_positions must contain valid nucleotide indices"
                    )
                position_sets.append(position_set)

            all_state_positions = [
                position
                for position_set in position_sets
                for position in position_set
            ]
            if len(set(all_state_positions)) != len(all_state_positions):
                raise ValueError("state_positions sets must be disjoint")

            fixed_positions = (
                set(base_force_paired) | set(base_force_unpaired)
            )
            if fixed_positions.intersection(all_state_positions):
                raise ValueError(
                    "state_positions must not overlap force_paired or "
                    "force_unpaired"
                )

            expected_shapes = [
                (2,) * len(position_set)
                for position_set in position_sets
            ]
            if state_biases is None:
                bias_sets = [
                    np.zeros(shape, dtype=float)
                    for shape in expected_shapes
                ]
            elif len(position_sets) == 1:
                expected_shape = expected_shapes[0]
                try:
                    candidate = np.asarray(state_biases, dtype=float)
                except (TypeError, ValueError):
                    candidate = np.asarray([], dtype=float)
                if candidate.shape == expected_shape:
                    bias_sets = [candidate.copy()]
                else:
                    try:
                        if len(state_biases) != 1:
                            raise ValueError
                        candidate = np.asarray(
                            state_biases[0],
                            dtype=float,
                        )
                    except (TypeError, ValueError, IndexError) as error:
                        raise ValueError(
                            f"state_biases must have shape {expected_shape}"
                        ) from error
                    if candidate.shape != expected_shape:
                        raise ValueError(
                            f"state_biases must have shape {expected_shape}"
                        )
                    bias_sets = [candidate.copy()]
            else:
                try:
                    if len(state_biases) != len(position_sets):
                        raise ValueError(
                            "state_biases must contain one tensor per "
                            "state_positions set"
                        )
                except TypeError as error:
                    raise ValueError(
                        "state_biases must contain one tensor per "
                        "state_positions set"
                    ) from error
                bias_sets = []
                for biases, expected_shape in zip(
                    state_biases,
                    expected_shapes,
                ):
                    biases = np.asarray(biases, dtype=float)
                    if biases.shape != expected_shape:
                        raise ValueError(
                            "each state_biases tensor must have shape "
                            f"{expected_shape}"
                        )
                    bias_sets.append(biases.copy())

            if any(
                not np.all(np.isfinite(biases))
                for biases in bias_sets
            ):
                raise ValueError(
                    "state_biases must contain only finite values"
                )

            self._state_position_sets = tuple(position_sets)
            self._state_bias_sets = tuple(bias_sets)

        # Preserve the input representation for compatibility. New code uses
        # the normalized plural attributes above.
        if not self._state_position_sets:
            self._state_positions = ()
            self._state_biases = np.zeros((), dtype=float)
        elif self._state_sets_were_nested:
            self._state_positions = self._state_position_sets
            self._state_biases = self._state_bias_sets
        else:
            self._state_positions = self._state_position_sets[0]
            self._state_biases = self._state_bias_sets[0]

        if lambdas1d is None:
            base_lambdas1d = np.zeros(len(str(seq)))
        else:
            base_lambdas1d = np.asarray(
                lambdas1d,
                dtype=float,
            ).copy()
        if base_lambdas1d.ndim != 1:
            raise ValueError("lambdas1d must be one-dimensional")
        if len(base_lambdas1d) != len(str(seq)):
            raise ValueError(
                "lambdas1d must contain one value per nucleotide"
            )
        if not np.all(np.isfinite(base_lambdas1d)):
            raise ValueError(
                "lambdas1d must contain only finite values"
            )

        # The final variable of every state set does not require explicit
        # branching. For each assignment of the preceding variables, its two
        # biases are equivalent to a constant plus a 1D pairing penalty.
        explicit_position_sets = []
        implicit_positions = []
        state_slices = []
        offset = 0
        for position_set in self._state_position_sets:
            if position_set and self._reduce_state_space:
                explicit_positions = position_set[:-1]
                implicit_position = position_set[-1]
            else:
                explicit_positions = position_set
                implicit_position = None
            explicit_position_sets.append(explicit_positions)
            implicit_positions.append(implicit_position)
            state_slices.append(slice(
                offset,
                offset + len(explicit_positions),
            ))
            offset += len(explicit_positions)

        self._explicit_state_position_sets = tuple(explicit_position_sets)
        self._implicit_state_positions = tuple(implicit_positions)
        self._state_slices = tuple(state_slices)
        explicit_state_positions = tuple(
            position
            for position_set in explicit_position_sets
            for position in position_set
        )
        self._explicit_state_positions = explicit_state_positions
        component_shape = (2,) * len(explicit_state_positions)

        # Preserve the legacy singular attribute for one state set.
        self._implicit_state_position = (
            implicit_positions[0]
            if len(implicit_positions) == 1
            else None
        )

        self._states = list(np.ndindex(component_shape))
        self._component_biases = np.empty(component_shape, dtype=float)
        self._dp_molecules = []

        for state in self._states:
            state_paired = [
                position
                for position, value in zip(
                    explicit_state_positions,
                    state,
                )
                if value
            ]
            state_unpaired = [
                position
                for position, value in zip(
                    explicit_state_positions,
                    state,
                )
                if not value
            ]

            component_lambdas1d = base_lambdas1d.copy()
            component_bias = 0.0
            for biases, state_slice, implicit_position in zip(
                self._state_bias_sets,
                self._state_slices,
                self._implicit_state_positions,
            ):
                set_state = state[state_slice]
                if implicit_position is None:
                    component_bias += float(biases[set_state])
                else:
                    unpaired_bias = float(biases[set_state + (0,)])
                    paired_bias = float(biases[set_state + (1,)])
                    component_bias += unpaired_bias
                    component_lambdas1d[implicit_position] += (
                        paired_bias - unpaired_bias
                    )

            self._component_biases[state] = component_bias
            self._dp_molecules.append(
                _DPMolecule(
                    seq,
                    lambdas1d=component_lambdas1d,
                    temperature=temperature,
                    NaCl=NaCl,
                    force_paired=base_force_paired + state_paired,
                    force_unpaired=base_force_unpaired + state_unpaired,
                    no_lonely_pair=self._no_lonely_pair,
                    parameters=parameters,
                )
            )

        first_molecule = self._dp_molecules[0]
        self._seq = first_molecule._seq
        self._lambdas1d = base_lambdas1d
        self._temperature = first_molecule._temperature
        self._salt = first_molecule._salt
        self._parameters = first_molecule._parameters
        self._no_lonely_pair = first_molecule._no_lonely_pair
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
        state_position_sets = list(self._state_position_sets)
        state_bias_sets = list(self._state_bias_sets)

        for set_index, position_set in enumerate(state_position_sets):
            if position not in position_set:
                continue
            axis = position_set.index(position)
            state_position_sets[set_index] = (
                position_set[:axis] + position_set[axis + 1:]
            )
            state_bias_sets[set_index] = np.take(
                state_bias_sets[set_index],
                0,
                axis=axis,
            )
            break

        if not state_position_sets:
            state_positions = None
            state_biases = None
        elif not self._state_sets_were_nested:
            state_positions = state_position_sets[0]
            state_biases = state_bias_sets[0]
        else:
            state_positions = state_position_sets
            state_biases = state_bias_sets

        return Molecule(
            self._seq,
            lambdas1d=self._lambdas1d,
            temperature=self._temperature,
            force_paired=self._force_paired,
            force_unpaired=force_unpaired,
            state_positions=state_positions,
            state_biases=state_biases,
            reduce_state_space=self._reduce_state_space,
            no_lonely_pair=self._no_lonely_pair,
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
            free_energies + self._component_biases.ravel()
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
            self._component_biases.ravel(),
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
        # A single DP component needs no mixture-level search.
        if len(self._dp_molecules) == 1:
            structure, energy = self._dp_molecules[0].mfe()
            return (
                structure,
                float(energy + self._component_biases.item()),
            )

        structures, energies = self._component_mfes()
        index = int(np.argmin(energies))
        return structures[index], float(energies[index])

    def evaluate(self, structure):
        """
        Return the exact free energy of a secondary structure.

        Parameters
        ----------
        structure : str
            Dot-bracket representation of a secondary structure.

        Returns
        -------
        float
            Exact free energy (kcal/mol), including continuous pairing
            penalties and state biases.
        """
        if not isinstance(structure, str):
            raise ValueError("structure must be a string")
        if len(structure) != len(self._seq):
            raise ValueError(
                "structure must contain one symbol per nucleotide"
            )
        if structure.strip(".()"):
            raise ValueError(
                "structure must use '.', '(', and ')' dot-bracket symbols"
            )

        # Components follow np.ndindex order, so the paired/unpaired state is
        # also the component index interpreted as a binary integer. Avoid
        # constructing a state tuple on this latency-sensitive path.
        component_index = 0
        for position in self._explicit_state_positions:
            component_index = (
                2 * component_index
                + (structure[position] != ".")
            )

        energy = self._dp_molecules[component_index].evaluate(structure)
        return float(
            energy + self._component_biases.flat[component_index]
        )

    def base_pairing_probability(self):
        """
        Return the base-pairing probability matrix.

        Returns
        -------
        ndarray
            Symmetric NxN matrix whose element (i,j) is the equilibrium
            probability that nucleotides i and j form a base pair.
        """
        # Compute BPPs before mixture weights: a BPP calculation also produces
        # the component free energy, while doing this in the opposite order
        # would require rebuilding each PF without probability backtracking.
        for molecule in self._dp_molecules:
            molecule._ensure_pf(compute_bpp=True)

        if len(self._dp_molecules) == 1:
            return self._dp_molecules[0]._base_pairing_probability.copy()

        _, probabilities = self._component_probabilities()
        matrix = np.zeros_like(
            self._dp_molecules[0]._base_pairing_probability
        )
        for probability, molecule in zip(
            probabilities,
            self._dp_molecules,
        ):
            if probability == 0.0:
                continue
            matrix += (
                probability * molecule._base_pairing_probability
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

    def d_free_energy_d_lambdas1d(self):
        """
        Return the derivatives of the free energy with respect to `lambdas1d`.

        The derivative for nucleotide ``i`` is its equilibrium pairing
        probability,

        ``dF / d lambda_i = <s_i>``.

        Returns
        -------
        ndarray
            One-dimensional array containing one derivative per nucleotide.
        """
        return np.sum(self.base_pairing_probability(), axis=1)

    def d_free_energy_d_state_biases(self):
        """
        Return the derivatives of the free energy with respect to state biases.

        The derivative with respect to a state's energy bias is the equilibrium
        probability of that state.

        Returns
        -------
        ndarray or list of ndarray
            With the flat, single-set `state_positions` syntax, an array with
            the same shape as `state_biases`. With the list-of-sets syntax, one
            array per set, including when the list contains only one set.

        Raises
        ------
        ValueError
            If this molecule was not initialized with state biases.
        """
        if not self._has_state_biases:
            raise ValueError(
                "This molecule was not initialized with state biases"
            )

        needs_pairing_probabilities = any(
            position is not None
            for position in self._implicit_state_positions
        )
        if needs_pairing_probabilities:
            component_pairing_probabilities = [
                np.sum(molecule.base_pairing_probability(), axis=1)
                for molecule in self._dp_molecules
            ]
        else:
            component_pairing_probabilities = [None] * len(
                self._dp_molecules
            )

        _, component_probabilities = self._component_probabilities()
        probability_sets = [
            np.zeros_like(biases)
            for biases in self._state_bias_sets
        ]

        for (
            state,
            component_probability,
            pairing_probabilities,
        ) in zip(
            self._states,
            component_probabilities,
            component_pairing_probabilities,
        ):
            if component_probability == 0.0:
                continue

            for probabilities, state_slice, implicit_position in zip(
                probability_sets,
                self._state_slices,
                self._implicit_state_positions,
            ):
                set_state = state[state_slice]
                if implicit_position is None:
                    probabilities[set_state] += component_probability
                    continue

                paired_probability = float(np.clip(
                    pairing_probabilities[implicit_position],
                    0.0,
                    1.0,
                ))
                probabilities[set_state + (0,)] += (
                    component_probability * (1.0 - paired_probability)
                )
                probabilities[set_state + (1,)] += (
                    component_probability * paired_probability
                )

        if not self._state_sets_were_nested:
            return probability_sets[0]
        return probability_sets

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
        # A single DP component needs no mixture-level enumeration.
        if len(self._dp_molecules) == 1:
            suboptimal = self._dp_molecules[0].suboptimal_structures(
                delta
            )
            bias = float(self._component_biases.item())
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
            self._component_biases.ravel(),
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

    def sample(self, number, weights=False):
        """
        Generate Boltzmann-distributed secondary structures.

        By default, rejection sampling corrects ViennaRNA's rounded soft
        constraints and returns unweighted structures from the exact continuous
        model. Weighted samples from the rounded model can be requested instead.

        Parameters
        ----------
        number : int
            Number of structures to sample.

        weights : bool, default=False
            If False, return unweighted structures sampled from the exact model.
            If True, return rounded-model samples and their log-weight
            corrections. The weighted path is faster when rounding corrections
            are present.

        Returns
        -------
        list of str or list of (str, float)
            With ``weights=False``, a list of dot-bracket structures. With
            ``weights=True``, each element contains a structure and its
            unnormalized log-weight correction

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
        if not isinstance(weights, (bool, np.bool_)):
            raise ValueError("weights must be a boolean")

        # A single DP component needs no mixture-level sampling.
        if len(self._dp_molecules) == 1:
            return self._dp_molecules[0].sample(
                number,
                weights=weights,
            )

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

            samples = molecule.sample(
                len(output_indices),
                weights=weights,
            )
            if weights:
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
            else:
                for output_index, structure in zip(
                    output_indices,
                    samples,
                ):
                    result[output_index] = structure

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
