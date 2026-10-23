import unittest
import numpy as np
import subprocess
import sys

try:
    import RNA
    _HAS_VIENNA = True
except ImportError:
    _HAS_VIENNA = False

if _HAS_VIENNA:
    import bussilab.rna2d as rna2d
    from bussilab.rna2d import Molecule, _KB


def _enumerate_secondary_structures(sequence):
    """Enumerate all pseudoknot-free structures for a short RNA sequence."""
    if not sequence:
        yield ""
        return

    yield from ("." + structure
                for structure in _enumerate_secondary_structures(sequence[1:]))

    canonical_pairs = {"AU", "UA", "CG", "GC", "GU", "UG"}
    for j in range(4, len(sequence)):
        if sequence[0] + sequence[j] not in canonical_pairs:
            continue
        for left in _enumerate_secondary_structures(sequence[1:j]):
            for right in _enumerate_secondary_structures(sequence[j + 1:]):
                yield "(" + left + ")" + right

@unittest.skipUnless(_HAS_VIENNA, "ViennaRNA not available")
class TestRNA2D(unittest.TestCase):

    def _run_in_vanilla_mode(self, test):
        native = rna2d._SUPPORTS_NATIVE_CONTINUOUS
        subopt = rna2d._SUPPORTS_SUBOPT_SOFT_CONSTRAINTS
        rna2d._SUPPORTS_NATIVE_CONTINUOUS = False
        rna2d._SUPPORTS_SUBOPT_SOFT_CONSTRAINTS = False
        try:
            test()
        finally:
            rna2d._SUPPORTS_NATIVE_CONTINUOUS = native
            rna2d._SUPPORTS_SUBOPT_SOFT_CONSTRAINTS = subopt

    def setUp(self):
        self.seq = "GGGAAACCC"

    def test_sample_to_numpy(self):

        structures = [
            "((..))",
            "......",
            "((..))",
            "(....)",
        ]
        states, logweights = rna2d.sample_to_numpy(structures)

        np.testing.assert_array_equal(states, np.array([
            [5, 4, -1, -1, 1, 0],
            [-1, -1, -1, -1, -1, -1],
            [5, -1, -1, -1, -1, 0],
        ], dtype=np.int16))
        self.assertEqual(states.dtype, np.dtype(np.int16))
        np.testing.assert_allclose(
            np.exp(logweights),
            [0.5, 0.25, 0.25],
        )
        self.assertAlmostEqual(np.sum(np.exp(logweights)), 1.0)

        vienna_table = np.concatenate((
            [states.shape[1]],
            states[0] + 1,
        ))
        np.testing.assert_array_equal(
            vienna_table,
            np.asarray(RNA.ptable(structures[0])),
        )

        repeated_states, repeated_logweights = rna2d.sample_to_numpy(
            structures,
            deduplicate=False,
        )
        self.assertEqual(repeated_states.shape, (4, 6))
        np.testing.assert_array_equal(
            repeated_states[0],
            repeated_states[2],
        )
        np.testing.assert_allclose(
            np.exp(repeated_logweights),
            np.full(4, 0.25),
        )

        weighted = [
            ("((..))", np.log(2.0)),
            ("......", np.log(3.0)),
            ("((..))", np.log(4.0)),
        ]
        weighted_states, weighted_logweights = rna2d.sample_to_numpy(
            weighted
        )
        self.assertEqual(weighted_states.shape, (2, 6))
        np.testing.assert_allclose(
            np.exp(weighted_logweights),
            [2.0 / 3.0, 1.0 / 3.0],
        )

        # Sort after merging duplicates; ties retain first-occurrence order.
        sorted_states, sorted_logweights = rna2d.sample_to_numpy([
            "......", "(....)", "((..))", "(....)", "((..))",
        ])
        np.testing.assert_array_equal(sorted_states, np.array([
            [5, -1, -1, -1, -1, 0],
            [5, 4, -1, -1, 1, 0],
            [-1, -1, -1, -1, -1, -1],
        ], dtype=np.int16))
        np.testing.assert_allclose(np.exp(sorted_logweights), [0.4, 0.4, 0.2])

        original_states, original_logweights = rna2d.sample_to_numpy([
            ("......", np.log(1.0)),
            ("(....)", np.log(3.0)),
            ("((..))", np.log(2.0)),
        ], deduplicate=False)
        np.testing.assert_array_equal(original_states, np.array([
            [-1, -1, -1, -1, -1, -1],
            [5, -1, -1, -1, -1, 0],
            [5, 4, -1, -1, 1, 0],
        ], dtype=np.int16))
        np.testing.assert_allclose(
            np.exp(original_logweights), [1.0 / 6.0, 0.5, 1.0 / 3.0]
        )

        weighted_states, weighted_logweights = rna2d.sample_to_numpy([
            ("......", np.log(1.0)),
            ("(....)", np.log(3.0)),
            ("((..))", np.log(2.0)),
        ])
        np.testing.assert_array_equal(weighted_states, original_states[[1, 2, 0]])
        np.testing.assert_allclose(
            np.exp(weighted_logweights), [0.5, 1.0 / 3.0, 1.0 / 6.0]
        )

        with self.assertRaises(ValueError):
            rna2d.sample_to_numpy([])
        with self.assertRaises(ValueError):
            rna2d.sample_to_numpy(["......", ("......", 0.0)])
        with self.assertRaises(ValueError):
            rna2d.sample_to_numpy([("......", np.inf)])
        with self.assertRaises(ValueError):
            rna2d.sample_to_numpy([".....", "......"])
        with self.assertRaises(ValueError):
            rna2d.sample_to_numpy(["((...)"])
        with self.assertRaises(ValueError):
            rna2d.sample_to_numpy([")....."])
        with self.assertRaises(ValueError):
            rna2d.sample_to_numpy(["[....]"])
        with self.assertRaises(ValueError):
            rna2d.sample_to_numpy(["......"], deduplicate="yes")

    def test_pairtable_parser_paths(self):
        structures = [
            "((..))",
            "......",
            "(....)",
        ]
        expected = np.array([
            [5, 4, -1, -1, 1, 0],
            [-1, -1, -1, -1, -1, -1],
            [5, -1, -1, -1, -1, 0],
        ], dtype=np.int16)

        original_has_numba = rna2d._HAS_NUMBA
        results = []
        try:
            # Force both the original Python parser and the encoded batch
            # parser. Without Numba, the latter executes as ordinary Python;
            # a Numba-enabled test run additionally compiles the same function.
            for use_numba_path in (False, True):
                with self.subTest(use_numba_path=use_numba_path):
                    rna2d._HAS_NUMBA = use_numba_path
                    tables = rna2d._structures_to_pairtables(structures)
                    results.append(tables)
                    np.testing.assert_array_equal(tables, expected)
                    np.testing.assert_array_equal(
                        rna2d._dotbracket_to_pairtable(structures[0]),
                        expected[0],
                    )

                    for invalid in ("((...)", ").....", "[....]", "à....."):
                        with self.subTest(invalid=invalid):
                            with self.assertRaises(ValueError):
                                rna2d._dotbracket_to_pairtable(invalid)

                    with self.assertRaises(ValueError):
                        rna2d._structures_to_pairtables([".....", "......"])
                    with self.assertRaises(ValueError):
                        rna2d._structures_to_pairtables(["......", None])
        finally:
            rna2d._HAS_NUMBA = original_has_numba

        np.testing.assert_array_equal(results[0], results[1])

    def test_numba_disabled_selects_python_parser(self):
        command = [
            sys.executable, "-c",
            "import bussilab.rna2d as r; "
            "assert not r._HAS_NUMBA; "
            "assert r._dotbracket_to_pairtable('((..))')[0] == 5",
        ]
        import os
        environment = os.environ.copy()
        environment["NUMBA_DISABLE_JIT"] = "1"
        subprocess.run(command, env=environment, check=True)

    def test_suboptimal_to_numpy(self):

        suboptimal = [
            ("((..))", 0.0),
            ("......", np.log(2.0)),
        ]
        states, logweights = rna2d.suboptimal_to_numpy(
            suboptimal,
            temperature=1.0 / _KB,
        )

        np.testing.assert_array_equal(states, np.array([
            [5, 4, -1, -1, 1, 0],
            [-1, -1, -1, -1, -1, -1],
        ], dtype=np.int16))
        np.testing.assert_allclose(
            np.exp(logweights),
            [2.0 / 3.0, 1.0 / 3.0],
        )
        self.assertAlmostEqual(np.sum(np.exp(logweights)), 1.0)

        with self.assertRaises(ValueError):
            rna2d.suboptimal_to_numpy([], temperature=310.15)
        with self.assertRaises(ValueError):
            rna2d.suboptimal_to_numpy(["......"], temperature=310.15)
        with self.assertRaises(ValueError):
            rna2d.suboptimal_to_numpy(
                [("......", np.nan)],
                temperature=310.15,
            )
        with self.assertRaises(ValueError):
            rna2d.suboptimal_to_numpy(
                [("......", 0.0)],
                temperature=0.0,
            )

    def test_numpy_conversion_validation(self):
        for weight, message in (("bad", "real numbers"),
                                ([0.0], "scalar")):
            with self.subTest(weight=weight):
                with self.assertRaisesRegex(ValueError, message):
                    rna2d.sample_to_numpy([("......", weight)])

        for energy, message in (("bad", "real numbers"),
                                ([0.0], "scalar"),
                                (1.7e308, "too large")):
            with self.subTest(energy=energy):
                with np.errstate(over="ignore"):
                    with self.assertRaisesRegex(ValueError, message):
                        rna2d.suboptimal_to_numpy(
                            [("......", energy)], temperature=310.15)

        with self.assertRaisesRegex(ValueError, "too long"):
            rna2d.sample_to_numpy(["." * 32769])

    def test_constructor(self):

        Molecule(self.seq)

        with self.assertRaises(ValueError):
            Molecule("")

        with self.assertRaises(ValueError):
            Molecule("ABCD")

        with self.assertRaises(ValueError):
            Molecule(self.seq, lambdas1d=np.zeros(len(self.seq)-1))

        with self.assertRaises(ValueError):
            Molecule(self.seq,
                     lambdas1d=np.zeros((2, len(self.seq))))

        with self.assertRaises(ValueError):
            Molecule(self.seq,
                     lambdas1d=np.full(len(self.seq), np.nan))

        with self.assertRaises(ValueError):
            Molecule(self.seq, temperature=-1)

        with self.assertRaises(ValueError):
            Molecule(self.seq, NaCl=-1)

        with self.assertRaises(ValueError):
            Molecule(self.seq, parameters="foo")

        with self.assertRaises(ValueError):
            Molecule(self.seq, no_lonely_pair="yes")

        with self.assertRaises(ValueError):
            Molecule(self.seq, pf_smooth=1)

    def test_thermodynamic_properties(self):
        molecule = Molecule(self.seq, temperature=300.0)
        self.assertEqual(molecule.temperature, 300.0)
        self.assertEqual(molecule.boltzmann_constant, _KB)
        with self.assertRaises(AttributeError):
            molecule.temperature = 310.15
        with self.assertRaises(AttributeError):
            molecule.boltzmann_constant = 1.0

    def test_base_pair_bias_validation(self):
        valid = Molecule(
            self.seq,
            base_pair_biases=[(8, 0, 0.0)],
        )
        self.assertEqual(valid._base_pair_biases, ((0, 8, 0.0),))
        self.assertEqual(
            valid.d_free_energy_d_base_pair_biases().shape,
            (1,),
        )
        self.assertEqual(
            Molecule(self.seq).d_free_energy_d_base_pair_biases().shape,
            (0,),
        )

        invalid = [
            1,
            ["bad"],
            [(0, 8)],
            [(0, 8, 1.0, 2.0)],
            [(False, 8, 1.0)],
            [(0.0, 8, 1.0)],
            [(-1, 8, 1.0)],
            [(0, len(self.seq), 1.0)],
            [(0, 0, 1.0)],
            [(0, 8, np.nan)],
            [(0, 8, "bad")],
            [(0, 8, 1.0), (8, 0, 2.0)],
        ]
        for base_pair_biases in invalid:
            with self.subTest(base_pair_biases=base_pair_biases):
                with self.assertRaises(ValueError):
                    Molecule(
                        self.seq,
                        base_pair_biases=base_pair_biases,
                    )

    def test_impossible_base_pair_biases_are_no_ops(self):
        sequence = "GCGCGCGC"
        biases = [
            (0, 1, -50.004),  # canonical, but too close
            (0, 7, -0.015),  # feasible
            (0, 4, -50.003),  # noncanonical
        ]

        def check():
            molecule = Molecule(sequence, base_pair_biases=biases)
            reference = Molecule(
                sequence,
                base_pair_biases=[biases[1]],
            )
            self.assertEqual(molecule._base_pair_biases, tuple(biases))
            self.assertEqual(
                molecule._prepared_base_pair_biases.sorted_indices,
                (1,),
            )
            np.testing.assert_array_equal(
                molecule._prepared_base_pair_biases.residuals[[0, 2]],
                [0.0, 0.0],
            )
            self.assertEqual(molecule.mfe(), reference.mfe())
            self.assertAlmostEqual(
                molecule.total_free_energy(),
                reference.total_free_energy(),
            )
            np.testing.assert_allclose(
                molecule.base_pairing_probability(),
                reference.base_pairing_probability(),
            )
            np.testing.assert_allclose(
                molecule.d_free_energy_d_base_pair_biases(),
                [0.0, reference.base_pairing_probability()[0, 7], 0.0],
            )
            self.assertEqual(
                molecule.suboptimal_structures(1.0),
                reference.suboptimal_structures(1.0),
            )
            self.assertEqual(
                molecule.evaluate(molecule.mfe()[0]),
                reference.evaluate(reference.mfe()[0]),
            )

        check()
        self._run_in_vanilla_mode(check)

        unpairable = Molecule(
            "AAAAAA",
            base_pair_biases=[(0, 5, -50.004)],
        )
        self.assertEqual(unpairable.mfe(), Molecule("AAAAAA").mfe())
        self.assertEqual(
            unpairable.d_free_energy_d_base_pair_biases().tolist(),
            [0.0],
        )

        excluded = Molecule(
            sequence,
            force_unpaired=[0],
            base_pair_biases=[(0, 7, -50.004)],
        )
        self.assertEqual(
            excluded.d_free_energy_d_base_pair_biases().tolist(),
            [0.0],
        )

    def test_default_parameters(self):

        rna2d.reset_default_parameters()
        original = Molecule(self.seq)

        try:
            rna2d.set_default_parameters(
                temperature=298.15,
                no_lonely_pair=True,
                pf_smooth=True,
                NaCl=2.0,
                parameters="turner1999",
            )

            inherited = Molecule(self.seq)
            self.assertEqual(inherited._temperature, 298.15)
            self.assertEqual(inherited.temperature, 298.15)
            self.assertTrue(inherited._no_lonely_pair)
            self.assertTrue(inherited._pf_smooth)
            self.assertEqual(inherited._salt, 2.0)
            self.assertEqual(inherited._parameters, "turner1999")

            # Existing molecules retain the defaults captured at construction.
            self.assertEqual(
                original._temperature,
                37 + rna2d._CELSIUS_TO_KELVIN,
            )
            self.assertEqual(
                original.temperature,
                37 + rna2d._CELSIUS_TO_KELVIN,
            )
            self.assertFalse(original._no_lonely_pair)
            self.assertFalse(original._pf_smooth)
            self.assertIsNone(original._salt)
            self.assertEqual(original._parameters, "turner2004")

            explicit = Molecule(
                self.seq,
                temperature=305.0,
                no_lonely_pair=False,
                pf_smooth=False,
                NaCl=0.5,
                parameters="turner2004",
            )
            self.assertEqual(explicit._temperature, 305.0)
            self.assertFalse(explicit._no_lonely_pair)
            self.assertFalse(explicit._pf_smooth)
            self.assertEqual(explicit._salt, 0.5)
            self.assertEqual(explicit._parameters, "turner2004")

            # None is consistently a sentinel for retaining/inheriting the
            # current defaults, including NaCl.
            defaults_before = rna2d._default_parameters.copy()
            rna2d.set_default_parameters(
                temperature=None,
                no_lonely_pair=None,
                pf_smooth=None,
                NaCl=None,
                parameters=None,
            )
            self.assertEqual(
                rna2d._default_parameters,
                defaults_before,
            )
            for value in (0.0, np.inf, np.nan):
                with self.subTest(temperature=value):
                    with self.assertRaises(ValueError):
                        rna2d.set_default_parameters(temperature=value)
                    with self.assertRaises(ValueError):
                        Molecule(self.seq, temperature=value)
            for value in (np.inf, np.nan):
                with self.subTest(NaCl=value):
                    with self.assertRaises(ValueError):
                        rna2d.set_default_parameters(NaCl=value)
                    with self.assertRaises(ValueError):
                        Molecule(self.seq, NaCl=value)
            self.assertEqual(Molecule(self.seq, NaCl=None)._salt, 2.0)

            # Validation is atomic: no valid values preceding an invalid one
            # are installed.
            with self.assertRaises(ValueError):
                rna2d.set_default_parameters(
                    temperature=280.0,
                    no_lonely_pair="yes",
                )
            self.assertEqual(
                rna2d._default_parameters,
                defaults_before,
            )
        finally:
            rna2d.reset_default_parameters()

        reset = Molecule(self.seq)
        self.assertEqual(
            reset._temperature,
            37 + rna2d._CELSIUS_TO_KELVIN,
        )
        self.assertFalse(reset._no_lonely_pair)
        self.assertFalse(reset._pf_smooth)
        self.assertIsNone(reset._salt)
        self.assertEqual(reset._parameters, "turner2004")

    def test_default_parameter_validation(self):
        rna2d.reset_default_parameters()
        try:
            for kwargs, message in (({"pf_smooth": "yes"}, "pf_smooth"),
                                    ({"parameters": "unknown"}, "not known")):
                with self.subTest(kwargs=kwargs):
                    with self.assertRaisesRegex(ValueError, message):
                        rna2d.set_default_parameters(**kwargs)
            self.assertFalse(Molecule(self.seq)._pf_smooth)
        finally:
            rna2d.reset_default_parameters()

    def test_no_lonely_pair(self):

        default = Molecule(self.seq)
        no_lonely_pair = Molecule(
            self.seq,
            state_positions=(0, 1),
            no_lonely_pair=True,
        )

        self.assertEqual(
            default._dp_molecules[0]._make_md_params().noLP,
            0,
        )
        self.assertTrue(all(
            component._make_md_params().noLP == 1
            for component in no_lonely_pair._dp_molecules
        ))

        conditioned = no_lonely_pair._condition_unpaired(2)
        self.assertTrue(conditioned._no_lonely_pair)
        self.assertTrue(all(
            component._make_md_params().noLP == 1
            for component in conditioned._dp_molecules
        ))

    def test_pf_smooth(self):

        sequence = "GGGAAACCC"
        energies = []

        for pf_smooth in (False, True):
            molecule = Molecule(sequence, pf_smooth=pf_smooth)
            energy = molecule.total_free_energy()

            # Construct the independent reference after Molecule has selected
            # its parameter set, since ViennaRNA parameter loads are global.
            md = RNA.md()
            md.uniq_ML = 1
            md.pf_smooth = int(pf_smooth)
            reference = RNA.fold_compound(sequence, md).pf()[1]

            self.assertEqual(molecule._pf_smooth, pf_smooth)
            self.assertTrue(all(
                component._make_md_params().pf_smooth == int(pf_smooth)
                for component in molecule._dp_molecules
            ))
            self.assertAlmostEqual(
                energy,
                reference,
                places=6,
            )
            energies.append(reference)

        # This sequence makes the test sensitive to the option rather than
        # merely checking that it is forwarded to ViennaRNA.
        self.assertNotAlmostEqual(energies[0], energies[1], places=5)

        mixture = Molecule(
            sequence,
            state_positions=(0, 1),
            pf_smooth=True,
        )
        conditioned = mixture._condition_unpaired(2)
        self.assertTrue(conditioned._pf_smooth)
        self.assertTrue(all(
            component._make_md_params().pf_smooth == 1
            for component in conditioned._dp_molecules
        ))

    def test_state_mixture_constructor(self):

        unbiased = Molecule(self.seq, state_positions=(0,))
        np.testing.assert_array_equal(
            unbiased._state_biases,
            np.zeros(2),
        )
        populations = unbiased.d_free_energy_d_state_biases()
        self.assertEqual(populations.shape, (2,))
        self.assertAlmostEqual(np.sum(populations), 1.0)

        with self.assertRaises(ValueError):
            Molecule(self.seq, state_biases=np.zeros(2))

        with self.assertRaises(ValueError):
            Molecule(self.seq, reduce_state_space="yes")

        with self.assertRaises(ValueError):
            Molecule(
                self.seq,
                state_positions=(0, 1),
                state_biases=np.zeros(4),
            )

        with self.assertRaises(ValueError):
            Molecule(
                self.seq,
                state_positions=(0, 0),
                state_biases=np.zeros((2, 2)),
            )

        with self.assertRaises(ValueError):
            Molecule(
                self.seq,
                state_positions=(len(self.seq),),
                state_biases=np.zeros(2),
            )

        with self.assertRaises(ValueError):
            Molecule(
                self.seq,
                state_positions=(0,),
                state_biases=np.array([0.0, np.nan]),
            )

        with self.assertRaises(ValueError):
            Molecule(
                self.seq,
                force_paired=(0,),
                state_positions=(0,),
                state_biases=np.zeros(2),
            )

        multiple = Molecule(
            self.seq,
            state_positions=((0, 1), (7, 8)),
        )
        self.assertEqual(
            multiple._state_position_sets,
            ((0, 1), (7, 8)),
        )
        self.assertEqual(len(multiple._dp_molecules), 4)
        derivatives = multiple.d_free_energy_d_state_biases()
        self.assertIsInstance(derivatives, list)
        self.assertEqual(len(derivatives), 2)
        for derivative in derivatives:
            self.assertEqual(derivative.shape, (2, 2))
            self.assertAlmostEqual(np.sum(derivative), 1.0)

        nested_single = Molecule(
            self.seq,
            state_positions=((0, 1),),
            state_biases=(np.zeros((2, 2)),),
        )
        self.assertEqual(nested_single._state_positions, ((0, 1),))
        self.assertIsInstance(nested_single._state_biases, tuple)
        self.assertEqual(len(nested_single._state_biases), 1)
        nested_derivatives = (
            nested_single.d_free_energy_d_state_biases()
        )
        self.assertIsInstance(nested_derivatives, list)
        self.assertEqual(len(nested_derivatives), 1)
        self.assertEqual(nested_derivatives[0].shape, (2, 2))

        with self.assertRaises(ValueError):
            Molecule(
                self.seq,
                state_positions=((0, 1), (1, 2)),
            )

        with self.assertRaises(ValueError):
            Molecule(
                self.seq,
                state_positions=((0, 1), (7, 8)),
                state_biases=(np.zeros((2, 2)),),
            )

        with self.assertRaises(ValueError):
            Molecule(
                self.seq,
                state_positions=((0, 1), (7, 8)),
                state_biases=(np.zeros((2, 2)), np.zeros(4)),
            )

    def test_additional_input_validation(self):
        invalid = (
            ({"force_paired": 1}, "iterable"),
            ({"state_positions": 1}, "sequence"),
            ({"state_positions": ((0,), np.zeros((2, 2), dtype=int))},
             "one-dimensional"),
            ({"state_positions": ((0.5,),)}, "integers"),
            ({"state_positions": (0,), "state_biases": ["bad"]}, "shape"),
            ({"state_positions": ((0,), (8,)), "state_biases": 1},
             "one tensor"),
        )
        for kwargs, message in invalid:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    Molecule(self.seq, **kwargs)

        molecule = Molecule(self.seq)
        for action in (lambda: molecule.sample(0),
                       lambda: molecule.sample(1, weights="yes"),
                       lambda: molecule.suboptimal_structures(-1)):
            with self.assertRaises(ValueError):
                action()
        mixture = Molecule(self.seq, state_positions=(0,),
                           reduce_state_space=False)
        with self.assertRaisesRegex(ValueError, "delta"):
            mixture.suboptimal_structures(-1)

    def test_infeasible_state_mixture(self):
        molecule = Molecule(
            "AAAAA", force_paired=(0,), state_positions=(1,),
            reduce_state_space=False,
        )
        with self.assertRaisesRegex(RuntimeError, "No state has a feasible MFE"):
            molecule.mfe()
        with self.assertRaisesRegex(RuntimeError, "No state has a finite partition"):
            molecule.total_free_energy()

    def test_condition_unpaired_with_nested_states(self):
        molecule = Molecule(
            self.seq,
            force_paired=(0,),
            force_unpaired=(1,),
            state_positions=((2,), (7,)),
            state_biases=([0.0, 0.1], [0.0, 0.2]),
            reduce_state_space=False,
        )
        self.assertIsNone(molecule._condition_unpaired(0))
        self.assertIs(molecule._condition_unpaired(1), molecule)
        conditioned = molecule._condition_unpaired(2)
        self.assertIn(2, conditioned._force_unpaired)
        self.assertEqual(conditioned._state_position_sets, ((), (7,)))
        np.testing.assert_allclose(conditioned._state_bias_sets[1], [0.0, 0.2])

    def test_unweighted_state_mixture_sampling(self):
        molecule = Molecule(
            self.seq, state_positions=(0,), reduce_state_space=False)
        samples = molecule.sample(10)
        self.assertEqual(len(samples), 10)
        self.assertTrue(all(isinstance(sample, str) and
                            len(sample) == len(self.seq) for sample in samples))

    def test_parameter_sets(self):

        parameter_loaders = {
            "turner1999": "params_load_RNA_Turner1999",
            "turner2004": "params_load_RNA_Turner2004",
            "andronescu2007": "params_load_RNA_Andronescu2007",
            "langdon2018": "params_load_RNA_Langdon2018",
        }

        for parameters, loader in parameter_loaders.items():
            # The first parameter load in a fresh process is reliable even in
            # ViennaRNA 2.7.2. It provides an independent reference that makes
            # this test sensitive to stale cache reuse between loads here.
            code = (
                "import RNA; "
                f"RNA.{loader}(); "
                "md=RNA.md(); md.uniq_ML=1; md.pf_smooth=0; "
                f"fc=RNA.fold_compound({self.seq!r}, md); "
                "print(fc.mfe()[1], fc.pf()[1])"
            )
            output = subprocess.check_output(
                [sys.executable, "-c", code],
                text=True,
            )
            reference_mfe, reference_free_energy = map(
                float,
                output.splitlines()[-1].split(),
            )

            molecule = Molecule(self.seq, parameters=parameters)
            self.assertAlmostEqual(
                molecule.mfe()[1],
                reference_mfe,
                places=6,
            )
            self.assertAlmostEqual(
                molecule.total_free_energy(),
                reference_free_energy,
                places=6,
            )

    def test_external_parameter_load_is_detected(self):
        reference = Molecule(self.seq, parameters="turner2004")
        reference_mfe = reference.mfe()
        reference_free_energy = reference.total_free_energy()
        reference_parameter_file = RNA.last_parameter_file()

        # Change ViennaRNA's global state without going through this module.
        RNA.params_load_RNA_Turner1999()
        self.assertNotEqual(
            RNA.last_parameter_file(),
            reference_parameter_file,
        )

        # Construction must notice the external change and reload Turner 2004
        # even though it is also the last parameter key requested above.
        restored = Molecule(self.seq, parameters="turner2004")
        self.assertEqual(restored.mfe()[0], reference_mfe[0])
        self.assertAlmostEqual(restored.mfe()[1], reference_mfe[1])
        self.assertAlmostEqual(
            restored.total_free_energy(),
            reference_free_energy,
        )
        self.assertEqual(
            RNA.last_parameter_file(),
            reference_parameter_file,
        )

    def test_hard_constraints(self):

        paired = 0
        unpaired = 1
        mol = Molecule(
            self.seq,
            force_paired=(paired,),
            force_unpaired=(unpaired,),
        )

        def satisfies_constraints(structure):
            return structure[paired] != "." and structure[unpaired] == "."

        structure, _ = mol.mfe()
        self.assertTrue(satisfies_constraints(structure))

        probabilities = np.sum(
            mol.base_pairing_probability(),
            axis=1,
        )
        self.assertAlmostEqual(probabilities[paired], 1.0)
        self.assertAlmostEqual(probabilities[unpaired], 0.0)

        self.assertTrue(all(
            satisfies_constraints(structure)
            for structure, _ in mol.suboptimal_structures(3.0)
        ))
        self.assertTrue(all(
            satisfies_constraints(structure)
            for structure in mol.sample(100)
        ))

    def test_hard_constraint_validation_and_infeasibility(self):
        invalid = (
            {"force_paired": (len(self.seq),)},
            {"force_unpaired": (-1,)},
            {"force_paired": (0.5,)},
            {"force_paired": (True,)},
            {"force_unpaired": (0, 0)},
            {"force_paired": (0,), "force_unpaired": (0,)},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Molecule(self.seq, **kwargs)

        impossible = Molecule("AAAAA", force_paired=(0,))
        with self.assertRaisesRegex(RuntimeError, "No feasible MFE"):
            impossible.mfe()
        self.assertEqual(impossible.suboptimal_structures(1.0), [])
        with self.assertRaisesRegex(RuntimeError, "No feasible suboptimal"):
            impossible.suboptimal_coverage(1.0)

        # An impossible component must be skipped even when a residual forces
        # the suboptimal-candidate branch of the internal MFE calculation.
        for reduced in (False, True):
            molecule = Molecule(
                "AAAAA",
                state_positions=(0,),
                reduce_state_space=reduced,
                lambdas1d=np.full(5, 0.003),
            )
            self.assertEqual(molecule.mfe()[0], ".....")
            self.assertEqual(molecule.suboptimal_structures(0.01)[0][0], ".....")

    def test_mfe(self):

        mol = Molecule(self.seq)

        structure, energy = mol.mfe()

        self.assertEqual(len(structure), len(self.seq))
        self.assertIsInstance(structure, str)
        self.assertIsInstance(energy, float)

    def test_evaluate(self):
        molecule = Molecule(self.seq)
        suboptimal = molecule.suboptimal_structures(5.0)
        for structure, energy in suboptimal:
            self.assertAlmostEqual(
                molecule.evaluate(structure),
                energy,
            )

        np.testing.assert_allclose(
            molecule.evaluate(suboptimal),
            [energy for _, energy in suboptimal],
        )

        structures = [structure for structure, _ in suboptimal]
        np.testing.assert_allclose(
            molecule.evaluate(structures),
            [molecule.evaluate(structure) for structure in structures],
        )

        structure_array = np.asarray(structures[:4])
        evaluated_array = molecule.evaluate(structure_array)
        self.assertEqual(evaluated_array.shape, structure_array.shape)
        np.testing.assert_allclose(
            evaluated_array,
            molecule.evaluate(structures[:4]),
        )

        samples = molecule.sample(10)
        np.testing.assert_allclose(
            molecule.evaluate(samples),
            [molecule.evaluate(structure) for structure in samples],
        )
        weighted_samples = molecule.sample(10, weights=True)
        np.testing.assert_allclose(
            molecule.evaluate(weighted_samples),
            [
                molecule.evaluate(structure)
                for structure, _ in weighted_samples
            ],
        )

        mixed = [structures[0], suboptimal[1], structures[2]]
        np.testing.assert_allclose(
            molecule.evaluate(mixed),
            molecule.evaluate([structures[0], structures[1], structures[2]]),
        )
        np.testing.assert_allclose(
            molecule.evaluate(item for item in mixed),
            molecule.evaluate(mixed),
        )

        self.assertEqual(molecule.evaluate([]).shape, (0,))

        with self.assertRaises(ValueError):
            molecule.evaluate(None)
        with self.assertRaises(ValueError):
            molecule.evaluate("." * (len(self.seq) - 1))
        with self.assertRaises(ValueError):
            molecule.evaluate("x" * len(self.seq))
        with self.assertRaises(ValueError):
            molecule.evaluate(["." * len(self.seq), None])
        with self.assertRaises(ValueError):
            molecule.evaluate(structure_array.reshape(2, 2))
        with self.assertRaises(ValueError):
            molecule.evaluate(np.asarray(suboptimal))

        for malformed in ("((.......", ".......))", ")......(."):
            command = [
                sys.executable, "-c",
                "from bussilab.rna2d import Molecule; "
                "Molecule('GGGAAACCC').evaluate(%r)" % malformed,
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ValueError", result.stderr)
            self.assertNotIn("Segmentation fault", result.stderr)

        constrained = Molecule(self.seq, force_paired=(0,))
        self.assertEqual(constrained.evaluate("." * len(self.seq)), np.inf)
        valid_structure = constrained.mfe()[0]
        np.testing.assert_array_equal(
            constrained.evaluate(["." * len(self.seq), valid_structure]),
            [np.inf, constrained.evaluate(valid_structure)],
        )

    def test_evaluate_pairs_outside_folding_grammar(self):
        too_short = Molecule(
            "GC",
            lambdas1d=[0.013, -0.017],
            base_pair_biases=[(0, 1, -0.015)],
        )
        self.assertEqual(too_short.evaluate("()"), np.inf)
        energies = too_short.evaluate(["..", "()"])
        self.assertAlmostEqual(energies[0], 0.0)
        self.assertEqual(energies[1], np.inf)

        sequence = "GAAAA"
        structure = "(...)"
        lambdas = [0.013, 0.0, 0.0, 0.0, -0.017]
        bias = -0.015
        self.assertTrue(np.isfinite(Molecule("GAAAC").evaluate(structure)))
        self.assertEqual(Molecule(sequence).evaluate(structure), np.inf)

        def check():
            molecule = Molecule(
                sequence,
                lambdas1d=lambdas,
                base_pair_biases=[(0, 4, bias)],
            )
            self.assertEqual(molecule.evaluate(structure), np.inf)
            self.assertEqual(
                molecule.evaluate([(structure, 123.0)])[0],
                np.inf,
            )
            self.assertEqual(
                molecule.d_free_energy_d_base_pair_biases().tolist(),
                [0.0],
            )

        check()
        self._run_in_vanilla_mode(check)

        mixture = Molecule(
            sequence,
            lambdas1d=lambdas,
            base_pair_biases=[(0, 4, bias)],
            state_positions=[0],
            state_biases=[0.0, 0.25],
            reduce_state_space=False,
        )
        self.assertEqual(mixture.evaluate(structure), np.inf)

        constrained = Molecule(sequence, force_unpaired=[0])
        self.assertEqual(constrained.evaluate(structure), np.inf)

    def test_dotbracket_validator(self):
        for sequence, structure, forbidden in (
            ("GCGCGCGC", "((....))", False),
            ("GCGCGCGC", "(..)....", True),
            ("AAAAAAAA", "(......)", True),
        ):
            with self.subTest(sequence=sequence, structure=structure):
                self.assertEqual(
                    rna2d._validate_dotbracket_structure(
                        structure, sequence
                    ),
                    forbidden,
                )

        for structure in ("((......", "......))", "(..x...)", "é......."):
            with self.subTest(structure=structure), self.assertRaises(
                ValueError
            ):
                rna2d._validate_dotbracket_structure(
                    structure, "GCGCGCGC"
                )

    def test_partition_function(self):

        mol = Molecule(self.seq)

        mfe = mol.mfe()[1]
        F = mol.total_free_energy()

        self.assertLessEqual(F, mfe)

        bpp = mol.base_pairing_probability()

        self.assertEqual(
            bpp.shape,
            (len(self.seq), len(self.seq))
        )

        self.assertTrue(np.allclose(bpp, bpp.T))
        self.assertAlmostEqual(np.trace(bpp), 0.0)

    def test_mean_bp_distance(self):
        molecule = Molecule(self.seq)
        diversity = molecule.mean_bp_distance()
        probability = molecule.base_pairing_probability()

        self.assertAlmostEqual(
            diversity,
            np.sum(probability * (1.0 - probability)),
        )
        self.assertAlmostEqual(
            diversity,
            molecule._dp_molecules[0]._fc.mean_bp_distance(),
        )

        mixture = Molecule(
            self.seq,
            state_positions=(0, 4),
            state_biases=np.array([
                [0.0, 0.4],
                [-0.3, 0.2],
            ]),
        )
        probability = mixture.base_pairing_probability()
        self.assertAlmostEqual(
            mixture.mean_bp_distance(),
            np.sum(probability * (1.0 - probability)),
        )

    def test_partition_function_vanilla(self):
        self._run_in_vanilla_mode(self.test_partition_function)

    def test_scalar_partition_function_omits_bpp_backtracking(self):
        molecule = Molecule(self.seq)
        dp_molecule = molecule._dp_molecules[0]

        free_energy = molecule.total_free_energy()
        scalar_fc = dp_molecule._fc
        self.assertFalse(dp_molecule._fc_compute_bpp)
        self.assertIsNone(dp_molecule._base_pairing_probability)

        bpp = molecule.base_pairing_probability()
        self.assertTrue(dp_molecule._fc_compute_bpp)
        self.assertIsNot(dp_molecule._fc, scalar_fc)
        self.assertEqual(bpp.shape, (len(self.seq), len(self.seq)))
        self.assertEqual(molecule.total_free_energy(), free_energy)

        molecule = Molecule(self.seq)
        dp_molecule = molecule._dp_molecules[0]
        molecule.base_pairing_probability()
        probability_fc = dp_molecule._fc
        molecule.total_free_energy()
        self.assertIs(dp_molecule._fc, probability_fc)

    def test_partition_function_rescaling_fallback(self):

        class FakeFoldCompound:

            def __init__(self, free_energies):
                self.free_energies = iter(free_energies)
                self.mfe_calls = 0
                self.rescaled_with = []

            def pf(self):
                return None, next(self.free_energies)

            def mfe(self):
                self.mfe_calls += 1
                return None, -3.5

            def exp_params_rescale(self, energy):
                self.rescaled_with.append(energy)

        fast = FakeFoldCompound([-2.0])
        self.assertEqual(
            rna2d._pf_with_mfe_rescaling_fallback(fast),
            -2.0,
        )
        self.assertEqual(fast.mfe_calls, 0)
        self.assertEqual(fast.rescaled_with, [])

        retry = FakeFoldCompound([RNA.INF / 100.0, -2.0])
        self.assertEqual(
            rna2d._pf_with_mfe_rescaling_fallback(retry),
            -2.0,
        )
        self.assertEqual(retry.mfe_calls, 1)
        self.assertEqual(retry.rescaled_with, [-3.5])

    def test_partition_function_rescaling_fallback_integration(self):
        sequence = "G" * 300 + "C" * 300

        md = RNA.md()
        md.uniq_ML = 1
        unscaled_free_energy = RNA.fold_compound(sequence, md).pf()[1]
        self.assertGreaterEqual(
            unscaled_free_energy,
            RNA.INF / 100.0,
        )

        free_energy = Molecule(sequence).total_free_energy()
        self.assertTrue(np.isfinite(free_energy))
        self.assertLess(free_energy, RNA.INF / 100.0)

    def test_free_energy_derivatives(self):

        lambdas = np.array(
            [0.004, -0.006, 0.013, -0.017, 0.021, -0.009, 0.007, -0.012, 0.003]
        )
        mol = Molecule(self.seq, lambdas1d=lambdas)

        derivatives = mol.d_free_energy_d_lambdas1d()
        np.testing.assert_allclose(
            derivatives,
            np.sum(mol.base_pairing_probability(), axis=1),
            atol=1e-14,
        )

        epsilon = 1e-3
        position = 0
        lambdas_plus = lambdas.copy()
        lambdas_minus = lambdas.copy()
        lambdas_plus[position] += epsilon
        lambdas_minus[position] -= epsilon
        finite_difference = (
            Molecule(
                self.seq,
                lambdas1d=lambdas_plus,
            ).total_free_energy()
            - Molecule(
                self.seq,
                lambdas1d=lambdas_minus,
            ).total_free_energy()
        ) / (2.0 * epsilon)
        self.assertAlmostEqual(
            derivatives[position],
            finite_difference,
            delta=2e-4,
        )

        with self.assertRaises(ValueError):
            mol.d_free_energy_d_state_biases()

        biases = np.array([
            [0.0, 0.2],
            [-0.1, 0.7],
        ])
        mixture = Molecule(
            self.seq,
            state_positions=(0, 1),
            state_biases=biases,
        )
        state_derivatives = (
            mixture.d_free_energy_d_state_biases()
        )
        self.assertEqual(state_derivatives.shape, biases.shape)
        self.assertAlmostEqual(np.sum(state_derivatives), 1.0)

        index = (1, 0)
        biases_plus = biases.copy()
        biases_minus = biases.copy()
        biases_plus[index] += epsilon
        biases_minus[index] -= epsilon
        finite_difference = (
            Molecule(
                self.seq,
                state_positions=(0, 1),
                state_biases=biases_plus,
            ).total_free_energy()
            - Molecule(
                self.seq,
                state_positions=(0, 1),
                state_biases=biases_minus,
            ).total_free_energy()
        ) / (2.0 * epsilon)
        self.assertAlmostEqual(
            state_derivatives[index],
            finite_difference,
            delta=2e-4,
        )

    def test_base_pair_biases(self):
        sequence = "GGGAAACCC"
        base_pair_biases = [
            (8, 0, 0.013),
            (1, 7, -0.027),
            (2, 6, 0.0),
        ]
        molecule = Molecule(
            sequence,
            base_pair_biases=base_pair_biases,
        )
        self.assertEqual(
            molecule._base_pair_biases,
            (
                (0, 8, 0.013),
                (1, 7, -0.027),
                (2, 6, 0.0),
            ),
        )
        conditioned = molecule._condition_unpaired(4)
        self.assertEqual(
            conditioned._base_pair_biases,
            molecule._base_pair_biases,
        )

        dp_molecule = molecule._dp_molecules[0]
        base_fc = dp_molecule._make_fold_compound()
        exact_energies = {}
        for structure in _enumerate_secondary_structures(sequence):
            table = rna2d._dotbracket_to_pairtable(structure)
            exact_energies[structure] = float(
                base_fc.eval_structure(structure)
                + sum(
                    energy
                    for i, j, energy in molecule._base_pair_biases
                    if table[i] == j
                )
            )

        structures = list(exact_energies)
        np.testing.assert_allclose(
            molecule.evaluate(structures),
            [exact_energies[structure] for structure in structures],
            atol=1e-6,
        )

        mfe_structure, mfe_energy = molecule.mfe()
        self.assertAlmostEqual(
            mfe_energy,
            min(exact_energies.values()),
            places=6,
        )
        self.assertAlmostEqual(
            mfe_energy,
            exact_energies[mfe_structure],
            places=6,
        )

        delta = 1.0
        cutoff = min(exact_energies.values()) + delta
        expected_suboptimal = {
            structure: energy
            for structure, energy in exact_energies.items()
            if energy <= cutoff
        }
        actual_suboptimal = dict(
            molecule.suboptimal_structures(delta)
        )
        self.assertEqual(set(actual_suboptimal), set(expected_suboptimal))
        for structure, energy in actual_suboptimal.items():
            self.assertAlmostEqual(
                energy,
                expected_suboptimal[structure],
                places=6,
            )

        energies = np.asarray(list(exact_energies.values()))
        reference = np.min(energies)
        kT = _KB * molecule._temperature
        expected_free_energy = reference - kT * np.log(np.sum(
            np.exp(-(energies - reference) / kT)
        ))
        self.assertAlmostEqual(
            molecule.total_free_energy(),
            expected_free_energy,
            places=6,
        )

        base_pairing_probability = molecule.base_pairing_probability()
        derivatives = molecule.d_free_energy_d_base_pair_biases()
        np.testing.assert_allclose(
            derivatives,
            [
                base_pairing_probability[i, j]
                for i, j, _ in molecule._base_pair_biases
            ],
            atol=1e-14,
        )

        epsilon = 1e-3
        plus = list(base_pair_biases)
        minus = list(base_pair_biases)
        plus[0] = plus[0][:2] + (plus[0][2] + epsilon,)
        minus[0] = minus[0][:2] + (minus[0][2] - epsilon,)
        finite_difference = (
            Molecule(
                sequence,
                base_pair_biases=plus,
            ).total_free_energy()
            - Molecule(
                sequence,
                base_pair_biases=minus,
            ).total_free_energy()
        ) / (2.0 * epsilon)
        self.assertAlmostEqual(
            derivatives[0],
            finite_difference,
            delta=2e-4,
        )

        np.random.seed(1977)
        inverse_kT = 1.0 / kT
        for structure, log_weight in molecule.sample(50, weights=True):
            table = rna2d._dotbracket_to_pairtable(structure)
            expected_correction = sum(
                residual
                for (i, j), residual in zip(
                    dp_molecule._base_pair_biases.pairs,
                    dp_molecule._base_pair_biases.residuals,
                )
                if table[i] == j
            )
            self.assertAlmostEqual(
                log_weight,
                -expected_correction * inverse_kT,
            )

        mixture = Molecule(
            sequence,
            base_pair_biases=base_pair_biases,
            state_positions=(3, 4),
            state_biases=np.zeros((2, 2)),
        )
        self.assertTrue(all(
            component._base_pair_biases
            is mixture._prepared_base_pair_biases
            for component in mixture._dp_molecules
        ))
        np.testing.assert_allclose(
            mixture.evaluate(structures),
            molecule.evaluate(structures),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            mixture.base_pairing_probability(),
            molecule.base_pairing_probability(),
            atol=1e-10,
        )

    def test_base_pair_biases_vanilla(self):
        self._run_in_vanilla_mode(self.test_base_pair_biases)

    def test_rescore_base_pair_biased_suboptimals(self):
        sequence = "GGGAAACCC"
        base_pair_biases = [(0, 8, -2.0), (1, 7, 0.5)]
        source = Molecule(
            sequence,
            base_pair_biases=base_pair_biases,
        )
        suboptimal = source.suboptimal_structures(5.0)

        unbiased = Molecule(sequence)
        rescored = unbiased.evaluate(suboptimal)
        np.testing.assert_allclose(
            rescored,
            [unbiased.evaluate(structure) for structure, _ in suboptimal],
        )
        self.assertTrue(any(
            abs(new_energy - old_energy) > 1e-8
            for new_energy, (_, old_energy) in zip(rescored, suboptimal)
        ))

        state_positions = (0, 4)
        state_biases = np.array([
            [0.2, -0.3],
            [0.4, -0.1],
        ])
        state_source = Molecule(
            sequence,
            base_pair_biases=base_pair_biases,
            state_positions=state_positions,
            state_biases=state_biases,
        )
        state_suboptimal = state_source.suboptimal_structures(5.0)

        # Rescore with the state model retained but the base-pair biases
        # removed, and also with every bias removed.
        state_target = Molecule(
            sequence,
            state_positions=state_positions,
            state_biases=state_biases,
        )
        np.testing.assert_allclose(
            state_target.evaluate(state_suboptimal),
            [
                state_target.evaluate(structure)
                for structure, _ in state_suboptimal
            ],
        )
        np.testing.assert_allclose(
            unbiased.evaluate(state_suboptimal),
            [
                unbiased.evaluate(structure)
                for structure, _ in state_suboptimal
            ],
        )

    def test_base_pair_residual_callback_selection(self):
        class FoldCompound:
            callback = None

            def sc_add_exp_f(self, callback):
                self.callback = callback

        sequence = "GGGAAACCC"
        kT = _KB * (37.0 + 273.15)
        pair_decomposition = next(iter(rna2d._PAIR_DECOMPOSITIONS))

        # Exactly representable 2D biases must not add a pair lookup to the
        # existing 1D-only callback.
        exact = rna2d._prepare_base_pair_biases(
            sequence,
            [(0, 8, 0.02)],
        )
        exact.residual_rows = None
        residuals = np.zeros(len(sequence))
        residuals[[0, 8]] = (0.003, -0.001)
        fc = FoldCompound()
        callback = rna2d._apply_residual_callback(
            fc,
            residuals,
            exact,
            kT,
        )
        self.assertIs(callback, fc.callback)
        self.assertAlmostEqual(
            callback(1, 9, 0, 0, pair_decomposition, None),
            np.exp(-0.002 / kT),
        )

        # With no 1D or 2D residual, no callback is installed at all.
        fc = FoldCompound()
        self.assertIsNone(rna2d._apply_residual_callback(
            fc,
            np.zeros(len(sequence)),
            exact,
            kT,
        ))
        self.assertIsNone(fc.callback)

        # When a 2D residual is present, it is added only for its exact pair.
        inexact = rna2d._prepare_base_pair_biases(
            sequence,
            [(0, 8, 0.023)],
        )
        fc = FoldCompound()
        callback = rna2d._apply_residual_callback(
            fc,
            residuals,
            inexact,
            kT,
        )
        self.assertAlmostEqual(
            callback(1, 9, 0, 0, pair_decomposition, None),
            np.exp(-0.005 / kT),
        )
        self.assertAlmostEqual(
            callback(2, 8, 0, 0, pair_decomposition, None),
            1.0,
        )
        self.assertEqual(
            callback(1, 9, 0, 0, object(), None),
            1.0,
        )

    def test_combined_1d_and_base_pair_residuals_vanilla(self):
        sequence = "GGGAAACCC"
        lambdas = np.array([
            0.003, -0.004, 0.002, 0.0, 0.0,
            0.0, -0.003, 0.004, -0.002,
        ])
        base_pair_biases = [
            (0, 8, 0.013),
            (1, 7, -0.027),
        ]

        def check():
            molecule = Molecule(
                sequence,
                lambdas1d=lambdas,
                base_pair_biases=base_pair_biases,
            )
            dp_molecule = molecule._dp_molecules[0]
            base_fc = dp_molecule._make_fold_compound()
            structures = list(_enumerate_secondary_structures(sequence))
            energies = []
            pair_indicators = []
            for structure in structures:
                table = rna2d._dotbracket_to_pairtable(structure)
                indicators = np.asarray([
                    table[i] == j
                    for i, j, _ in base_pair_biases
                ])
                pair_indicators.append(indicators)
                energies.append(
                    base_fc.eval_structure(structure)
                    + sum(
                        value
                        for value, symbol in zip(lambdas, structure)
                        if symbol != "."
                    )
                    + sum(
                        value
                        for indicator, (_, _, value) in zip(
                            indicators,
                            base_pair_biases,
                        )
                        if indicator
                    )
                )

            energies = np.asarray(energies)
            reference = np.min(energies)
            kT = _KB * molecule._temperature
            weights = np.exp(-(energies - reference) / kT)
            expected_free_energy = reference - kT * np.log(np.sum(weights))
            expected_derivatives = (
                weights @ np.asarray(pair_indicators)
                / np.sum(weights)
            )

            self.assertAlmostEqual(
                molecule.total_free_energy(),
                expected_free_energy,
                places=6,
            )
            np.testing.assert_allclose(
                molecule.d_free_energy_d_base_pair_biases(),
                expected_derivatives,
                atol=1e-7,
            )

        self._run_in_vanilla_mode(check)

    def test_combined_1d_base_pair_and_state_biases(self):
        sequence = self.seq
        lambdas = np.array([
            0.003, -0.004, 0.002, 0.0, 0.0,
            0.0, -0.003, 0.004, -0.002,
        ])
        pair_biases = [(0, 8, 0.013), (1, 7, -0.027)]
        state_biases = np.array([[0.123, -0.234], [0.345, 0.456]])

        def check():
            molecule = Molecule(
                sequence,
                lambdas1d=lambdas,
                base_pair_biases=pair_biases,
                state_positions=(0, 8),
                state_biases=state_biases,
            )
            fold_compound = molecule._dp_molecules[0]._make_fold_compound()
            structures = list(_enumerate_secondary_structures(sequence))
            energies = []
            indicators = []
            for structure in structures:
                table = rna2d._dotbracket_to_pairtable(structure)
                pair_indicators = [table[i] == j for i, j, _ in pair_biases]
                indicators.append(pair_indicators)
                energies.append(
                    fold_compound.eval_structure(structure)
                    + sum(
                        value for value, symbol in zip(lambdas, structure)
                        if symbol != "."
                    )
                    + sum(
                        value for present, (_, _, value)
                        in zip(pair_indicators, pair_biases) if present
                    )
                    + state_biases[
                        int(structure[0] != "."),
                        int(structure[8] != "."),
                    ]
                )
            energies = np.asarray(energies)
            reference = np.min(energies)
            kT = _KB * molecule._temperature
            weights = np.exp(-(energies - reference) / kT)
            expected_free_energy = reference - kT * np.log(np.sum(weights))
            np.testing.assert_allclose(
                molecule.evaluate(structures), energies, atol=1e-6
            )
            self.assertAlmostEqual(molecule.mfe()[1], reference, places=6)
            self.assertAlmostEqual(
                molecule.total_free_energy(), expected_free_energy, places=6
            )
            np.testing.assert_allclose(
                molecule.d_free_energy_d_base_pair_biases(),
                weights @ np.asarray(indicators) / np.sum(weights),
                atol=1e-7,
            )

        check()
        self._run_in_vanilla_mode(check)

    def test_centikcal_biases_have_no_rounding_residual(self):
        molecule = Molecule(
            self.seq,
            lambdas1d=[0.35] + [0.0] * (len(self.seq) - 1),
            base_pair_biases=[(0, 8, 0.35)],
        )
        component = molecule._dp_molecules[0]
        self.assertEqual(component._rounding_residuals_range, 0.0)
        self.assertEqual(component._base_pair_biases.residual_by_pair, {})
        self.assertTrue(all(
            log_weight == 0.0
            for _, log_weight in molecule.sample(5, weights=True)
        ))

        mixture = Molecule(
            self.seq,
            state_positions=(0, 8),
            state_biases=[[0.123, -0.234], [0.345, 0.456]],
            reduce_state_space=True,
        )
        self.assertTrue(any(
            abs(log_weight) > 1e-6
            for _, log_weight in mixture.sample(20, weights=True)
        ))

    def test_dense_base_pair_residual_corrections(self):
        sequence = "G" * 8 + "C" * 8
        biases = [
            (i, j, 0.003 * (1 if (i + j) % 2 else -1))
            for i in range(8)
            for j in range(8, 16)
            if j - i > 3
        ]
        prepared = rna2d._prepare_base_pair_biases(sequence, biases)
        self.assertGreater(
            len(prepared.residual_by_pair),
            len(sequence),
        )

        structures = list(_enumerate_secondary_structures(sequence))[:50]
        expected = np.asarray([
            rna2d._correct_base_pair_rounding_energy(
                structure,
                prepared,
            )
            for structure in structures
        ])
        actual = rna2d._correct_base_pair_rounding_energies(
            structures,
            prepared,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-15)

    def test_zero_bias_state_mixture(self):

        reference = Molecule(self.seq)
        mixture = Molecule(
            self.seq,
            state_positions=(0, 1),
        )

        self.assertEqual(len(mixture._dp_molecules), 2)
        self.assertAlmostEqual(
            mixture.total_free_energy(),
            reference.total_free_energy(),
            places=6,
        )
        np.testing.assert_allclose(
            mixture.base_pairing_probability(),
            reference.base_pairing_probability(),
            atol=1e-7,
        )
        self.assertAlmostEqual(
            mixture.mfe()[1],
            reference.mfe()[1],
        )
        self.assertEqual(
            mixture.suboptimal_structures(2.0),
            reference.suboptimal_structures(2.0),
        )
        self.assertAlmostEqual(
            mixture.suboptimal_coverage(2.0),
            reference.suboptimal_coverage(2.0),
            places=6,
        )

    def test_reduced_state_mixture_representation(self):

        positions = (0, 1, 2)
        biases = np.arange(8, dtype=float).reshape((2, 2, 2)) / 10.0
        lambdas = np.arange(len(self.seq), dtype=float) / 100.0
        mixture = Molecule(
            self.seq,
            lambdas1d=lambdas,
            state_positions=positions,
            state_biases=biases,
        )

        self.assertEqual(len(mixture._dp_molecules), 4)
        np.testing.assert_array_equal(mixture._lambdas1d, lambdas)

        for state, component in zip(
            mixture._states,
            mixture._dp_molecules,
        ):
            expected_lambdas = lambdas.copy()
            expected_lambdas[positions[-1]] += (
                biases[state + (1,)] - biases[state + (0,)]
            )
            np.testing.assert_allclose(
                component._lambdas1d,
                expected_lambdas,
            )
            self.assertEqual(
                mixture._component_biases[state],
                biases[state + (0,)],
            )
            self.assertNotIn(
                positions[-1],
                component._force_paired,
            )
            self.assertNotIn(
                positions[-1],
                component._force_unpaired,
            )

        full = Molecule(
            self.seq,
            lambdas1d=lambdas,
            state_positions=positions,
            state_biases=biases,
            reduce_state_space=False,
        )
        self.assertEqual(len(full._dp_molecules), 8)
        self.assertAlmostEqual(
            mixture.total_free_energy(),
            full.total_free_energy(),
            places=6,
        )
        np.testing.assert_allclose(
            mixture.base_pairing_probability(),
            full.base_pairing_probability(),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            mixture.d_free_energy_d_state_biases(),
            full.d_free_energy_d_state_biases(),
            atol=1e-7,
        )

    def test_multiple_state_sets(self):

        positions_a = (0, 1)
        positions_b = (7, 8)
        biases_a = np.array([
            [0.0, 0.2],
            [-0.1, 0.7],
        ])
        biases_b = np.array([
            [0.3, -0.2],
            [0.4, 0.1],
        ])
        lambdas = np.arange(len(self.seq), dtype=float) / 100.0

        separate = Molecule(
            self.seq,
            lambdas1d=lambdas,
            state_positions=(positions_a, positions_b),
            state_biases=(biases_a, biases_b),
        )
        self.assertEqual(len(separate._dp_molecules), 4)
        for state, component in zip(
            separate._states,
            separate._dp_molecules,
        ):
            state_a = (state[0],)
            state_b = (state[1],)
            expected_lambdas = lambdas.copy()
            expected_lambdas[positions_a[-1]] += (
                biases_a[state_a + (1,)]
                - biases_a[state_a + (0,)]
            )
            expected_lambdas[positions_b[-1]] += (
                biases_b[state_b + (1,)]
                - biases_b[state_b + (0,)]
            )
            np.testing.assert_allclose(
                component._lambdas1d,
                expected_lambdas,
            )
            self.assertAlmostEqual(
                separate._component_biases[state],
                biases_a[state_a + (0,)]
                + biases_b[state_b + (0,)],
            )

        separate_full = Molecule(
            self.seq,
            lambdas1d=lambdas,
            state_positions=(positions_a, positions_b),
            state_biases=(biases_a, biases_b),
            reduce_state_space=False,
        )
        self.assertEqual(len(separate_full._dp_molecules), 16)

        joint_biases = (
            biases_a[:, :, np.newaxis, np.newaxis]
            + biases_b[np.newaxis, np.newaxis, :, :]
        )
        joint = Molecule(
            self.seq,
            lambdas1d=lambdas,
            state_positions=positions_a + positions_b,
            state_biases=joint_biases,
        )
        self.assertEqual(len(joint._dp_molecules), 8)

        for reference in (separate_full, joint):
            self.assertAlmostEqual(
                separate.total_free_energy(),
                reference.total_free_energy(),
                places=6,
            )
            np.testing.assert_allclose(
                separate.base_pairing_probability(),
                reference.base_pairing_probability(),
                atol=1e-7,
            )
            self.assertAlmostEqual(
                separate.mfe()[1],
                reference.mfe()[1],
                places=6,
            )

        derivatives = separate.d_free_energy_d_state_biases()
        full_derivatives = (
            separate_full.d_free_energy_d_state_biases()
        )
        self.assertIsInstance(derivatives, list)
        self.assertEqual(len(derivatives), 2)
        for derivative, full_derivative in zip(
            derivatives,
            full_derivatives,
        ):
            self.assertAlmostEqual(np.sum(derivative), 1.0)
            np.testing.assert_allclose(
                derivative,
                full_derivative,
                atol=1e-7,
            )

        joint_derivatives = joint.d_free_energy_d_state_biases()
        np.testing.assert_allclose(
            derivatives[0],
            np.sum(joint_derivatives, axis=(2, 3)),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            derivatives[1],
            np.sum(joint_derivatives, axis=(0, 1)),
            atol=1e-7,
        )

        epsilon = 1e-3
        index = (1, 0)
        biases_a_plus = biases_a.copy()
        biases_a_minus = biases_a.copy()
        biases_a_plus[index] += epsilon
        biases_a_minus[index] -= epsilon
        finite_difference = (
            Molecule(
                self.seq,
                lambdas1d=lambdas,
                state_positions=(positions_a, positions_b),
                state_biases=(biases_a_plus, biases_b),
            ).total_free_energy()
            - Molecule(
                self.seq,
                lambdas1d=lambdas,
                state_positions=(positions_a, positions_b),
                state_biases=(biases_a_minus, biases_b),
            ).total_free_energy()
        ) / (2.0 * epsilon)
        self.assertAlmostEqual(
            derivatives[0][index],
            finite_difference,
            delta=2e-4,
        )

    def test_biased_state_mixture(self):

        position = 0
        biases = np.array([0.0, 0.8])
        mixture = Molecule(
            self.seq,
            state_positions=(position,),
            state_biases=biases,
        )
        components = [
            Molecule(self.seq, force_unpaired=(position,)),
            Molecule(self.seq, force_paired=(position,)),
        ]

        component_free_energies = np.array([
            molecule.total_free_energy()
            for molecule in components
        ])
        biased_free_energies = component_free_energies + biases
        reference = np.min(biased_free_energies)
        relative_weights = np.exp(
            -(biased_free_energies - reference)
            / (_KB * mixture._dp_molecules[0]._temperature)
        )
        probabilities = relative_weights / np.sum(relative_weights)
        expected_free_energy = (
            reference
            - _KB * mixture._dp_molecules[0]._temperature
            * np.log(np.sum(relative_weights))
        )

        self.assertAlmostEqual(
            mixture.total_free_energy(),
            expected_free_energy,
        )

        expected_bpp = sum(
            probability * molecule.base_pairing_probability()
            for probability, molecule in zip(probabilities, components)
        )
        np.testing.assert_allclose(
            mixture.base_pairing_probability(),
            expected_bpp,
            atol=1e-14,
        )

        component_mfes = [
            molecule.mfe()
            for molecule in components
        ]
        expected_mfe_index = np.argmin([
            result[1] + bias
            for result, bias in zip(component_mfes, biases)
        ])
        expected_mfe = component_mfes[expected_mfe_index]
        structure, energy = mixture.mfe()
        self.assertEqual(structure, expected_mfe[0])
        self.assertAlmostEqual(
            energy,
            expected_mfe[1] + biases[expected_mfe_index],
        )

        base_fc = mixture._dp_molecules[0]._make_fold_compound()
        exhaustive = [
            (
                structure,
                float(
                    base_fc.eval_structure(structure)
                    + biases[int(structure[position] != ".")]
                ),
            )
            for structure in _enumerate_secondary_structures(self.seq)
        ]
        exhaustive.sort(key=lambda item: item[1])
        cutoff = exhaustive[0][1] + 2.0
        expected_suboptimal = [
            item
            for item in exhaustive
            if item[1] <= cutoff
        ]
        actual_suboptimal = mixture.suboptimal_structures(2.0)
        self.assertEqual(
            [item[0] for item in actual_suboptimal],
            [item[0] for item in expected_suboptimal],
        )
        np.testing.assert_allclose(
            [item[1] for item in actual_suboptimal],
            [item[1] for item in expected_suboptimal],
            atol=1e-7,
        )

        equivalent_lambdas = np.zeros(len(self.seq))
        equivalent_lambdas[position] = biases[1] - biases[0]
        equivalent = Molecule(
            self.seq,
            lambdas1d=equivalent_lambdas,
        )
        np.testing.assert_allclose(
            mixture.pairing_correlation_matrix(),
            equivalent.pairing_correlation_matrix(),
            atol=1e-7,
        )

        samples = mixture.sample(5000)
        sampled_paired_probability = np.mean([
            structure[position] != "."
            for structure in samples
        ])
        self.assertAlmostEqual(
            sampled_paired_probability,
            probabilities[1],
            delta=0.03,
        )

        weighted_samples = mixture.sample(20, weights=True)
        self.assertTrue(all(
            abs(log_weight) < 1e-12
            for _, log_weight in weighted_samples
        ))
        for component in mixture._dp_molecules:
            self.assertAlmostEqual(
                component.sample_rounding_correction(),
                0.0,
            )

    def test_copy_semantics(self):

        mol = Molecule(self.seq)

        bpp = mol.base_pairing_probability()
        bpp[:] = 0.0

        bpp2 = mol.base_pairing_probability()

        self.assertGreater(np.sum(bpp2), 0.0)

    def test_suboptimal(self):

        mol = Molecule(self.seq)

        s = mol.suboptimal_structures(2.0)

        self.assertGreater(len(s), 0)

        energies = [x[1] for x in s]

        self.assertEqual(energies, sorted(energies))

        self.assertAlmostEqual(
            energies[0],
            mol.mfe()[1],
        )

    def test_single_state_energy_offset(self):
        baseline = Molecule(self.seq).suboptimal_structures(0.5)
        shifted = Molecule(
            self.seq,
            state_positions=(),
            state_biases=0.5,
            reduce_state_space=False,
        ).suboptimal_structures(0.5)
        self.assertEqual([structure for structure, _ in shifted],
                         [structure for structure, _ in baseline])
        np.testing.assert_allclose(
            [energy for _, energy in shifted],
            [energy + 0.5 for _, energy in baseline],
        )

    def test_suboptimal_vanilla(self):
        self._run_in_vanilla_mode(self.test_suboptimal)

    def test_suboptimal_structures_with_negative_soft_constraint(self):

        import bussilab

        seq = "CGACGUACCGUUUUGCAAAGGCGUGGCGGCCCCCAUGAACAUUGACCGUCACUGUUUCCACGUAUGUUCU"

        lambdas = np.zeros(len(seq))
        lambdas[11] = +0.01

        baseline = {
            s[0]: s[1]
            for s in Molecule(seq).suboptimal_structures(1.3)
        }

        expected = {
            structure: energy + (0.01 if structure[11] != "." else 0.0)
            for structure, energy in baseline.items()
        }
        best = min(expected.values())
        expected = {
            structure: energy
            for structure, energy in expected.items()
            if energy - best <= 1.0001
        }

        subopt = Molecule(seq, lambdas1d=lambdas).suboptimal_structures(1.0001)

        structures = [structure for structure, _ in subopt]
        self.assertEqual(len(structures), len(set(structures)))
        self.assertEqual(set(structures), set(expected))

        for structure, energy in subopt:
            self.assertAlmostEqual(energy, expected[structure], places=3)

    def test_sampling(self):

        np.random.seed(1977)

        mol = Molecule(self.seq)

        samples = mol.sample(100)

        self.assertEqual(len(samples), 100)

        for structure in samples:
            self.assertEqual(len(structure), len(self.seq))
            self.assertIsInstance(structure, str)

        weighted_samples = mol.sample(100, weights=True)

        self.assertEqual(len(weighted_samples), 100)

        for structure, logw in weighted_samples:
            self.assertEqual(len(structure), len(self.seq))
            self.assertIsInstance(logw, float)

        with self.assertRaises(ValueError):
            mol.sample(1, weights="yes")

    def test_set_random_seed(self):
        for invalid in (-1, 2**32, 1.5, True, None, "123"):
            with self.subTest(seed=invalid), self.assertRaises(ValueError):
                rna2d.set_random_seed(invalid)

        molecule = Molecule(
            self.seq,
            lambdas1d=np.full(len(self.seq), 0.003),
            state_positions=(0, 8),
            state_biases=[[0.0, 0.2], [-0.1, 0.3]],
        )
        global_state = np.random.get_state()
        for weights in (False, True):
            rna2d.set_random_seed(123)
            first = molecule.sample(40, weights=weights)
            rna2d.set_random_seed(123)
            second = molecule.sample(40, weights=weights)
            self.assertEqual(first, second)

        after = np.random.get_state()
        self.assertEqual(after[0], global_state[0])
        np.testing.assert_array_equal(after[1], global_state[1])
        self.assertEqual(after[2:], global_state[2:])

    def test_sampling_vanilla(self):
        self._run_in_vanilla_mode(self.test_sampling)

    def test_sampling_zero_residuals(self):

        lam = np.zeros(len(self.seq))
        lam[0] = 0.01
        lam[3] = -0.02

        mol = Molecule(self.seq, lambdas1d=lam)

        samples = mol.sample(20, weights=True)

        for _, logw in samples:
            self.assertAlmostEqual(logw, 0.0)

    def test_nonzero_rounding_residual_rescoring(self):
        sequence = "GCGCGCGC"
        lambdas = np.array([
            0.004,
            -0.006,
            0.013,
            -0.017,
            0.021,
            -0.009,
            0.007,
            -0.012,
        ])
        molecule = Molecule(sequence, lambdas1d=lambdas)
        dp_molecule = molecule._dp_molecules[0]
        self.assertGreater(dp_molecule._lambdas1d_residuals_range, 0.0)

        base_fc = dp_molecule._make_fold_compound()
        exact_energies = {
            structure: float(
                base_fc.eval_structure(structure)
                + sum(
                    penalty
                    for penalty, symbol in zip(lambdas, structure)
                    if symbol != "."
                )
            )
            for structure in _enumerate_secondary_structures(sequence)
        }

        mfe_structure, mfe_energy = molecule.mfe()
        self.assertAlmostEqual(
            mfe_energy,
            min(exact_energies.values()),
            places=6,
        )
        self.assertAlmostEqual(
            mfe_energy,
            exact_energies[mfe_structure],
            places=6,
        )

        delta = 1.0
        cutoff = min(exact_energies.values()) + delta
        expected_suboptimal = {
            structure: energy
            for structure, energy in exact_energies.items()
            if energy <= cutoff
        }
        actual_suboptimal = dict(
            molecule.suboptimal_structures(delta)
        )
        self.assertEqual(
            set(actual_suboptimal),
            set(expected_suboptimal),
        )
        for structure, energy in actual_suboptimal.items():
            self.assertAlmostEqual(
                energy,
                expected_suboptimal[structure],
                places=6,
            )

        np.random.seed(1977)
        weighted_samples = molecule.sample(100, weights=True)
        inverse_kT = 1.0 / (_KB * dp_molecule._temperature)
        self.assertTrue(any(log_weight != 0.0
                            for _, log_weight in weighted_samples))
        for structure, log_weight in weighted_samples:
            expected_log_weight = -sum(
                residual
                for residual, symbol in zip(
                    dp_molecule._lambdas1d_residuals,
                    structure,
                )
                if symbol != "."
            ) * inverse_kT
            self.assertAlmostEqual(log_weight, expected_log_weight)

    def test_batched_rounding_residual_corrections(self):
        structures = list(_enumerate_secondary_structures("GCGCGCGC"))
        residuals = np.array([
            0.004,
            0.004,
            0.003,
            0.003,
            0.001,
            0.001,
            -0.003,
            -0.002,
        ])
        expected = np.array([
            rna2d._correct_rounding_energy(structure, residuals)
            for structure in structures
        ])

        # Force several small chunks so the test covers batch boundaries.
        original_batch_bytes = (
            rna2d._RESIDUAL_CORRECTION_BATCH_BYTES
        )
        rna2d._RESIDUAL_CORRECTION_BATCH_BYTES = 2 * len(residuals)
        try:
            actual = rna2d._correct_rounding_energies(
                structures,
                residuals,
            )
        finally:
            rna2d._RESIDUAL_CORRECTION_BATCH_BYTES = (
                original_batch_bytes
            )

        np.testing.assert_allclose(actual, expected, atol=1e-15)

    def test_evaluate_mixture_with_rounding_residuals(self):
        sequence = "GCGCGCGC"
        lambdas = np.array([
            0.004,
            -0.006,
            0.013,
            -0.017,
            0.021,
            -0.009,
            0.007,
            -0.012,
        ])
        positions = (0, 7)
        biases = np.array([
            [0.123, -0.234],
            [0.345, 0.456],
        ])
        molecules = [
            Molecule(
                sequence,
                lambdas1d=lambdas,
                state_positions=positions,
                state_biases=biases,
                reduce_state_space=reduce_state_space,
            )
            for reduce_state_space in (True, False)
        ]

        base_fc = Molecule(
            sequence
        )._dp_molecules[0]._make_fold_compound()
        exact_energies = {
            structure: float(
                base_fc.eval_structure(structure)
                + sum(
                    penalty
                    for penalty, symbol in zip(lambdas, structure)
                    if symbol != "."
                )
                + biases[
                    int(structure[positions[0]] != "."),
                    int(structure[positions[1]] != "."),
                ]
            )
            for structure in _enumerate_secondary_structures(sequence)
        }

        delta = (
            max(exact_energies.values())
            - min(exact_energies.values())
            + 0.01
        )
        for molecule in molecules:
            for structure, expected_energy in exact_energies.items():
                self.assertAlmostEqual(
                    molecule.evaluate(structure),
                    expected_energy,
                    places=6,
                )

            suboptimal = molecule.suboptimal_structures(delta)
            self.assertEqual(
                {structure for structure, _ in suboptimal},
                set(exact_energies),
            )
            for structure, energy in suboptimal:
                self.assertAlmostEqual(
                    molecule.evaluate(structure),
                    energy,
                    places=6,
                )
            np.testing.assert_allclose(
                molecule.evaluate(suboptimal),
                [energy for _, energy in suboptimal],
                atol=1e-6,
            )

            weighted_samples = molecule.sample(20, weights=True)
            np.testing.assert_allclose(
                molecule.evaluate(weighted_samples),
                [
                    exact_energies[structure]
                    for structure, _ in weighted_samples
                ],
                atol=1e-6,
            )

    def test_internal_counters(self):

        n = len(self.seq)

        mol = Molecule(
            self.seq,
            lambdas1d=np.zeros(n)
        )

        mol.total_free_energy()
        dp_mol = mol._dp_molecules[0]

        self.assertEqual(dp_mol._fc_n_1d_constraints, 0)
        self.assertEqual(dp_mol._fc_n_2d_constraints, 0)

        mol = Molecule(
            self.seq,
            lambdas1d=-np.ones(n)
        )

        mol.total_free_energy()
        dp_mol = mol._dp_molecules[0]

        from bussilab.rna2d import _SUPPORTS_SUBOPT_SOFT_CONSTRAINTS

        if _SUPPORTS_SUBOPT_SOFT_CONSTRAINTS:
            self.assertEqual(dp_mol._fc_n_1d_constraints, n)
            self.assertEqual(dp_mol._fc_n_2d_constraints, 0)

        mol = Molecule(
            self.seq,
            lambdas1d=np.full(n, 100.0)
        )

        mol.total_free_energy()
        dp_mol = mol._dp_molecules[0]

        self.assertEqual(dp_mol._fc_n_1d_constraints, 0)
        from bussilab.rna2d import _ALLOWED_PAIRS

        expected_pairs = sum(
            self.seq[i] + self.seq[j] in _ALLOWED_PAIRS
            for i in range(n)
            for j in range(i + 1, n)
        )
        self.assertEqual(dp_mol._fc_n_2d_constraints, expected_pairs)
        self.assertLess(expected_pairs, n * (n - 1) // 2)

    def test_sparse_2d_constraint_enumeration(self):
        class FakeFoldCompound:
            def __init__(self):
                self.bp = []

            def sc_add_up(self, i, energy):
                raise AssertionError("no 1D constraint expected")

            def sc_add_bp(self, i, j, energy):
                self.bp.append((i, j, energy))

        class CountingPairs:
            def __init__(self, pairs):
                self.pairs = pairs
                self.lookups = 0

            def __contains__(self, pair):
                self.lookups += 1
                return pair in self.pairs

        sequence = "AUAUAUAUAGAGCAUAUAUAU"
        position = len(sequence) // 2
        lambdas = np.zeros(len(sequence))
        lambdas[position] = 100.0
        fc = FakeFoldCompound()

        allowed_pairs = rna2d._ALLOWED_PAIRS
        counting_pairs = CountingPairs(allowed_pairs)
        rna2d._ALLOWED_PAIRS = counting_pairs
        try:
            shift, n_1d, n_2d = rna2d._apply_constraint(
                fc,
                sequence,
                lambdas,
                _KB * 300.0,
            )
        finally:
            rna2d._ALLOWED_PAIRS = allowed_pairs

        expected = [
            (i + 1, j + 1, 100.0)
            for i in range(len(sequence))
            for j in range(i + 1, len(sequence))
            if position in (i, j)
            and sequence[i] + sequence[j] in allowed_pairs
        ]
        self.assertEqual(shift, 0.0)
        self.assertEqual(n_1d, 0)
        self.assertEqual(n_2d, len(expected))
        self.assertEqual(fc.bp, expected)
        self.assertEqual(counting_pairs.lookups, len(sequence) - 1)

    def test_importance_sampling(self):

        seq = "UGCGCCAACUUUGUAGACUCCGCAGAUUACGAACGCCAACGAACGAACAGACACCCUUCCUAGCCUCGCCACUACAUAUGUCUAACAGUCUCUUGUGCUG"

        seq_rev = seq.translate(
            str.maketrans("ACGU", "UGCA")
        )[::-1]

        full_seq = (
            seq
            + "AAAA"
            + seq_rev
            + "AAAA"
            + seq
        )

        lam = np.zeros(len(full_seq))

        lam[:len(seq):2] += 0.01
        lam[-len(seq)::2] += 0.01

        lam[:len(seq)] += 0.004
        lam[-len(seq):] -= 0.004

        mol_base = Molecule(full_seq)

        mol = Molecule(full_seq, lambdas1d=lam)

        bpp_base = np.sum(
            mol_base.base_pairing_probability(),
            axis=1
        )

        bpp = np.sum(
            mol.base_pairing_probability(),
            axis=1
        )

        reference_base = np.average(bpp_base[:len(seq)])

        reference = np.average(bpp[:len(seq)])

        self.assertGreater(reference_base - reference, 0.01)

        numerator = 0.0
        denominator = 0.0

        for structure, logw in mol.sample(100000, weights=True):
            w = np.exp(logw)

            denominator += w

            numerator += (
                sum(c != "." for c in structure[:len(seq)])
                / len(seq)
            ) * w

        estimate = numerator / denominator

        self.assertGreater(
            np.abs(estimate-reference_base),
            np.abs(estimate-reference)
        )

        unweighted_samples = mol.sample(50000)
        unweighted_estimate = np.mean([
            sum(c != "." for c in structure[:len(seq)])
            / len(seq)
            for structure in unweighted_samples
        ])
        self.assertAlmostEqual(
            unweighted_estimate,
            reference,
            delta=0.005,
        )

    def test_importance_sampling_vanilla(self):
        self._run_in_vanilla_mode(self.test_importance_sampling)

    def test_pairing_correlation_matrix_exhaustive(self):
        seq = "GCGCGCGC"
        mol = Molecule(seq)
        dp_mol = mol._dp_molecules[0]
        dp_mol._ensure_pf()

        structures = list(_enumerate_secondary_structures(seq))
        energies = np.array([
            dp_mol._fc.eval_structure(structure)
            for structure in structures
        ])
        weights = np.exp(
            -(energies - np.min(energies))
            / (1.98717 / 1000 * dp_mol._temperature)
        )
        weights /= np.sum(weights)

        expected = np.zeros((len(seq), len(seq)))
        for structure, weight in zip(structures, weights):
            paired = np.array([base != "." for base in structure])
            expected[np.ix_(paired, paired)] += weight

        actual = mol.pairing_correlation_matrix()

        np.testing.assert_allclose(actual, expected, atol=1e-7)
        np.testing.assert_allclose(actual, actual.T, atol=1e-14)

    def test_pairing_correlation_matrix_exhaustive_with_soft_constraints(self):
        seq = "GCGCGCGC"
        lambdas = np.array(
            [0.004, -0.006, 0.013, -0.017, 0.021, -0.009, 0.007, -0.012]
        )
        mol = Molecule(seq, lambdas1d=lambdas)
        dp_mol = mol._dp_molecules[0]
        dp_mol._ensure_pf()
        base_fc = dp_mol._make_fold_compound()

        structures = list(_enumerate_secondary_structures(seq))
        energies = np.array([
            base_fc.eval_structure(structure)
            + sum(
                penalty
                for penalty, base in zip(lambdas, structure)
                if base != "."
            )
            for structure in structures
        ])
        weights = np.exp(
            -(energies - np.min(energies)) / (_KB * dp_mol._temperature)
        )
        weights /= np.sum(weights)

        expected = np.zeros((len(seq), len(seq)))
        for structure, weight in zip(structures, weights):
            paired = np.array([base != "." for base in structure])
            expected[np.ix_(paired, paired)] += weight

        actual = mol.pairing_correlation_matrix()

        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_pairing_correlation_matrix_with_hard_constraints(self):
        seq = "GCGCGCGC"
        paired = 0
        unpaired = 1
        mol = Molecule(
            seq,
            force_paired=(paired,),
            force_unpaired=(unpaired,),
        )
        dp_mol = mol._dp_molecules[0]
        base_fc = dp_mol._make_fold_compound()

        structures = [
            structure
            for structure in _enumerate_secondary_structures(seq)
            if structure[paired] != "." and structure[unpaired] == "."
        ]
        energies = np.array([
            base_fc.eval_structure(structure)
            for structure in structures
        ])
        weights = np.exp(
            -(energies - np.min(energies))
            / (_KB * dp_mol._temperature)
        )
        weights /= np.sum(weights)

        expected = np.zeros((len(seq), len(seq)))
        for structure, weight in zip(structures, weights):
            is_paired = np.array([base != "." for base in structure])
            expected[np.ix_(is_paired, is_paired)] += weight

        bpp_before = mol.base_pairing_probability()
        actual = mol.pairing_correlation_matrix()
        bpp_after = mol.base_pairing_probability()

        np.testing.assert_allclose(actual, expected, atol=1e-7)
        np.testing.assert_allclose(bpp_after, bpp_before, atol=1e-14)

    def test_pairing_correlation_matrix_state_mixture(self):
        seq = "GCGCGCGC"
        positions = (0, 1)
        biases = np.array([
            [0.0, 0.2],
            [-0.1, 0.7],
        ])
        mol = Molecule(
            seq,
            state_positions=positions,
            state_biases=biases,
        )
        base_fc = mol._dp_molecules[0]._make_fold_compound()

        structures = list(_enumerate_secondary_structures(seq))
        energies = np.array([
            base_fc.eval_structure(structure)
            + biases[
                int(structure[positions[0]] != "."),
                int(structure[positions[1]] != "."),
            ]
            for structure in structures
        ])
        weights = np.exp(
            -(energies - np.min(energies))
            / (_KB * mol._temperature)
        )
        weights /= np.sum(weights)

        expected = np.zeros((len(seq), len(seq)))
        for structure, weight in zip(structures, weights):
            is_paired = np.array([base != "." for base in structure])
            expected[np.ix_(is_paired, is_paired)] += weight

        exact = mol.pairing_correlation_matrix()
        np.testing.assert_allclose(exact, expected, atol=1e-7)

        delta = float(np.max(energies) - np.min(energies) + 0.01)
        self.assertAlmostEqual(
            mol.suboptimal_coverage(delta),
            1.0,
            places=6,
        )

    def test_suboptimal_coverage(self):
        mol = Molecule("GCGCGCGC")

        self.assertAlmostEqual(mol.suboptimal_coverage(5.0), 1.0, places=6)

    def test_partition_function_matches_evaluated_structures_with_constraints(
        self
    ):
        seq = "GCGCGCGC"
        structures = list(_enumerate_secondary_structures(seq))
        state_biases = np.array([[0.0, 0.2], [-0.1, 0.7]])
        cases = (
            {"force_paired": (0,), "force_unpaired": (1,)},
            {"state_positions": (0, 1), "state_biases": state_biases},
            {
                "force_paired": (0,), "force_unpaired": (1,),
                "state_positions": (2, 3), "state_biases": state_biases,
            },
        )
        for options in cases:
            for reduce_state_space in (False, True):
                with self.subTest(
                    options=options, reduce_state_space=reduce_state_space
                ):
                    mol = Molecule(
                        seq, no_lonely_pair=False, pf_smooth=False,
                        reduce_state_space=reduce_state_space, **options
                    )
                    energies = []
                    for structure in structures:
                        energies.append(mol.evaluate(structure))
                    energies = np.asarray(energies)
                    self.assertTrue(np.isfinite(energies).any())
                    kT = _KB * mol.temperature
                    expected = -kT * np.logaddexp.reduce(
                        -energies / kT
                    )
                    self.assertAlmostEqual(
                        mol.total_free_energy(), expected, delta=1e-6
                    )

    def test_suboptimal_coverage_with_soft_constraints(self):
        lambdas = np.array(
            [0.004, -0.006, 0.013, -0.017, 0.021, -0.009, 0.007, -0.012]
        )
        mol = Molecule("GCGCGCGC", lambdas1d=lambdas)

        self.assertAlmostEqual(mol.suboptimal_coverage(5.0), 1.0, places=6)

    def test_pairing_correlation_matrix_restores_unconstrained_pf(self):
        mol = Molecule("GCGCGCGC")
        before = mol.base_pairing_probability()

        mol.pairing_correlation_matrix()

        after = mol.base_pairing_probability()
        np.testing.assert_allclose(after, before, atol=1e-12)

    def test_pairing_correlation_matrix_restores_soft_constrained_pf(self):
        lambdas = np.array(
            [0.004, -0.006, 0.013, -0.017, 0.021, -0.009, 0.007, -0.012]
        )
        mol = Molecule("GCGCGCGC", lambdas1d=lambdas)
        before = mol.base_pairing_probability()

        mol.pairing_correlation_matrix()

        after = mol.base_pairing_probability()
        np.testing.assert_allclose(after, before, atol=1e-12)

if __name__ == "__main__":
    unittest.main()
