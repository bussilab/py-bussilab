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
            rna2d.sample_to_numpy(["[....]"])
        with self.assertRaises(ValueError):
            rna2d.sample_to_numpy(["......"], deduplicate="yes")

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
            self.assertTrue(inherited._no_lonely_pair)
            self.assertTrue(inherited._pf_smooth)
            self.assertEqual(inherited._salt, 2.0)
            self.assertEqual(inherited._parameters, "turner1999")

            # Existing molecules retain the defaults captured at construction.
            self.assertEqual(
                original._temperature,
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

    def test_mfe(self):

        mol = Molecule(self.seq)

        structure, energy = mol.mfe()

        self.assertEqual(len(structure), len(self.seq))
        self.assertIsInstance(structure, str)
        self.assertIsInstance(energy, float)

    def test_evaluate(self):
        molecule = Molecule(self.seq)
        for structure, energy in molecule.suboptimal_structures(5.0):
            self.assertAlmostEqual(
                molecule.evaluate(structure),
                energy,
            )

        with self.assertRaises(ValueError):
            molecule.evaluate(None)
        with self.assertRaises(ValueError):
            molecule.evaluate("." * (len(self.seq) - 1))
        with self.assertRaises(ValueError):
            molecule.evaluate("x" * len(self.seq))

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
        #self.assertEqual(set(structures), set(expected))

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
