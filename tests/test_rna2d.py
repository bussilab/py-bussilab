import unittest
import numpy as np

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

    def test_parameter_sets(self):

        for p in (
            "turner1999",
            "turner2004",
            "andronescu2007",
            "langdon2018",
        ):
            Molecule(self.seq, parameters=p).mfe()

    def test_mfe(self):

        mol = Molecule(self.seq)

        structure, energy = mol.mfe()

        self.assertEqual(len(structure), len(self.seq))
        self.assertIsInstance(structure, str)
        self.assertIsInstance(energy, float)

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

        for structure, logw in samples:
            self.assertEqual(len(structure), len(self.seq))
            self.assertIsInstance(logw, float)

    def test_sampling_vanilla(self):
        self._run_in_vanilla_mode(self.test_sampling)

    def test_sampling_zero_residuals(self):

        lam = np.zeros(len(self.seq))
        lam[0] = 0.01
        lam[3] = -0.02

        mol = Molecule(self.seq, lambdas1d=lam)

        samples = mol.sample(20)

        for _, logw in samples:
            self.assertAlmostEqual(logw, 0.0)

    def test_internal_counters(self):

        n = len(self.seq)

        mol = Molecule(
            self.seq,
            lambdas1d=np.zeros(n)
        )

        mol.total_free_energy()

        self.assertEqual(mol._fc_n_1d_constraints, 0)
        self.assertEqual(mol._fc_n_2d_constraints, 0)

        mol = Molecule(
            self.seq,
            lambdas1d=-np.ones(n)
        )

        mol.total_free_energy()

        from bussilab.rna2d import _SUPPORTS_SUBOPT_SOFT_CONSTRAINTS

        if _SUPPORTS_SUBOPT_SOFT_CONSTRAINTS:
            self.assertEqual(mol._fc_n_1d_constraints, n)
            self.assertEqual(mol._fc_n_2d_constraints, 0)

        mol = Molecule(
            self.seq,
            lambdas1d=np.full(n, 100.0)
        )

        mol.total_free_energy()

        self.assertEqual(mol._fc_n_1d_constraints, 0)
        self.assertGreater(mol._fc_n_2d_constraints, 0)

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

        for structure, logw in mol.sample(100000):
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

    def test_importance_sampling_vanilla(self):
        self._run_in_vanilla_mode(self.test_importance_sampling)

    def test_pairing_correlation_matrix_exhaustive(self):
        seq = "GCGCGCGC"
        mol = Molecule(seq)
        mol._ensure_pf()

        structures = list(_enumerate_secondary_structures(seq))
        energies = np.array([
            mol._fc.eval_structure(structure)
            for structure in structures
        ])
        weights = np.exp(
            -(energies - np.min(energies))
            / (1.98717 / 1000 * mol._temperature)
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
        mol._ensure_pf()
        base_fc = mol._make_fold_compound()

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
            -(energies - np.min(energies)) / (_KB * mol._temperature)
        )
        weights /= np.sum(weights)

        expected = np.zeros((len(seq), len(seq)))
        for structure, weight in zip(structures, weights):
            paired = np.array([base != "." for base in structure])
            expected[np.ix_(paired, paired)] += weight

        actual = mol.pairing_correlation_matrix()

        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_pairing_correlation_matrix_suboptimal_and_coverage(self):
        mol = Molecule("GCGCGCGC")

        exhaustive = mol.pairing_correlation_matrix(suboptimal_delta=5.0)
        exact = mol.pairing_correlation_matrix()

        np.testing.assert_allclose(exhaustive, exact, atol=1e-7)
        self.assertAlmostEqual(mol.suboptimal_coverage(5.0), 1.0, places=6)

    def test_pairing_correlation_matrix_suboptimal_with_soft_constraints(self):
        lambdas = np.array(
            [0.004, -0.006, 0.013, -0.017, 0.021, -0.009, 0.007, -0.012]
        )
        mol = Molecule("GCGCGCGC", lambdas1d=lambdas)

        exhaustive = mol.pairing_correlation_matrix(suboptimal_delta=5.0)
        exact = mol.pairing_correlation_matrix()

        np.testing.assert_allclose(exhaustive, exact, atol=1e-7)
        self.assertAlmostEqual(mol.suboptimal_coverage(5.0), 1.0, places=6)

    def test_pairing_correlation_matrix_sampling(self):
        mol = Molecule("GCGCGCGC")
        exact = mol.pairing_correlation_matrix()
        sampled = mol.pairing_correlation_matrix(samples=20000)

        np.testing.assert_allclose(sampled, exact, atol=0.03)

    def test_pairing_correlation_matrix_sampling_with_soft_constraints(self):
        lambdas = np.array(
            [0.004, -0.006, 0.013, -0.017, 0.021, -0.009, 0.007, -0.012]
        )
        mol = Molecule("GCGCGCGC", lambdas1d=lambdas)
        exact = mol.pairing_correlation_matrix()
        sampled = mol.pairing_correlation_matrix(samples=20000)

        np.testing.assert_allclose(sampled, exact, atol=0.03)

    def test_pairing_correlation_matrix_stable_log_weights(self):
        mol = Molecule("GC")
        probability = np.exp(1.0) / (1.0 + np.exp(1.0))

        matrix = mol._pairing_correlation_matrix_from_iterable(
            [("..", 1000.0), ("()", 1001.0)]
        )

        np.testing.assert_allclose(
            matrix,
            np.full((2, 2), probability),
        )

        matrix = mol._pairing_correlation_matrix_from_iterable(
            [("..", -1000.0), ("()", -999.0)]
        )

        np.testing.assert_allclose(
            matrix,
            np.full((2, 2), probability),
        )

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
