import unittest
import numpy as np

try:
    import RNA
    _HAS_VIENNA = True
except ImportError:
    _HAS_VIENNA = False

if _HAS_VIENNA:
    import bussilab.rna2d as rna2d
    from bussilab.rna2d import Molecule

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

if __name__ == "__main__":
    unittest.main()
