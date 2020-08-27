import unittest

import bussilab.coretools as coretools

class Test(unittest.TestCase):
    def test_ensure_np_array(self):
        import numpy as np

        # None
        A = None
        B = coretools.ensure_np_array(A)
        self.assertIsNone(B)

        # Tuple
        A = (1.0, 2.0, 3.0)
        B = coretools.ensure_np_array(A)
        self.assertIsInstance(B, np.ndarray)
        self.assertSequenceEqual(list(B), list(A))
        # Check that it is a deep copy
        B[1] = 10.0
        self.assertSequenceEqual(A, [1, 2, 3])

        # List
        A = [1.0, 2.0, 3.0]
        B = coretools.ensure_np_array(A)
        self.assertIsInstance(B, np.ndarray)
        self.assertSequenceEqual(list(B), list(A))
        # Check that it is a deep copy
        B[1] = 10.0
        self.assertSequenceEqual(A, [1, 2, 3])

        # Np array
        A = np.array([1.0, 2.0, 3.0])
        B = coretools.ensure_np_array(A)
        self.assertIsInstance(B, np.ndarray)
        self.assertSequenceEqual(list(B), list(A))
        # Check that it is a shallow copy
        B[1] = 10.0
        self.assertSequenceEqual(list(A), list(B))

        # Rank2 tuple
        A = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
        B = coretools.ensure_np_array(A)
        self.assertIsInstance(B, np.ndarray)
        self.assertAlmostEqual((B[0, 0]-1)**2 + (B[0, 1]-2)**2 + (B[0, 2]-3)**2, 0.0)
        self.assertAlmostEqual((B[1, 0]-4)**2 + (B[1, 1]-5)**2 + (B[1, 2]-6)**2, 0.0)
        # Check that it is a deep copy
        B[1, 1] = 10.0
        self.assertAlmostEqual((A[0][0]-1)**2 + (A[0][1]-2)**2 + (A[0][2]-3)**2, 0.0)
        self.assertAlmostEqual((A[1][0]-4)**2 + (A[1][1]-5)**2 + (A[1][2]-6)**2, 0.0)

        # Rank2 list
        A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        B = coretools.ensure_np_array(A)
        self.assertIsInstance(B, np.ndarray)
        self.assertAlmostEqual((B[0, 0]-1)**2 + (B[0, 1]-2)**2 + (B[0, 2]-3)**2, 0.0)
        self.assertAlmostEqual((B[1, 0]-4)**2 + (B[1, 1]-5)**2 + (B[1, 2]-6)**2, 0.0)
        # Check that it is a deep copy
        B[1, 1] = 10.0
        self.assertAlmostEqual((A[0][0]-1)**2 + (A[0][1]-2)**2 + (A[0][2]-3)**2, 0.0)
        self.assertAlmostEqual((A[1][0]-4)**2 + (A[1][1]-5)**2 + (A[1][2]-6)**2, 0.0)

        # Tuple of np arrays
        A = (np.array((1.0, 2.0, 3.0)), np.array((4.0, 5.0, 6.0)))
        B = coretools.ensure_np_array(A)
        self.assertIsInstance(B, np.ndarray)
        self.assertAlmostEqual((B[0, 0]-1)**2 + (B[0, 1]-2)**2 + (B[0, 2]-3)**2, 0.0)
        self.assertAlmostEqual((B[1, 0]-4)**2 + (B[1, 1]-5)**2 + (B[1, 2]-6)**2, 0.0)
        # Check that it is a deep copy
        B[1, 1] = 10.0
        self.assertAlmostEqual((A[0][0]-1)**2 + (A[0][1]-2)**2 + (A[0][2]-3)**2, 0.0)
        self.assertAlmostEqual((A[1][0]-4)**2 + (A[1][1]-5)**2 + (A[1][2]-6)**2, 0.0)

        # List of np arrays
        A = [np.array((1.0, 2.0, 3.0)), np.array((4.0, 5.0, 6.0))]
        B = coretools.ensure_np_array(A)
        self.assertIsInstance(B, np.ndarray)
        self.assertAlmostEqual((B[0, 0]-1)**2 + (B[0, 1]-2)**2 + (B[0, 2]-3)**2, 0.0)
        self.assertAlmostEqual((B[1, 0]-4)**2 + (B[1, 1]-5)**2 + (B[1, 2]-6)**2, 0.0)
        # Check that it is a deep copy
        B[1, 1] = 10.0
        self.assertAlmostEqual((A[0][0]-1)**2 + (A[0][1]-2)**2 + (A[0][2]-3)**2, 0.0)
        self.assertAlmostEqual((A[1][0]-4)**2 + (A[1][1]-5)**2 + (A[1][2]-6)**2, 0.0)

        # Rank2 np array
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        B = coretools.ensure_np_array(A)
        self.assertIsInstance(B, np.ndarray)
        self.assertAlmostEqual((B[0, 0]-1)**2 + (B[0, 1]-2)**2 + (B[0, 2]-3)**2, 0.0)
        self.assertAlmostEqual((B[1, 0]-4)**2 + (B[1, 1]-5)**2 + (B[1, 2]-6)**2, 0.0)
        # Check that it is a shallow copy
        B[1, 1] = 10.0
        self.assertAlmostEqual((A[0][0]-1)**2 + (A[0][1]-2)**2 + (A[0][2]-3)**2, 0.0)
        self.assertAlmostEqual((A[1][0]-4)**2 + (A[1][1]-10)**2 + (A[1][2]-6)**2, 0.0)

    def test_results(self):

        class Result(coretools.Result):
            pass

        res = Result(a=1, b="test")
        self.assertEqual(res.a, 1)
        self.assertEqual(res.b, "test")
        self.assertEqual(str(res), " a: 1\n b: 'test'")

        res2 = Result(a=2, b="test2", c=res)
        self.assertEqual(res2.a, 2)
        self.assertEqual(res2.b, "test2")
        self.assertEqual(str(res2), " a: 2\n b: 'test2'\n c:  a: 1\n     b: 'test'")
        self.assertEqual(res2.c.a, 1)
        self.assertEqual(res2.c.b, "test")
        self.assertEqual(str(res2.c), " a: 1\n b: 'test'")

    def test_submodules(self):
        import bussilab
        with self.assertRaises(ModuleNotFoundError):
            import bussilab.not_existing_module
        with self.assertRaises(AttributeError):
            m = bussilab.not_existing_module
        self.assertIsInstance(bussilab.describe_submodule(""),str)
        self.assertIsInstance(bussilab.describe_submodule("coretools"),str)
        with self.assertRaises(ModuleNotFoundError):
            bussilab.describe_submodule("not_existing_module")



if __name__ == "__main__":
    unittest.main()
