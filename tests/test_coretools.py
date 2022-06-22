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

    def test_results2(self):
        class Result(coretools.Result):
            def __init__(self,*,val1,val2):
                self.val2=val2
                self.val1=val1
        r=Result(val1=1,val2=2)
        print(r.val1)
        print(r["val1"])
        print(r.val2)
        print(r["val2"])
        self.assertEqual(str(r)," val1: 1\n val2: 2")
        self.assertEqual(dir(r),["val1","val2"])

        del r.val1
        self.assertEqual(str(r)," val2: 2")
        self.assertEqual(dir(r),["val2"])

        with self.assertRaises(KeyError):
            del r.val1

        with self.assertRaises(AttributeError):
            print(r.attribute)
        with self.assertRaises(KeyError):
            print(r['attribute'])

        class Empty(coretools.Result):
            pass
        e=Empty()
        self.assertEqual(str(e),"Empty()")

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

        self.assertIsInstance(bussilab.list_submodules(),list)

    def test_cd(self):
        import bussilab
        import os
        class test_raise(Exception):
            pass
        with coretools.cd(os.path.dirname(os.path.abspath(__file__))):
            try:
                with coretools.cd("test_dir",create=True):
                    with open("test_file","w") as f:
                        print("content",file=f)
                raise test_raise # test if directory is changed back
            except test_raise:
                pass
            with open("test_dir/test_file") as f:
                for l in f:
                    self.assertEqual(l,"content\n")
            os.remove("test_dir/test_file")
            os.rmdir("test_dir")

    def test_cd2(self):
        import bussilab
        import os
        class test_raise(Exception):
            pass
        with coretools.cd(os.path.dirname(os.path.abspath(__file__))):
            try:
                with coretools.cd("test_dir/subdir",create=True):
                    with open("test_file","w") as f:
                        print("content",file=f)
                raise test_raise # test if directory is changed back
            except test_raise:
                pass
            with open("test_dir/subdir/test_file") as f:
                for l in f:
                    self.assertEqual(l,"content\n")
            os.remove("test_dir/subdir/test_file")
            os.rmdir("test_dir/subdir")
            os.rmdir("test_dir")

if __name__ == "__main__":
    unittest.main()
