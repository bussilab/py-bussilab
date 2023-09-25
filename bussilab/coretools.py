"""
General purpose tools.
"""
from contextlib import contextmanager
import gzip
import os
import unittest
import pathlib
import re
from typing import List, Optional

import numpy as np
import yaml

def ensure_np_array(arg) -> Optional[np.ndarray]:
    """Convert arg to np.array if necessary."""
    if arg is not None and not isinstance(arg, np.ndarray):
        return np.array(arg)
    return arg

def file_or_path(arg, mode: str):
    """Convert a path to an open file object if necessary."""
    if isinstance(arg, str):
        arg = open(arg, mode)
    if isinstance(arg, bytes):
        arg = open(str(arg), mode)
    if re.match(r".*\.gz", arg.name):
        arg = gzip.open(arg, mode)
    return arg

def import_numba_jit():
    """Return a numba.njit object. If import fails, return a fake jit object and emits a warning.

       Currently, the returned object can only be used as @njit (no option). It might be extended
       to allow more jit options.
    """
    try:
        from numba import njit as numba_jit
        return numba_jit
    except ImportError:
        import warnings
        warnings.warn("There was a problem importing numba, jit functions will work but will be MUCH slower.")
        def numba_jit(x):
            return x
        return numba_jit

class Result(dict):
    # triple ' instead of triple " to allow using docstrings in the example
    '''Base class for objects returning results.

       It allows one to create a return type that is similar to those
       created by `scipy.optimize.minimize`.
       The string representation of such an object contains a list
       of attributes and values and is easy to visualize on notebooks.

       Examples
       --------

       The simplest usage is this one:

       ```python
       from bussilab import coretools

       class MytoolResult(coretools.Result):
           """Result of a mytool calculation."""
           pass

       def mytool():
           a = 3
           b = "ciao"
           return MytoolResult(a=a, b=b)

       m=mytool()
       print(m)
       ```

       Notice that the class variables are dynamic: any keyword argument
       provided in the class constructor will be processed.
       If you want to enforce the class attributes you should add an explicit
       constructor. This will also allow you to add pdoc docstrings.
       The recommended usage is thus:

       ````
       from bussilab import coretools

       class MytoolResult(coretools.Result):
           """Result of a mytool calculation."""
           def __init__(a, b):
               super().__init__()
               self.a = a
               """Documentation for attribute a."""
               self.b = b
               """Documentation for attribute b."""

       def mytool():
           a = 3
           b = "ciao"
           return MytoolResult(a=a, b=b)

       m = mytool()
       print(m)
       ````

    '''

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, item: str, value):
        self[item] = value

    def __delattr__(self, item: str):
        del self[item]

    def __repr__(self) -> str:
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
# when used recursively, the inner repr is properly indented:
            return '\n'.join([k.rjust(m) + ': ' + re.sub("\n", "\n"+" "*(m+2), repr(v))
                              for k, v in sorted(self.items())])
        return self.__class__.__name__ + "()"

    def __dir__(self) -> List[str]:
        return list(sorted(self.keys()))

@contextmanager
def cd(newdir: os.PathLike, *, create: bool = False):
    """Context manager to temporarily change working directory.

       Can be used to change working directory temporarily making sure that at the
       end of the context the working directory is restored. Notably,
       it also works if an exception is raised within the context.

       Parameters
       ----------

       newdir : path
           Path to the desired directory.

       create : bool (default False)
           Create directory first.
           If the directory exists already, no error is reported

       Examples
       --------

       ```python
       from bussilab.coretools import cd
       with cd("/path/to/dir"):
           do_something() # this is executed in the /path/to/dir directory
       do_something_else() # this is executed in the original directory
       ```
    """
    prevdir = os.getcwd()
    path = os.path.expanduser(newdir)
    if create:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prevdir)

class TestCase(unittest.TestCase):
    """Improved base class for test cases.

       Extends the `unittest.TestCase` class with some additional assertion.

    """
    def assertEqualFile(self, file1: os.PathLike, file2: Optional[os.PathLike] = None):
        """Check if two files are equal.

           Parameters
           ----------

           file1: path
               Path to the first file

           file2: path, optional
               Path to the second file. If not provided, defaults to `file1+".ref"`.
        """
        if file2 is None:
            file2 = pathlib.PurePath(str(file1)+".ref")

        try:
            f1=open(file1, "r")
        except FileNotFoundError:
            self.fail("file " +str(file1) + " was not found")

        try:
            f2=open(file2, "r")
        except FileNotFoundError:
            self.fail("file " +str(file2) + " was not found")

        with f1:
            with f2:
                self.assertEqual(f1.read(), f2.read())

def config_path(path: Optional[os.PathLike] = None):
    if path is None:
        path = pathlib.PurePath(os.environ["HOME"]+"/.bussilabrc")
    return path

def config(path: Optional[os.PathLike] = None):
    with open(config_path(path)) as rc:
        return yaml.load(rc,Loader=yaml.BaseLoader)


