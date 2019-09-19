"""A package collecting a heterogeneous set of tools.

This package is only compatible with Python >=3.6 (no compatibility with Python 2!).
This is the documentation for version __version__.

# Install

The recommended way to install this package depends on how you prefer to manage your
python dependencies.

_pip_.
If you manage your dependencies with pip, just use:[^pipuser]
```bash
pip install --user .
```
Required packages will be downloaded and installed automatically.

[^pipuser]:
The flag `--user` is not necessary if you are in a [virtual
environment](https://docs.python.org/3/library/venv.html).

_conda_.
If you manage your dependencies with conda you might prefer to install required packages first:
```bash
_conda_install_
pip install --no-deps .
```

_macports_.
If you manage your dependencies with macports you might prefer to install required packages
first:[^macports]
```bash
_macports_install_
pip-_macports_py_ install --user --no-deps .
```

[^macports]:
Notice that on macports the name of the python executable is `python_macports_py_`
and the name of the pip installer is `pip-_macports_py_`.

### Checking the installation

Once the package is installed you should be able to import the module from the python interpreter:
```python
import bussilab
```
You should also have access to an executable script that can be used from the command line:[^path]
```bash
bussilab -h
```

[^path]:
If you installed the package using the `--user` option, in order to be able to execute
the script you will have to make sure that the directory returned by the command
`python -c 'import site; print(site.USER_BASE + "/bin")'` is included in your `PATH`
environment variable.

### Setting autocompletion

In order to benefit from autocompletion for the executable script you should
add the following command to your `.bashrc` file:[^macports_auto]
```bash
eval "$(register-python-argcomplete bussilab)"
```

[^macports_auto]:
If you are using macports, the command would be
`eval "$(register-python-argcomplete-_macports_py_ bussilab)"`.

### Install with no dependencies

You might want to ignore dependencies completely:
```bash
pip install --user --no-deps .
```
If you proceed this way, you will still be able to import `bussilab` module and
to execute the `bussilab` script, but some of the submodules might not be importable.
The following command will report which submodules can be used then:
```bash
bussilab check --import
```
The result will depend on which of the required packages are already installed on
your system.

# Getting started

### Python

The `bussilab` module itself only contains some basic infrastructure.
Most of the features are implemented in the submodules listed below.
Submodules should be explicitly imported using, e.g.:
```python
from bussilab import wham
```
Check their documentation to see how to use them.

### Command line

In addition, the `bussilab` script allows to access some functionality
directly from the command line, without entering python.
For instance, you can execute the `wham` subcommand typing
```bash
bussilab wham
```
Check their documentation in the [cli_documentation](cli_documentation.html) submodule.

# Advanced stuff

Notice that instructions below assume you are using pip.
If you use conda or macports you might have to adjust the commands.

### Running tests

You can run the tests using `pytest`:
```bash
pip install pytest
py.test
```

All tests should succeed.

### Building documentation

You can build the documentation using `pdoc3`:
```bash
pip install pdoc3
pdoc3 -f --html -o doc/ bussilab
```

Documentation will be found in `doc/bussilab/index.html`.

### Static analysis of the code

You can check for correct formatting of the code using `pylint`:
```bash
pip install pylint
pylint bussilab
```

You can also check static types using `mypy`:
```bash
pip install mypy
mypy bussilab
```

Other static checks can be done with `pyflakes`:
```bash
pip install pyflakes
pyflakes bussilab
```

Notice that these three tools all report a number of irrelevant warnings.

"""

# No import here to speedup the loading of this module.

# We only import 'typing' that is needed for static type checks.
from typing import List

# version number:
__version__ = "0.1"

# required packages:
_required_ = [
    'argcomplete',
    'numpy',
    'scipy'
]

_macports_py_ = "3.7"

# process documentation in order to update variables in a single point.
def _process_doc(doc: str) -> str:
    import re
    _conda_install_ = "conda install "+" ".join(_required_)
    _macports_py_not_dot = _macports_py_[0] + _macports_py_[2]
    _macports_install_ = "sudo port install "+" ".join(
        ["py" + _macports_py_not_dot + "-" + mod for mod in _required_])
    doc = re.sub("__version__", __version__, doc)
    doc = re.sub("_conda_install_", _conda_install_, doc)
    doc = re.sub("_macports_install_", _macports_install_, doc)
    doc = re.sub("_macports_py_", _macports_py_, doc)
    return doc

__doc__ = _process_doc(__doc__)

def list_submodules(*, _hidden: bool = False) -> List[str]:
    """Return a list of all the available submodules.

       It can be used to quickly show which submodules are available for importing.

       Returns
       -------

       list
           A list of names of available submodules.

       Examples
       --------

       Print the available submodules and a short description for each of them.
       ```
       from bussilab import list_submodules, describe_submodule
       for m in list_submodules():
           print(m, describe_submodule(m))
       ```
    """
    import os
    submodules = []
    for module in os.listdir(os.path.dirname(__file__)):
        if module[:2] == '__' or module[-3:] != '.py':
            continue
        if not _hidden and module[:1] == '_':
            continue
        submodules.append(module[:-3])
    return submodules

def import_submodules() -> None:
    """Import all the available submodules.

       The main `bussilab` module does not explicitly import
       the available submodules so as not to slow down the behavior
       of the command line interface and to allow importing individual modules
       even if not all the dependencies of the other modules are installed.
       Use this function to import all the submodules (might take a few seconds).

       Mostly for testing that the packages required by the available
       submodules are installed, but it can also be used
       to preload all the submodules (and required packages) making them load faster later.

       Raises
       ------

       If one or more submodules cannot be imported it will raise an `Exception`.
    """
    import importlib
    import os
    failed = []
    for module in list_submodules(_hidden=True):
        try:
            print(module+" ...", end="")
            importlib.import_module("."+module, os.path.basename(os.path.dirname(__file__)))
            print(" ok")
        except ImportError:
            failed.append(module)
            print(" failed")
    if failed:
        msg = "error importing submodules "+str(failed)
        raise Exception(msg)

def describe_submodule(module: str) -> str:
    """Return a short description of a submodule without importing it.

       Parameters
       ----------

       module : str

           Name of the module.

       Returns
       -------

       str
           The docstring of the module. If the docstring is not present,
           returns an empty string. If an empy string is passed as `module`,
           the docstring ot the main package is returned.

       Raises
       ------

       ModuleNotFoundError
           If the module does not exist.

       Examples
       --------

       ```
       from bussilab import describe_submodule
       print(describe_submodule("lohman"))
       ```
    """
    import os
    import ast
    if module != "" and not module in list_submodules(_hidden=True):
        raise ModuleNotFoundError(module)
    if module == "":
        module="__init__"
    M = ast.parse(''.join(open(os.path.dirname(__file__) + "/" + module + ".py")))
    d = ast.get_docstring(M)
    if d is None:
        return ""
    # grab first line
    return d.partition('\n')[0]
