"""A package collecting a heterogeneous set of tools.

This package collects a number of tools that are useful enough
to be distributed but too small to deserve being published as
separate packages.
Submodules are listed at the end of the current page.
Command-line tools are described at [this page](cli_documentation.html).
This is the documentation for version __version__.

# Install

This package is only compatible with Python >=3.6 (no compatibility with Python 2!).
The recommended way to install this package depends on how you prefer to manage your
python dependencies.

_pip_.
If you manage your dependencies with pip and install packages in your home, use:
```bash
pip install --user bussilab
```
Required packages will be downloaded and installed automatically in your home.

_pip + venv_.
If you manage your dependencies with pip and work in a [virtual
environment](https://docs.python.org/3/library/venv.html), use:
```bash
pip install bussilab
```
Required packages will be downloaded and installed automatically in the virtual
environment.


_conda_.
If you manage your dependencies with conda you might prefer to install [required
packages first](https://www.anaconda.com/using-pip-in-a-conda-environment/):
```bash
_conda_install_
pip install --no-deps bussilab
```

_macports_.
If you manage your dependencies with macports you might prefer to install required packages
first:[^macports]
```bash
_macports_install_
pip-_macports_py_ install --user --no-deps bussilab
```

[^macports]:
Notice that on macports the name of the python executable is `python_macports_py_`
and the name of the pip installer is `pip-_macports_py_`.
A different python version should work as well.

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
environment variable. Alternatively, if the `bussilab` script is not in your
`PATH` you can run it as `python -m bussilab -h`.

### Setting autocompletion

In order to benefit from autocompletion for the executable script you should
add the following command to your `.bashrc` file:[^macports_auto]
```bash
eval "$(register-python-argcomplete bussilab)"
```

[^macports_auto]:
If you are using macports, the command would be
`eval "$(register-python-argcomplete-_macports_py_ bussilab)"`.

# Getting started

### Python

The `bussilab` module itself only contains some basic infrastructure.
Most of the features are implemented in the submodules listed at the end of
this page.
Submodules should be explicitly imported using, e.g.:
```python
from bussilab import wham
```
Check their documentation to see how to use them.

If you are using Python >=3.7, you can directly use the submodules
without importing them explicitly, e.g.:
```python
import bussilab as bl
bl.wham.wham()
```

You can also have a look at the notebooks in the [examples](../examples) directory.

### Command line

In addition, the `bussilab` script allows to access some functionality
directly from the command line, without entering python.
For instance, you can execute the wham subcommand typing
```bash
bussilab wham
```
Check their documentation in the [cli_documentation](cli_documentation.html) submodule.

# Advanced stuff

Notice that instructions below assume you are using pip.
If you use conda or macports you might have to adjust the commands.

### Install with no dependencies

You might want to ignore dependencies completely:
```bash
pip install --no-deps bussilab
```
If you proceed this way, you will still be able to import `bussilab` module and
to execute the `bussilab` script, but some of the submodules might not be importable.
The following command will report which submodules can be used then:
```bash
bussilab check --import
```
The result will depend on which of the required packages are already installed on
your system.

### Run without installing

You might want to use this module without installing it. Since the module is
written in pure python,[^pure_python] it can be used by just adding its path to `PYTHONPATH`
and the `bin` directory to `PATH`:
```bash
export PATH="/path/to/bussilab/bin:$PATH"
export PYTHONPATH="/path/to/bussilab:$PYTHONPATH"
```

[^pure_python]: This might change in the future.

### Running tests

You can run the tests using `pytest`:
```bash
pip install pytest
py.test
```

All tests should succeed.

### Running jupyter examples from the command line

You can rerun all the jupyter examples from the command line:
```bash
cd examples
pip install jupyter jupyter_contrib_nbextensions matplotlib
bash rerun.sh
```

### Building documentation

You can build the documentation using `pdoc3`:
```bash
pip install pdoc3
pdoc3 -f --html -o doc/ bussilab
```

Documentation will be found in `doc/bussilab/index.html`.

### Static analysis of the code

Static types can be checked using `mypy`:
```bash
pip install mypy
mypy bussilab
```
This check should succeed.

Other static checks can be done with `pyflakes`:
```bash
pip install pyflakes
pyflakes bussilab
```
This check should succeed.

Correct code formatting can be checked using `pylint`:
```bash
pip install pylint
pylint bussilab
```
This check might report a number of false positives.

"""

# Minimal import here to speedup the loading of this module.

from typing import List as _List
import sys as _sys

# version number:
__version__ = "0.0.6"

# required packages:
_required_ = [
    'argcomplete',
    'numba',
    'numpy',
    'scipy',
    'slackclient',
    'pyyaml'
]

_macports_py_ = "3.7"

# process documentation in order to update variables in a single point.
def _process_doc(doc: str) -> str:
    import re
    _conda_install_ = "conda install "+" ".join(_required_)
    _macports_py_not_dot = _macports_py_[0] + _macports_py_[2]
    _macports_install_ = (
        "sudo port install "
        + "py" + _macports_py_not_dot + "-pip "
        + "py" + _macports_py_not_dot + "-setuptools "
        +" ".join(["py" + _macports_py_not_dot + "-" + mod for mod in _required_])
    )
    _macports_install_ = re.sub('-pyyaml', '-yaml', _macports_install_)
    doc = re.sub("__version__", __version__, doc)
    doc = re.sub("_conda_install_", _conda_install_, doc)
    doc = re.sub("_macports_install_", _macports_install_, doc)
    doc = re.sub("_macports_py_", _macports_py_, doc)
    return doc

__doc__ = _process_doc(__doc__)

def list_submodules(*, _hidden: bool = False) -> _List[str]:
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
    for module in sorted(os.listdir(os.path.dirname(__file__))):
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
        module = "__init__"
    M = ast.parse(''.join(open(os.path.dirname(__file__) + "/" + module + ".py")))
    d = ast.get_docstring(M)
    if d is None:
        return ""
    # grab first line
    return d.partition('\n')[0]

# See this https://www.python.org/dev/peps/pep-0562/
# Also notice that this solution only works with python>=3.7
if _sys.version_info >= (3, 6):
    def __getattr__(name):
        if name in list_submodules(_hidden=True):
            import importlib
            import os
            importlib.import_module("."+name, os.path.basename(os.path.dirname(__file__)))
            return globals()[name]
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    def __dir__():
# if later on we support __all__, globals() should be replaced by __all__
        return sorted(list(globals()) + list_submodules(_hidden=True))
