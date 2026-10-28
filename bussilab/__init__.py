"""Scientific and utility tools collected in one Python package.

Most features live in submodules, which are imported only when requested.
Use `list_submodules()` to see what is available, or run `bussilab list`.
Command usage is documented in [cli_documentation](cli_documentation.html).
Source code is on [GitHub](https://github.com/bussilab/py-bussilab).
A [PDF manual](../bussilab.pdf) is also available.
This documentation describes version __version__.

# Install

This package supports Python 3.8 through 3.15.
The recommended way to install this package depends on how you prefer to manage your
python dependencies.

_pip_.
If you manage your dependencies with pip and install packages in your home, use:
```bash
pip install --user bussilab
# if the user site is not on sys.path, add it to PYTHONPATH
export PYTHONPATH="$(python -c 'import site; print(site.USER_SITE)'):$PYTHONPATH"
# if the user scripts directory is not on PATH, add it
export PATH="$(python -c 'import site; print(site.USER_BASE + "/bin")'):$PATH"
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
If you manage your dependencies with conda, use:
```bash
conda install -c conda-forge -c bussilab py-bussilab
```
Required packages will be downloaded and installed automatically in the active
conda environment.

_macports_.
If you manage your dependencies with macports you might prefer to install required packages
first. Since the list of requirements might change, it is recommended to use
the bussilab package itself to obtain the list of requirements.
You can do it as follows:
[^macports]
```bash
# install pip and setuptools first
sudo port install py_macports_pynd_-pip py_macports_pynd_-setuptools
# install a bare version of the package, without dependencies
pip-_macports_py_ install --user --no-deps bussilab
# if the user site is not on sys.path, add it to PYTHONPATH
export PYTHONPATH="$(python_macports_py_ -c 'import site; print(site.USER_SITE)'):$PYTHONPATH"
# if the user scripts directory is not on PATH, add it
export PATH="$(python_macports_py_ -c 'import site; print(site.USER_BASE + "/bin")'):$PATH"
# install the dependencies
sudo port install $(bussilab required --macports --pyver _macports_pynd_)
```
Notice that the list of required packages might change. It is
thus recommended to run the commands above every time you update the bussilab
package.

[^macports]:
Notice that on macports the name of the python executable is `python_macports_py_`
and the name of the pip installer is `pip-_macports_py_`.
A different python version should work as well.

### Checking the installation

Once the package is installed, check that Python can import it and its submodules:[^pythonpath]
```python
import bussilab
# this command can be used to check if all dependencies are in place:
bussilab.import_submodules()
```

[^pythonpath]:
If you installed the package using the `--user` option, in order to be able to import
the package you will have to make sure that the directory returned by the command
`python -c 'import site; print(site.USER_SITE)'` is included in your python search path.
If not, you can add it to the environment variable `PYTHONPATH`.

You can check the command-line entry point as well:[^path]
```bash
bussilab -h
# this command can be used to check if all dependencies are in place:
bussilab check --import
```

[^path]:
If you installed the package using the `--user` option, in order to be able to execute
the script you will have to make sure that the directory returned by the command
`python -c 'import site; print(site.USER_BASE + "/bin")'` is included in your `PATH`
environment variable. Alternatively, if the `bussilab` script is not in your
`PATH` you can run it as `python -m bussilab -h`.

### Setting autocompletion

In order to benefit from autocompletion for the executable script you should
install the [argcomplete](https://github.com/kislyuk/argcomplete) package
(with pip, conda, or macports)
and add the following command to your `.bashrc` file:[^macports_auto]
```bash
eval "$(register-python-argcomplete bussilab)"
```

[^macports_auto]:
If you are using macports, the command would be
`eval "$(register-python-argcomplete-_macports_py_ bussilab)"`.

# Getting started

### Python

The `bussilab` package itself contains lightweight infrastructure. Import
submodules as needed:
```python
from bussilab import wham
import numpy as np

bias = np.array([[1.0, 2.0], [2.0, 1.0]])
result = wham.wham(bias)
print(result.logW)
```
Submodules can also be accessed as attributes of `bussilab`; they are imported
on first access. See each submodule's documentation for its Python API.
`rna2d.Molecule` additionally requires the optional ViennaRNA Python package.

### Examples

In the [examples](../examples) directory you can find a number of notebooks
that can be used as a source of inspiration.

### Command line

In addition, the `bussilab` script allows to access some functionality
directly from the command line, without entering python.
For example, to see the WHAM command's required arguments, run:
```bash
bussilab wham -h
```
See [cli_documentation](cli_documentation.html) for all subcommands.

### Changes

Each submodule has its own release history. To receive updates when a release
includes changes to a submodule you use, subscribe to its RSS feed.
Feeds are available at `feeds/MODULE.xml` for modules with a changelog;
for example, subscribe to the [WHAM RSS feed](feeds/wham.xml) for WHAM updates.

# Advanced stuff

Notice that instructions below assume you are using pip.
If you use conda or macports you might have to adjust the commands.

### Install with no dependencies

You might want to ignore dependencies completely:
```bash
pip install --no-deps bussilab
```
You can still import the `bussilab` package and run the CLI, but submodules
whose dependencies are missing may not import. Check them with:
```bash
bussilab check --import
```
This command reports imports that succeed or fail, then raises an error if any
submodule fails to import.

### Building documentation

You can build the documentation using `pdoc3`:
```bash
pip install pdoc3
pdoc3 -f --html -o doc/ bussilab
```

Documentation will be visible at `doc/bussilab/index.html`.
This is normally not necessary, since
the pre-built documentation of the latest version can be found at [this link](https://bussilab.github.io/doc-py-bussilab),
but can be useful to test changes to the documentation before pushing them to GitHub.

### Implementing your modifications

If you want to modify the Python source code you should download it from GitHub:

```bash
git clone https://github.com/bussilab/py-bussilab.git
cd py-bussilab
```

Since the module is
written in pure python,[^pure_python] it can be used by just adding its path to `PYTHONPATH`
and the `bin` directory to `PATH`:
```bash
export PATH="/path/to/py-bussilab/bin:$PATH"
export PYTHONPATH="/path/to/py-bussilab:$PYTHONPATH"
```

[^pure_python]: This might change in the future.

### Testing the code

If you modified the code, it is recommended to test that your changes did not break
existing features.

You can run the tests using `pytest`:
```bash
pip install pytest
pytest
```
Tests for optional features may be skipped when their dependencies are absent.
Notice that the tests will import the `bussilab` module.
Thus, if you want the version modified by you to be tested (and not another version
that you might have installed with pip already) you should properly set
the `PYTHONPATH` variable as explained above.

Static types can be checked using `mypy`:
```bash
pip install mypy
mypy bussilab
```
Run this optional check when changing type annotations.

Other static checks can be done with `pyflakes`:
```bash
pip install pyflakes
pyflakes bussilab
```
This check should succeed.

Additional static errors can be checked using `pylint`:
```bash
pip install pylint
pylint -E bussilab
```
This check should succeed.

### Running and rendering jupyter examples from the command line

The GitHub repository also contains jupyter examples.
You can rerun all the jupyter examples from the command line:
```bash
cd examples
pip install jupyter jupyter_contrib_nbextensions matplotlib
./rerun.sh
```

You can render all the jupyter examples as html from the command line:
```bash
cd examples
pip install jupyter nbconvert
./render.sh
```

"""

# Minimal import here to speedup the loading of this module.
# Notice that we should not import any required package here, or
# setup will fail (this package is imported to find our requirements).

from typing import List
from ._version import __version__

# required packages:
_required_ = [
    'argcomplete',
    'networkx',
    'numba',
    'numpy',
    'scipy',
    'slack-sdk>=3.19.0',
    'pyyaml',
    'typing_extensions'
]

_macports_py_ = "3.9"

def _process_macports(requirements: List[str]) -> List[str]:
    import re
    l = []
    for d in requirements:
        d = re.sub(">.*$", "", d)  # remove versions
        d = re.sub("^pyyaml$", "yaml", d)  # fix yaml
        l.append("py-" + d)
    return l

_macports_required_ = _process_macports(_required_)

# process documentation in order to update variables in a single point.
def _process_doc(doc: str) -> str:
    import re
    _macports_py_not_dot = _macports_py_[0] + _macports_py_[2]
    _macports_install_ = (
        "sudo port install "
        + re.sub("py-", "py" + _macports_py_not_dot + '-', ' '.join(_macports_required_))
    )
    doc = re.sub("__version__", __version__, doc)
    doc = re.sub("_macports_install_", _macports_install_, doc)
    doc = re.sub("_macports_py_", _macports_py_, doc)
    doc = re.sub("_macports_pynd_", _macports_py_not_dot, doc)
    return doc

__doc__ = _process_doc(__doc__)

def list_submodules(*, _hidden: bool = False) -> List[str]:
    """Return a list of all the available submodules.

       The names come from Python files in the package directory. By default,
       modules whose names start with an underscore are omitted; `_hidden=True`
       includes them. Listing modules does not import them.

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
    """Import all the available submodules, including hidden ones.

       The main `bussilab` module does not explicitly import
       the available submodules so as not to slow down the behavior
       of the command line interface and to allow importing individual modules
       even if not all the dependencies of the other modules are installed.
       Use this function to import all the submodules (might take a few seconds).

       Prints the result of each import. Import failures are collected and
       reported after all modules have been tried. This is useful for checking
       dependencies or preloading submodules.

       Raises
       ------

       Exception
           If one or more submodules raises `ImportError` during import.
           Other exceptions propagate immediately.
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
    """Return the first line of a submodule's source docstring without importing it.

       Parameters
       ----------

       module : str

           Name of the module.

       Returns
       -------

       str
           The first line of the module docstring, or an empty string if it has
           no docstring. For `module=""`, return the first line of the package
           docstring.

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
    if module != "" and module not in list_submodules(_hidden=True):
        raise ModuleNotFoundError(module)
    if module == "":
        module = "__init__"
    with open(os.path.dirname(__file__) + "/" + module + ".py") as f:
        M = ast.parse(''.join(f))
    d = ast.get_docstring(M)
    if d is None:
        return ""
    # grab first line
    return d.partition('\n')[0]

def required_macports(pyver="") -> str:
    """Return space-separated MacPorts dependencies.

       `pyver` is the two-digit Python version suffix, such as `"39"`.
       When omitted, generic `py-` port names are returned.
    """
    import re
    req = re.sub("py-", "py" + pyver + '-', ' '.join(_macports_required_))
    if pyver=="36" or pyver=="37":
        req = re.sub("-slack-sdk","-slackclient",req)
    return req

def required_conda() -> List[str]:
    """Return package dependencies as a list for conda environments."""
    return list(_required_)

def required_pip() -> str:
    """Return space-separated package dependencies for pip."""
    return str(' '.join(_required_))

# See PEP 562.
def __getattr__(name):
    if name in list_submodules(_hidden=True):
        import importlib
        import os
        importlib.import_module("." + name, os.path.basename(os.path.dirname(__file__)))
        return globals()[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    # If later on we support __all__, globals() should be replaced by __all__.
    return sorted(set(globals()) | set(list_submodules(_hidden=True)))
