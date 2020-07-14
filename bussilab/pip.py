"""
Module implementing a small tool for installing and updating packages with pip.
"""

import subprocess
import sys

from typing import List, Union

def install(packages: Union[str, List[str]], *,
            upgrade: bool = False,
            user: bool = False):
    """Install one or more packages with pip.

       Install packages making sure they get installed with the currently used
       python interpreter.

       Parameters
       ----------

       packages : str or list
           Package to be installed/upgraded. If a list is passed, multiple
           packages are installed/upgraded.

       upgrade : bool, optional
           if True, run pip with `--upgrade`.

       user : bool, optional
           if True, run pip with `--user`.
    """

    args = [sys.executable, "-m", "pip", "install"]
    if user:
        args.append("--user")
    if upgrade:
        args.append("--upgrade")
    if isinstance(packages, str):
        args.append(packages)
    else:
        args.extend(packages)
    print("calling ", args)
    subprocess.check_call(args)

def upgrade_all(user: bool = False):
    """Upgrade all installed packages using pip.

       Warning: it assumes all available packages are installed with pip.

       Parameters
       ----------

       user : bool, optional
           if True, install/upgrade packages in with `--user` option.
    """
    try:
        import pkg_resources
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "pkg_resources not found, you should install setuptools"
            )
    packages = [dist.project_name for dist in pkg_resources.working_set] # pylint: disable=not-an-iterable
    install(packages, user=user, upgrade=True)
