"""
Helpers for invoking pip with the current Python interpreter.

For shared or version-pinned environments, build and validate a new environment
and publish it under a new version or date, keeping the previous one available
for rollback. Upgrading an existing environment does not preserve its pins.

## Changelog

- **v0.0.53**
  - Deprecated `upgrade_all()` and the `pip_upgrade_all` command in favor of
    rebuilding environments from pinned requirements.
  - Replaced `pkg_resources` package discovery with `importlib.metadata`.
- **v0.0.35**
  - Added `upgrade_self()` for upgrading bussilab, used by cron's self-update
    step. Added optional subprocess timeouts to all pip helpers.
- **v0.0.2**
  - Introduced `install()` and `upgrade_all()`, invoking pip with the current
    Python interpreter and supporting upgrades and user installations.
"""

import subprocess
import sys
from importlib import metadata
import warnings

from typing import List, Union, Optional

def install(packages: Union[str, List[str]], *,
            upgrade: bool = False,
            user: bool = False,
            timeout: Optional[int] = None):
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
    subprocess.check_call(args, timeout=timeout)

def upgrade_all(user: bool = False,*,
                timeout: Optional[int] = None):
    """Upgrade all discovered packages using pip (deprecated).

       This modifies the environment in place, without preserving version pins
       or providing validation or rollback. Build a new environment from pinned
       requirements instead. This function remains available for compatibility.
       It assumes all discovered packages are managed by pip; with `user=True`,
       packages visible outside the user location can also be installed there.

       Parameters
       ----------

       user : bool, optional
           if True, install/upgrade packages with the `--user` option.
    """
    warnings.warn(
        "upgrade_all() and bussilab pip_upgrade_all are deprecated. "
        "Build and validate a new environment from pinned requirements instead "
        "of upgrading all packages in place.",
        FutureWarning, stacklevel=2)
    packages = sorted({dist.metadata["Name"] for dist in metadata.distributions()
                       if dist.metadata["Name"]})
    if packages:
        install(packages, user=user, upgrade=True, timeout=timeout)

def upgrade_self(*,
                 user: bool = False,
                 timeout: Optional[int] = None):
    """Upgrade bussilab in place using pip, potentially updating dependencies.

    Intended for environments deliberately maintained in place, including
    cron's explicit `selfupdate` step. For shared or pinned environments,
    publish a new environment instead.
    """
    install("bussilab", user=user, timeout=timeout, upgrade=True)
