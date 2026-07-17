"""
Install and build selected external scientific packages.

At present, the functions in this module support Google Colab only.
"""

import importlib
import os
import platform
import re
import shutil
import site
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple, Union


PathLike = Union[str, os.PathLike]

_VIENNARNA_RELEASE_REPOSITORY = "bussilab/ViennaRNA-patches"
_DEFAULT_VIENNARNA_VERSION = "2.7.2-patch1"


def _in_colab() -> bool:
    """Return True when running inside a Google Colab kernel."""
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def _require_colab() -> None:
    """Raise an exception unless the current runtime is Google Colab."""
    if not _in_colab():
        raise RuntimeError(
            "This installer currently supports Google Colab only. "
            "Support for other runtimes may be added in the future."
        )


def _parse_viennarna_version(version: str) -> Tuple[str, int]:
    """Parse strings such as '2.7.2-patch1'."""
    match = re.fullmatch(
        r"(?P<upstream>[0-9]+(?:\.[0-9]+)+)-patch(?P<patch>[1-9][0-9]*)",
        version,
    )
    if match is None:
        raise ValueError(
            "ViennaRNA version must have the form "
            "'UPSTREAM_VERSION-patchNUMBER', for example "
            "'2.7.2-patch1'."
        )

    return match.group("upstream"), int(match.group("patch"))


def _read_os_release() -> Dict[str, str]:
    """Read /etc/os-release."""
    values = {}

    with open("/etc/os-release", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key] = value.strip().strip('"')

    return values


def _glibc_version() -> str:
    """Return the major.minor glibc version."""
    libc_name, libc_version = platform.libc_ver()

    if libc_name.lower() == "glibc" and libc_version:
        match = re.match(r"([0-9]+\.[0-9]+)", libc_version)
        if match is not None:
            return match.group(1)

    output = subprocess.check_output(
        ["ldd", "--version"],
        text=True,
        stderr=subprocess.STDOUT,
    ).splitlines()[0]

    match = re.search(r"([0-9]+\.[0-9]+)(?:[^\d].*)?$", output)
    if match is None:
        raise RuntimeError(
            "Could not determine the glibc version from: " + output
        )

    return match.group(1)


def _runtime_information() -> Dict[str, str]:
    """Return information used to identify compatible binary archives."""
    os_release = _read_os_release()

    ubuntu = os_release.get("VERSION_ID")
    if not ubuntu:
        raise RuntimeError("Could not determine the Ubuntu version.")

    return {
        "platform": "colab",
        "architecture": platform.machine(),
        "ubuntu": ubuntu,
        "glibc": _glibc_version(),
        "python": "{}.{}".format(
            sys.version_info.major,
            sys.version_info.minor,
        ),
        "accelerator": "cpu",
    }


def _viennarna_asset_name(
    version: str = _DEFAULT_VIENNARNA_VERSION,
) -> str:
    """Construct the ViennaRNA release asset name for this runtime."""
    upstream, patch = _parse_viennarna_version(version)
    runtime = _runtime_information()

    return (
        "ViennaRNA-{}-patch{}-{}-{}-ubuntu{}-glibc{}-python{}-{}.tgz"
    ).format(
        upstream,
        patch,
        runtime["platform"],
        runtime["architecture"],
        runtime["ubuntu"],
        runtime["glibc"],
        runtime["python"],
        runtime["accelerator"],
    )


def _viennarna_release_url(
    version: str = _DEFAULT_VIENNARNA_VERSION,
) -> str:
    """Construct the GitHub Release URL for a ViennaRNA archive."""
    upstream, patch = _parse_viennarna_version(version)
    asset = _viennarna_asset_name(version)

    return (
        "https://github.com/{repository}/releases/download/"
        "v{upstream}-patch{patch}/{asset}"
    ).format(
        repository=_VIENNARNA_RELEASE_REPOSITORY,
        upstream=upstream,
        patch=patch,
        asset=asset,
    )


def _safe_extract_viennarna(
    archive: Path,
    destination: Path,
) -> None:
    """Extract an archive containing only ViennaRNA and its descendants."""
    destination = destination.resolve()

    with tarfile.open(str(archive), "r:gz") as tar:
        members = tar.getmembers()

        for member in members:
            name = member.name

            if name != "ViennaRNA" and not name.startswith("ViennaRNA/"):
                raise RuntimeError(
                    "Unexpected member in ViennaRNA archive: {!r}".format(
                        name
                    )
                )

            if member.issym() or member.islnk():
                raise RuntimeError(
                    "Links are not permitted in the ViennaRNA archive: "
                    "{!r}".format(name)
                )

            target = (destination / name).resolve()

            try:
                target.relative_to(destination)
            except ValueError as exc:
                raise RuntimeError(
                    "Unsafe path in ViennaRNA archive: {!r}".format(name)
                ) from exc

        # Paths and member types have been checked explicitly above.
        tar.extractall(str(destination), members=members)


def _viennarna_python_directory(prefix: Path) -> Path:
    """Return the expected ViennaRNA Python installation directory."""
    python_version = "{}.{}".format(
        sys.version_info.major,
        sys.version_info.minor,
    )

    return (
        prefix
        / "local"
        / "lib"
        / ("python" + python_version)
        / "dist-packages"
    )


def _prepend_environment_path(variable: str, value: Path) -> None:
    """Prepend a directory to a colon-separated environment variable."""
    value_string = str(value)
    current = os.environ.get(variable, "")

    entries = [entry for entry in current.split(os.pathsep) if entry]

    if value_string in entries:
        entries.remove(value_string)

    os.environ[variable] = os.pathsep.join([value_string] + entries)


def install_viennarna(
    version: str = _DEFAULT_VIENNARNA_VERSION,
    prefix: Optional[PathLike] = None,
    force: bool = False,
    verbose: bool = True,
):
    """
    Install a precompiled patched ViennaRNA distribution in Google Colab.

    Parameters
    ----------
    version
        Patched release identifier, for example ``"2.7.2-patch1"``.
    prefix
        Installation directory. The default is ``ViennaRNA`` below the
        current working directory.
    force
        Download and extract the archive again if the installation directory
        already exists.
    verbose
        Print progress information.

    Returns
    -------
    module
        The imported :mod:`RNA` module.

    Raises
    ------
    RuntimeError
        If the function is not running in Google Colab, the archive is
        malformed, or the installed module cannot be imported.
    """
    _require_colab()
    _parse_viennarna_version(version)

    if prefix is None:
        prefix_path = Path.cwd() / "ViennaRNA"
    else:
        prefix_path = Path(prefix).expanduser().resolve()

    if prefix_path.name != "ViennaRNA":
        raise ValueError(
            "The installation prefix must end in 'ViennaRNA', because "
            "the release archive contains that top-level directory."
        )

    archive_name = _viennarna_asset_name(version)
    url = _viennarna_release_url(version)

    if verbose:
        print("ViennaRNA asset:", archive_name)
        print("Installation prefix:", prefix_path)

    if force and prefix_path.exists():
        shutil.rmtree(str(prefix_path))

    if not prefix_path.exists():
        prefix_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(
            prefix="bussilab-viennarna-"
        ) as temporary_directory:
            archive = Path(temporary_directory) / archive_name

            if verbose:
                print("Downloading:", url)

            try:
                urllib.request.urlretrieve(url, str(archive))
            except Exception as exc:
                raise RuntimeError(
                    "Could not download the ViennaRNA binary archive.\n"
                    "Expected asset: {}\n"
                    "URL: {}\n"
                    "The current Colab runtime may no longer match an "
                    "available release.".format(archive_name, url)
                ) from exc

            _safe_extract_viennarna(
                archive,
                prefix_path.parent,
            )

    binary_directory = prefix_path / "bin"
    library_directory = prefix_path / "lib"
    python_directory = _viennarna_python_directory(prefix_path)

    if not python_directory.is_dir():
        raise RuntimeError(
            "ViennaRNA was extracted, but its Python package directory "
            "was not found: {}".format(python_directory)
        )

    _prepend_environment_path("PATH", binary_directory)
    _prepend_environment_path("LD_LIBRARY_PATH", library_directory)

    site.addsitedir(str(python_directory))
    importlib.invalidate_caches()

    # Ensure that a failed or previous import does not hide the installation
    # that was just selected.
    sys.modules.pop("RNA", None)

    try:
        rna = importlib.import_module("RNA")
    except Exception as exc:
        raise RuntimeError(
            "ViennaRNA was installed at {}, but importing RNA failed.".format(
                prefix_path
            )
        ) from exc

    if verbose:
        print(
            "ViennaRNA version:",
            getattr(rna, "__version__", "unknown"),
        )
        print("Installation successful.")

    return rna


def build_viennarna(
    version: str = _DEFAULT_VIENNARNA_VERSION,
    output_directory: Optional[PathLike] = None,
    jobs: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[Path, Path]:
    """
    Build a patched ViennaRNA archive in Google Colab.

    The archive and its complete build log receive the same basename.

    Parameters
    ----------
    version
        Patched release identifier, for example ``"2.7.2-patch1"``.
    output_directory
        Destination for the ``.tgz`` and ``.log`` files. The default is the
        current working directory.
    jobs
        Number of parallel make jobs. The default is at most two, to avoid
        excessive memory use while compiling the SWIG wrapper.
    verbose
        Print progress information and the final elapsed time.

    Returns
    -------
    archive, log
        Paths of the generated archive and build log.
    """
    _require_colab()

    upstream, patch = _parse_viennarna_version(version)

    if output_directory is None:
        output_path = Path.cwd()
    else:
        output_path = Path(output_directory).expanduser().resolve()

    output_path.mkdir(parents=True, exist_ok=True)

    asset_name = _viennarna_asset_name(version)
    archive = output_path / asset_name
    log = archive.with_suffix(".log")

    if jobs is None:
        jobs = min(2, os.cpu_count() or 1)

    if jobs < 1:
        raise ValueError("jobs must be a positive integer.")

    prefix = Path("/content/ViennaRNA")
    build_directory = Path("/content/build-viennarna")

    patch_file = "patch-viennarna-{}-{}.patch".format(
        upstream,
        patch,
    )

    source_url = (
        "https://github.com/ViennaRNA/ViennaRNA/releases/download/"
        "v{version}/ViennaRNA-{version}.tar.gz"
    ).format(version=upstream)

    patch_url = (
        "https://raw.githubusercontent.com/"
        "bussilab/ViennaRNA-patches/main/{patch_file}"
    ).format(patch_file=patch_file)

    script = r"""
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y \
    autoconf \
    automake \
    build-essential \
    curl \
    libtool \
    pkg-config \
    python3-dev \
    wget

python -m pip install --upgrade "swig>=4.3.0"

PYTHON_EXE="$(python -c 'import sys; print(sys.executable)')"
export PYTHON="$PYTHON_EXE"

echo "Build started: $(date --iso-8601=seconds)"
echo "Python executable: $PYTHON"
python --version
swig -version
echo "Source URL: $SOURCE_URL"
echo "Patch URL: $PATCH_URL"
echo "Prefix: $PREFIX"
echo "Parallel jobs: $JOBS"

rm -rf "$BUILD_DIRECTORY" "$PREFIX" "$ARCHIVE"
mkdir -p "$BUILD_DIRECTORY"
cd "$BUILD_DIRECTORY"

wget -O "ViennaRNA-${UPSTREAM}.tar.gz" "$SOURCE_URL"
tar -xzf "ViennaRNA-${UPSTREAM}.tar.gz"

wget -O "$PATCH_FILE" "$PATCH_URL"

cd "ViennaRNA-${UPSTREAM}"
patch -p1 < "../${PATCH_FILE}"

./configure \
    PYTHON="$PYTHON" \
    --prefix="$PREFIX" \
    --without-doc \
    --with-python

make -j"$JOBS"
make install

find "$PREFIX" \( -name 'RNA.py' -o -name '_RNA*.so' \) -print

cd "$(dirname "$PREFIX")"
tar -czf "$ARCHIVE" "$(basename "$PREFIX")"

echo "Created archive:"
ls -lh "$ARCHIVE"
echo "Build completed: $(date --iso-8601=seconds)"
"""

    environment = os.environ.copy()
    environment.update(
        {
            "UPSTREAM": upstream,
            "PATCH_FILE": patch_file,
            "SOURCE_URL": source_url,
            "PATCH_URL": patch_url,
            "PREFIX": str(prefix),
            "BUILD_DIRECTORY": str(build_directory),
            "ARCHIVE": str(archive),
            "JOBS": str(jobs),
        }
    )

    if verbose:
        print("Building:", asset_name)
        print("Log:", log)

    started = time.monotonic()

    with log.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            ["bash", "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()

            if verbose:
                print(line, end="")

        return_code = process.wait()

    elapsed = time.monotonic() - started

    with log.open("a", encoding="utf-8") as log_handle:
        log_handle.write(
            "Elapsed time: {:.1f} seconds ({:.2f} minutes)\n".format(
                elapsed,
                elapsed / 60.0,
            )
        )

    if verbose:
        print(
            "Elapsed time: {:.1f} seconds ({:.2f} minutes)".format(
                elapsed,
                elapsed / 60.0,
            )
        )

    if return_code != 0:
        raise RuntimeError(
            "ViennaRNA build failed with exit status {}. "
            "See the complete log at {}.".format(return_code, log)
        )

    if not archive.is_file():
        raise RuntimeError(
            "The ViennaRNA build completed without producing {}".format(
                archive
            )
        )

    return archive, log
