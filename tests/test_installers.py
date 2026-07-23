import io
import tarfile

import pytest

from bussilab import installers


@pytest.fixture
def colab_runtime(monkeypatch):
    """Provide deterministic mocked Colab runtime information."""
    monkeypatch.setattr(installers, "_in_colab", lambda: True)
    monkeypatch.setattr(
        installers,
        "_runtime_information",
        lambda: {
            "platform": "colab",
            "architecture": "x86_64",
            "ubuntu": "22.04",
            "glibc": "2.35",
            "python": "3.12",
            "accelerator": "cpu",
        },
    )


def test_rejects_non_colab(monkeypatch):
    monkeypatch.setattr(installers, "_in_colab", lambda: False)

    with pytest.raises(RuntimeError, match="Google Colab only"):
        installers.install_viennarna()


def test_viennarna_asset_and_url(colab_runtime):
    asset = installers._viennarna_asset_name("2.7.2-patch1")

    assert asset == (
        "ViennaRNA-2.7.2-patch1-colab-x86_64-ubuntu22.04-"
        "glibc2.35-python3.12-cpu.tgz"
    )

    assert installers._viennarna_release_url("2.7.2-patch1") == (
        "https://github.com/bussilab/ViennaRNA-patches/releases/download/"
        "v2.7.2-patch1/"
        + asset
    )


def test_viennarna_patch2_asset_url_and_patch_sequence(colab_runtime):
    asset = installers._viennarna_asset_name("2.7.2-patch2")

    assert asset == (
        "ViennaRNA-2.7.2-patch2-colab-x86_64-ubuntu22.04-"
        "glibc2.35-python3.12-cpu.tgz"
    )

    assert installers._viennarna_release_url("2.7.2-patch2") == (
        "https://github.com/bussilab/ViennaRNA-patches/releases/download/"
        "v2.7.2-patch2/"
        + asset
    )

    assert installers._viennarna_patch_files("2.7.2-patch2") == (
        "patch-viennarna-2.7.2-1.patch",
        "patch-viennarna-2.7.2-2.patch",
    )


def test_viennarna_default_is_patch2():
    assert installers._DEFAULT_VIENNARNA_VERSION == "2.7.2-patch2"


def test_invalid_viennarna_version():
    with pytest.raises(ValueError, match="2.7.2-patch1"):
        installers._parse_viennarna_version("2.7.2")


def test_safe_extraction(tmp_path):
    archive = tmp_path / "vienna.tgz"

    with tarfile.open(str(archive), "w:gz") as tar:
        directory = tarfile.TarInfo("ViennaRNA")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        tar.addfile(directory)

        contents = b"test executable\n"
        member = tarfile.TarInfo("ViennaRNA/bin/RNAfold")
        member.size = len(contents)
        member.mode = 0o755
        tar.addfile(member, io.BytesIO(contents))

    installers._safe_extract_viennarna(archive, tmp_path)

    assert (
        tmp_path / "ViennaRNA" / "bin" / "RNAfold"
    ).read_bytes() == b"test executable\n"


def test_unsafe_extraction_is_rejected(tmp_path):
    archive = tmp_path / "malicious.tgz"

    with tarfile.open(str(archive), "w:gz") as tar:
        contents = b"bad\n"
        member = tarfile.TarInfo("../outside")
        member.size = len(contents)
        tar.addfile(member, io.BytesIO(contents))

    with pytest.raises(RuntimeError, match="Unexpected member"):
        installers._safe_extract_viennarna(archive, tmp_path)
