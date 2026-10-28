"""Exercise pip helpers without modifying the test environment."""

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bussilab import cli, pip


class TestPip(unittest.TestCase):
    def test_install_preserves_pins_and_interpreter(self):
        with patch.object(pip.subprocess, "check_call") as call:
            pip.install(["numpy==1.24.4", "scipy==1.10.1"], timeout=30)
        call.assert_called_once_with(
            [sys.executable, "-m", "pip", "install", "numpy==1.24.4", "scipy==1.10.1"],
            timeout=30)

    def test_upgrade_all_warns_and_supports_current_metadata(self):
        distributions = [SimpleNamespace(metadata={"Name": name})
                         for name in ("numpy", "scipy", "numpy", None)]
        with patch.object(pip.metadata, "distributions", return_value=distributions), \
                patch.object(pip, "install") as install:
            with self.assertWarnsRegex(FutureWarning, "pinned requirements"):
                pip.upgrade_all(user=True, timeout=20)
        install.assert_called_once_with(["numpy", "scipy"], user=True,
                                        upgrade=True, timeout=20)

    def test_cli_retains_legacy_command_with_visible_warning(self):
        distributions = [SimpleNamespace(metadata={"Name": "numpy"})]
        with patch.object(pip.metadata, "distributions", return_value=distributions), \
                patch.object(pip, "install") as install:
            with self.assertWarnsRegex(FutureWarning, "deprecated"):
                cli.cli(["pip_upgrade_all", "--user"])
        install.assert_called_once_with(["numpy"], user=True, upgrade=True, timeout=None)

    def test_empty_environment_does_not_invoke_pip(self):
        with patch.object(pip.metadata, "distributions", return_value=[]), \
                patch.object(pip, "install") as install:
            with self.assertWarns(FutureWarning):
                pip.upgrade_all()
        install.assert_not_called()

    def test_upgrade_self_keeps_cron_compatibility(self):
        with patch.object(pip, "install") as install:
            pip.upgrade_self(user=True, timeout=15)
        install.assert_called_once_with("bussilab", user=True, timeout=15, upgrade=True)


if __name__ == "__main__":
    unittest.main()
