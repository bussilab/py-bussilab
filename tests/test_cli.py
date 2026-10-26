import unittest
import bussilab.cli as cli

class Test(unittest.TestCase):
    def test_invalid_command_exit_code(self):
        import subprocess
        import sys
        result=subprocess.run([sys.executable, "-m", "bussilab", "not_existing_command"],
                              capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_wham_one_dimensional_file(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import numpy as np
        from bussilab.wham import wham

        with TemporaryDirectory() as directory:
            path=Path(directory)/"bias.dat"
            for bias in (np.array([[1.0], [2.0]]),
                         np.array([[1.0, 2.0]])):
                np.savetxt(path, bias)
                output=StringIO()
                with redirect_stdout(output):
                    cli.cli(["wham", "-b", str(path)])
                log_weights=[float(line) for line in output.getvalue().splitlines()
                             if not line.startswith("#")]
                np.testing.assert_allclose(log_weights, wham(bias).logW)

            np.savetxt(path, np.array([[1.0], [2.0]]))
            with self.assertRaisesRegex(ValueError, "at least one bias column"):
                cli.cli(["wham", "-b", str(path), "--use-frame-weight"])
            self.assertEqual(cli.cli(["wham", "-b", str(path), "--traj-weight"],
                                     throws_on_parser_errors=False), 2)

    def test_check(self):
        self.assertIsNone(cli.cli("check"))
        self.assertIsNone(cli.cli("check --import"))
        self.assertIsNone(cli.cli(["check", "--import"]))
        self.assertIsNone(cli.cli("--version"))
        self.assertIsNone(cli.cli("list"))
        self.assertIsNone(cli.cli(""))

        self.assertEqual(cli.cli("not_existing_command",throws_on_parser_errors=False), 2)
        self.assertEqual(cli.cli(["check","not_existing_option"],throws_on_parser_errors=False), 2)

        with self.assertRaises(TypeError):
            cli.cli("not_existing_command")

        with self.assertRaises(TypeError):
            cli.cli(["check","not_existing_option"])

        with self.assertRaises(OSError):
            cli.cli("wham -b not_existing_file")

        self.assertEqual(cli.cli(["wham", "-b"], throws_on_parser_errors=False), 2)  # here argument is missing

        with self.assertRaises(TypeError):
            cli.cli(["wham", "-b"])  # here argument is missing

        with self.assertRaises(OSError):
            cli.cli(["wham", "-b", ""])  # "" is a file with no name

        with self.assertRaises(NotImplementedError):
            cli._list_submodules()

        self.assertIsNone(cli.cli("required"))
        self.assertIsNone(cli.cli("required --conda"))
        self.assertIsNone(cli.cli("required --macports --pyver 38"))

    def test_decorator(self):
        @cli.command("x")
        def _x():
            pass
        with self.assertRaises(NotImplementedError):
            _x()
        self.assertIsNone(cli.cli("x"))
        self.assertIsNone(cli.cli("x"))

if __name__ == "__main__":
    unittest.main()
