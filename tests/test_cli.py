import unittest
import bussilab.cli as cli

class Test(unittest.TestCase):
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

if __name__ == "__main__":
    unittest.main()
