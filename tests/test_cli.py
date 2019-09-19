import unittest
import bussilab.cli as cli

class Test(unittest.TestCase):
    def test_check(self):
        self.assertIsNone(cli.cli("check"))
        self.assertIsNone(cli.cli("check --import"))
        self.assertIsNone(cli.cli(["check", "--import"]))

        self.assertEqual(cli.cli("not_existing_command"), 2)
        self.assertEqual(cli.cli(["check", "not_existing_option"]), 2)

        with self.assertRaises(OSError):
            cli.cli("wham -b not_existing_file")

        self.assertEqual(cli.cli(["wham", "-b"]), 2) # here argument is missing

        with self.assertRaises(OSError):
            cli.cli(["wham", "-b", ""]) # "" is a file with no name

if __name__ == "__main__":
    unittest.main()
