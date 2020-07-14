import os
import unittest
import time

from bussilab.cli import cli
from bussilab.coretools import TestCase
from bussilab.coretools import cd

class TestCron(TestCase):
    def test_cron(self):
        with cd(os.path.dirname(os.path.abspath(__file__))):
            try:
                os.remove("cron.out")
            except FileNotFoundError:
                pass
            cli("cron --cron-file cron.yml --period 2 --max-times 2 --no-screen")
            self.assertEqualFile("cron.out")
            os.remove("cron.out")

    def  test_cron_screen(self):
        with cd(os.path.dirname(os.path.abspath(__file__))):
            try:
                os.remove("cron.out")
            except FileNotFoundError:
                pass
            cli("cron --cron-file cron.yml --period 2 --max-times 2 --detach")
            time.sleep(6)
            self.assertEqualFile("cron.out")
            os.remove("cron.out")

if __name__ == "__main__":
    unittest.main()
