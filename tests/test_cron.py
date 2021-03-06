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
            try:
                os.remove("cron_screen.out")
            except FileNotFoundError:
                pass
            try:
                os.remove("cron_screen2.out")
            except FileNotFoundError:
                pass
            # this is asynchronous
            cli("cron --cron-file cron_screen.yml --period 2 --max-times 2 --detach")
            now=time.time()
            # this is synchronous, and will take some time
            cli("cron --cron-file cron.yml --period 2 --max-times 2 --no-screen")
            # make sure to give enough time to the asynchronous test
            time.sleep(7-(time.time()-now))
            self.assertEqualFile("cron.out")
            self.assertEqualFile("cron_screen.out")
            self.assertEqualFile("cron_screen2.out")
            os.remove("cron.out")
            os.remove("cron_screen.out")
            os.remove("cron_screen2.out")

if __name__ == "__main__":
    unittest.main()
