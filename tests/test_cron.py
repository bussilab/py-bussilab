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
            try:
                os.remove("screenlog.0")
            except FileNotFoundError:
                pass
            try:
                os.remove("cron_reboot.out")
            except FileNotFoundError:
                pass
            try:
                os.remove("cron_reboot_unsorted.out")
            except FileNotFoundError:
                pass
            # this is asynchronous
            cli("cron --cron-file cron_screen.yml --period 2 --max-times 2 --detach --screen-log screenlog.0 --unique")
            # the second instance will not be run (--unique)
            cli("cron --cron-file cron_screen.yml --period 2 --max-times 2 --detach --screen-log screenlog.0 --unique")
            # also this is asynchronoous
            cli("cron --cron-file cron_reboot.yml --max-times 4 --detach --screen-log screenlog.0")
            now=time.time()
            # this is synchronous, and will take some time
            cli("cron --cron-file cron.yml --max-times 2 --no-screen")
            # make sure to give enough time to the asynchronous test
            time.sleep(9-(time.time()-now))
            # this is for debugging, should be shown only if there is an error later
            os.system("ls -ltr")
            os.system("echo == screenlog ==")
            os.system("cat screenlog.0")
            os.system("echo == end screenlog ==")
            self.assertEqualFile("cron.out")
            self.assertEqualFile("cron_screen.out")
            self.assertEqualFile("cron_screen2.out")
            self.assertEqualFile("cron_reboot.out")
            with open("cron_reboot_unsorted.out") as file:
                lines = [line.rstrip() for line in file]
                lines.sort()
                self.assertEqual(lines,["C0","C0","C1","C1"])
            os.remove("cron.out")
            os.remove("cron_screen.out")
            os.remove("cron_screen2.out")
            os.remove("cron_reboot.out")
            os.remove("cron_reboot_unsorted.out")
            os.remove("screenlog.0")

if __name__ == "__main__":
    unittest.main()
