import os
import unittest
import time
import subprocess

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
            # Detached Screen startup is asynchronous: wait until the session
            # exists before checking that --unique refuses another instance.
            sockname = "bussilab-test-cron-{}".format(os.getpid())
            command = ("cron --cron-file cron_screen.yml --period 2 --max-times 2"
                       " --detach --screen-log screenlog.0 --unique --sockname " + sockname)
            cli(command)
            deadline = time.monotonic() + 5
            while True:
                sessions = subprocess.run(
                    ["screen", "-ls"], stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, universal_newlines=True).stdout
                if any(line.split() and line.split()[0].partition(".")[2] == sockname
                       for line in sessions.splitlines()):
                    break
                if time.monotonic() >= deadline:
                    self.fail("Detached Screen session did not start: " + sessions)
                time.sleep(0.05)
            # the second instance will not be run (--unique)
            cli(command)
            # also this is asynchronoous
            cli("cron --cron-file cron_reboot.yml --max-times 4 --detach --screen-log screenlog.0")
            now=time.time()
            # this is synchronous, and will take some time
            cli("cron --cron-file cron.yml --max-times 2 --no-screen")
            # make sure to give enough time to the asynchronous test
            time.sleep(max(0, 9-(time.time()-now)))
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
