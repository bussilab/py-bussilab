"""Remote Jupyter tests without network access or browser launches."""
from contextlib import redirect_stdout
from io import StringIO
import json
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from bussilab import jremote


URL = "http://localhost:8888/lab?token=not-just-hex&next=%2Ftree#section"


def result(code=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=code, stdout=stdout, stderr=stderr, args=["ssh"])


class TestJremote(unittest.TestCase):
    def setUp(self):
        self.output = StringIO()
        capture = redirect_stdout(self.output)
        capture.__enter__()
        self.addCleanup(capture.__exit__, None, None, None)

    def test_json_discovery_and_deduplication(self):
        record = json.dumps({"url": "http://localhost:8888/lab", "token": "a+b"})
        with patch.object(jremote.subprocess, "run", return_value=result(
                stdout="Shell banner\n" + record)) as run:
            urls = jremote._list_servers("host", "module load python; python3")
        self.assertEqual(urls, ["http://localhost:8888/lab?token=a%2Bb"])
        self.assertIn("module load python; python3 -m jupyter server list --json",
                      run.call_args_list[0].args[0][-1])

    def test_legacy_notebook_survives_missing_lab(self):
        responses = [result(1), result(1), result(1),
                     result(stdout="Currently running servers:\n" + URL + " :: /work"),
                     result(1)]
        with patch.object(jremote.subprocess, "run", side_effect=responses):
            self.assertEqual(jremote._list_servers("host", "python"), [URL])

    def test_discovery_errors(self):
        with patch.object(jremote.subprocess, "run", return_value=result(255)):
            with self.assertRaises(subprocess.CalledProcessError):
                jremote._list_servers("host", "python")
        with patch.object(jremote.subprocess, "run", return_value=result(1)):
            with self.assertRaisesRegex(RuntimeError, "remote Python"):
                jremote._list_servers("host", "python")

    def test_empty_results_eof_and_invalid_indices(self):
        with patch.object(jremote, "_list_servers", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "No running"):
                jremote.remote("host")
            jremote.remote("host", list_only=True)
        with patch.object(jremote, "_list_servers", return_value=[URL, URL]):
            for index in (-1, 3):
                with self.assertRaises(ValueError):
                    jremote.remote("host", index=index)
            with patch.object(jremote.sys, "stdin", StringIO("")):
                with self.assertRaises(EOFError):
                    jremote.remote("host")
            with patch.object(jremote.sys, "stdin", StringIO("bad\n3\n2\n")):
                jremote.remote("host", dry_run=True, port=9999, open_cmd="browser")

    def test_url_rewrite_preserves_path_query_and_fragment(self):
        with patch.object(jremote.subprocess, "Popen") as popen:
            jremote.remote("host", server_url=URL, dry_run=True, port=9999,
                           open_cmd='browser --profile "work profile"')
        popen.assert_not_called()
        self.assertIn("localhost:9999/lab?token=not-just-hex&next=%2Ftree#section",
                      self.output.getvalue())
        for url in ("http://127.0.0.1:8888/", "https://localhost/", "http://[::1]:8888/base/"):
            jremote._server_url(url)
        for url in ("file:///tmp/a", "http://remote:8888/", "http://user@localhost/",
                    "http://localhost:99999/", "http://localhost:0/"):
            with self.assertRaises(ValueError):
                jremote._server_url(url)

    def test_tunnel_failure_and_timeout(self):
        ssh = MagicMock()
        ssh.poll.return_value = 255
        with self.assertRaisesRegex(RuntimeError, "255"):
            jremote._wait_for_tunnel(ssh, 9999)
        ssh.poll.return_value = None
        with patch.object(jremote.time, "monotonic", side_effect=[0, 31]):
            with self.assertRaises(TimeoutError):
                jremote._wait_for_tunnel(ssh, 9999)

    def test_tunnel_waits_for_listener(self):
        ssh = MagicMock()
        ssh.poll.return_value = None
        connection = MagicMock()
        with patch.object(jremote.socket, "create_connection",
                          side_effect=[ConnectionRefusedError(), connection]) as connect, \
                patch.object(jremote.time, "sleep"):
            jremote._wait_for_tunnel(ssh, 9999)
        self.assertEqual(connect.call_count, 2)
        connection.__exit__.assert_called_once()

    def test_successful_ipv6_connection(self):
        ssh = MagicMock()
        ssh.poll.return_value = 0
        ssh.wait.return_value = 0
        events = []
        with patch.object(jremote.subprocess, "Popen", return_value=ssh) as popen, \
                patch.object(jremote, "_wait_for_tunnel",
                             side_effect=lambda *args, **kwargs: events.append("ready")) as ready, \
                patch.object(jremote.subprocess, "run",
                             side_effect=lambda *args, **kwargs: events.append("browser")) as run:
            jremote.remote("host", server_url="https://[::1]:8888/lab?token=a",
                           port=9999, open_cmd='browser --profile "work profile"')
        self.assertEqual(events, ["ready", "browser"])
        self.assertIn("[::1]:9999:[::1]:8888", popen.call_args.args[0])
        ready.assert_called_once_with(ssh, 9999, host="::1")
        run.assert_called_once_with(["browser", "--profile", "work profile",
                                     "https://[::1]:9999/lab?token=a"], check=True)
        ssh.terminate.assert_not_called()

    def test_browser_failure_cleans_up_tunnel(self):
        ssh = MagicMock()
        ssh.poll.return_value = None
        with patch.object(jremote.subprocess, "Popen", return_value=ssh) as popen, \
                patch.object(jremote, "_wait_for_tunnel") as ready, \
                patch.object(jremote.subprocess, "run", side_effect=OSError("browser missing")):
            with self.assertRaises(OSError):
                jremote.remote("host", server_url=URL, port=9999, open_cmd="browser")
        ready.assert_called_once_with(ssh, 9999)
        self.assertIn("ExitOnForwardFailure=yes", popen.call_args.args[0])
        ssh.terminate.assert_called_once()
        ssh.wait.assert_called_once_with(timeout=5)

    def test_interrupt_kills_unresponsive_tunnel(self):
        ssh = MagicMock()
        ssh.poll.return_value = None
        ssh.wait.side_effect = [subprocess.TimeoutExpired("ssh", 5), 0]
        with patch.object(jremote.subprocess, "Popen", return_value=ssh), \
                patch.object(jremote, "_wait_for_tunnel", side_effect=KeyboardInterrupt):
            jremote.remote("host", server_url=URL, port=9999, open_cmd="browser")
        ssh.terminate.assert_called_once()
        ssh.kill.assert_called_once()

    def test_launch_failure_propagates_and_dry_run_does_not_launch(self):
        with patch.object(jremote.subprocess, "run", side_effect=
                          subprocess.CalledProcessError(1, ["jupyter"])):
            with self.assertRaises(subprocess.CalledProcessError):
                jremote.run_server(no_screen=True, port=8888)
        with patch.object(jremote.subprocess, "run") as run:
            jremote.run_server(dry_run=True, port=8888, lab=True, detach=True)
            jremote.remote("host", dry_run=True, list_only=True)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
