import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from bussilab import reports


CPU = '%Cpu(s): 1.0 us, 2.0 sy, 0.0 ni,97.0 id, 0.0 wa'
DISKS = '/dev/a 100G 10G 90G 10% /scratch\n/dev/b 100G 10G 90G 10% /var'


REAL_RUN = subprocess.run


class TestReports(unittest.TestCase):
    def setUp(self):
        token_patch = patch('bussilab.reports.secrets.token_hex', return_value='TOKEN')
        token_patch.start()
        self.addCleanup(token_patch.stop)
        run_patch = patch('bussilab.reports.subprocess.run')
        self.run = run_patch.start()
        self.addCleanup(run_patch.stop)
        self.run.return_value = self.response()

    @staticmethod
    def response(gpu='', cpu=CPU, disks=DISKS):
        return SimpleNamespace(stdout='Login message\nTOKEN\n' + cpu + '\n' + disks + '\n' + gpu)

    def test_cpu_spacing_and_decimal_comma(self):
        for cpu in (CPU, CPU.replace('.', ',')):
            with self.subTest(cpu=cpu):
                self.assertEqual(reports._parse_cpu(cpu.split())['id'], 97.0)

    def test_cpu_missing_fields(self):
        with self.assertRaisesRegex(ValueError, 'missing CPU fields'):
            reports._parse_cpu(['unexpected output'])

    def test_without_gpu(self):
        self.assertEqual(reports.workstations(['host']),
                         'host CPU :sleeping: disk :smile:\n')
        args = self.run.call_args.args[0]
        self.assertEqual(args[:2], ['ssh', 'host'])
        self.assertIn('export LC_ALL=C;', args[2])
        self.assertIn('command -v nvidia-smi', args[2])
        self.assertIn('--query-gpu=utilization.gpu', args[2])
        self.assertIn('|| echo __BUSSILAB_GPU_ERROR__', args[2])
        self.assertTrue(self.run.call_args.kwargs['check'])

    def test_remote_command_locally_without_ssh(self):
        # Execute the generated shell command against fake remote utilities.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for utility in ('head', 'tail'):
                (root / utility).symlink_to(shutil.which(utility))
            scripts = {'top': "printf '%s\\n' header tasks " + repr(CPU),
                       'df': "printf '%s\\n' " + repr(DISKS.splitlines()[0])}
            for name, body in scripts.items():
                script = root / name
                script.write_text('#!/bin/sh\n' + body + '\n')
                script.chmod(0o755)

            def remote(args, **kwargs):
                return REAL_RUN(['/bin/sh', '-c', args[2]],
                                env=dict(os.environ, PATH=directory), **kwargs)

            self.run.side_effect = remote
            for body, expected in (
                (None, 'CPU :sleeping: disk :smile:'),
                ("printf '80\\n'", 'GPU :running: disk :smile:'),
                ('echo driver-failed >&2; exit 1', 'GPU :warning: disk :smile:'),
            ):
                with self.subTest(body=body):
                    if body is not None:
                        script = root / 'nvidia-smi'
                        script.write_text('#!/bin/sh\n' + body + '\n')
                        script.chmod(0o755)
                    self.assertIn(expected, reports.workstations(['host']))

    def test_legacy_nvidia_setting_is_ignored(self):
        self.run.return_value = self.response('0\n30\n90\n')
        for setting in (False, 'False', 'false', True):
            with self.subTest(setting=setting):
                result = reports.workstations([{'url': 'host', 'nvidia': setting}])
                self.assertIn('GPU :sleeping: :walking: :running:', result)
                self.assertIn('command -v nvidia-smi', self.run.call_args.args[0][2])

    def test_gpu_failure_and_unavailable_utilization(self):
        for gpu in ('__BUSSILAB_GPU_ERROR__\n', 'N/A\n', '[N/A]\n'):
            with self.subTest(gpu=gpu):
                self.run.return_value = self.response(gpu)
                result = reports.workstations(['host'])
                self.assertIn('GPU :warning: disk :smile:', result)
                self.assertNotIn('Error connecting', result)

    def test_parsing_errors_have_diagnostics(self):
        for response, diagnostic in (
            (SimpleNamespace(stdout='no marker\n'), 'missing report start marker'),
            (SimpleNamespace(stdout='TOKEN\n'), 'incomplete CPU/disk output'),
            (self.response(cpu='invalid cpu'), 'missing CPU fields'),
            (self.response(disks='invalid\ninvalid'), 'list index out of range'),
            (self.response(gpu='invalid\n'), 'could not convert string to float'),
        ):
            with self.subTest(diagnostic=diagnostic):
                self.run.return_value = response
                result = reports.workstations(['host'])
                self.assertIn('Error parsing report: ' + diagnostic, result)
                self.assertNotIn('Error connecting', result)

    def test_connection_error_preserves_stderr_and_continues(self):
        self.run.side_effect = [
            subprocess.CalledProcessError(255, ['ssh', 'bad'], stderr='Permission denied'),
            self.response(),
        ]
        result = reports.workstations(['bad', 'good'])
        self.assertIn('Error connecting:', result)
        self.assertIn('Permission denied', result)
        self.assertIn('good CPU :sleeping: disk :smile:', result)

    def test_timeout_and_missing_ssh(self):
        for exc, diagnostic in (
            (subprocess.TimeoutExpired(['ssh'], 60), 'Timeout'),
            (FileNotFoundError('ssh missing'), 'Error connecting: ssh missing'),
        ):
            with self.subTest(diagnostic=diagnostic):
                self.run.side_effect = exc
                self.assertIn(diagnostic, reports.workstations(['host']))

    def test_long_report_and_quoted_paths(self):
        result = reports.workstations([{'url': 'host', 'name': 'Workstation',
                                       'disk': '/scratch space', 'tmpdisk': '/var space'}],
                                     short=False)
        self.assertTrue(result.startswith('Workstation CPU'))
        self.assertIn(CPU, result)
        self.assertNotIn('Login message', result)
        command = self.run.call_args.args[0][2]
        self.assertIn("df -Ph '/scratch space'", command)
        self.assertIn("df -Ph '/var space'", command)

    def test_config_default(self):
        with patch.object(reports.coretools, 'config', return_value={'workstations': ['host']}):
            self.assertIn('host CPU', reports.workstations())


if __name__ == '__main__':
    unittest.main()
