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
                         '| Workstation | CPU | GPU | Disk | Temp |\n'
                         '| --- | --- | --- | --- | --- |\n'
                         '| 🟢 host | 3% | — | 10% | 10% |\n')
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
                (None, '| 3% | — | 10% |'),
                ("printf '80\\n'", '| 80% | 10% |'),
                ('echo driver-failed >&2; exit 1', '| unavailable ⚠️ | 10% |'),
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
                self.assertIn('| 0%, 30%, 90% |', result)
                self.assertIn('command -v nvidia-smi', self.run.call_args.args[0][2])

    def test_gpu_failure_and_unavailable_utilization(self):
        for gpu in ('__BUSSILAB_GPU_ERROR__\n', 'N/A\n', '[N/A]\n'):
            with self.subTest(gpu=gpu):
                self.run.return_value = self.response(gpu)
                result = reports.workstations(['host'])
                self.assertIn('| unavailable ⚠️ | 10% |', result)
                self.assertNotIn('Error connecting', result)

    def test_parsing_errors_have_diagnostics(self):
        for response, diagnostic in (
            (SimpleNamespace(stdout='no marker\n'), 'missing report start marker'),
            (SimpleNamespace(stdout='TOKEN\n'), 'incomplete CPU/disk output'),
            (self.response(cpu='invalid cpu'), 'missing CPU fields'),
            (self.response(disks='invalid\ninvalid'), 'missing disk usage percentage'),
            (self.response(gpu='invalid\n'), 'could not convert string to float'),
        ):
            with self.subTest(diagnostic=diagnostic):
                self.run.return_value = response
                result = reports.workstations(['host'])
                self.assertIn('Error parsing report: ' + diagnostic, result)
                self.assertIn('| 🟢 host |', result)
                self.assertNotIn('Error connecting', result)

    def test_connection_error_preserves_stderr_and_continues(self):
        self.run.side_effect = [
            subprocess.CalledProcessError(255, ['ssh', 'bad'], stderr='Permission denied'),
            self.response(),
        ]
        result = reports.workstations(['bad', 'good'])
        self.assertIn('Error connecting:', result)
        self.assertIn('Permission denied', result)
        self.assertIn('| 🔴 bad |', result)
        self.assertIn('| 🟢 good | 3% | — | 10% |', result)

    def test_timeout_and_missing_ssh(self):
        for exc, diagnostic in (
            (subprocess.TimeoutExpired(['ssh'], 60), 'Timeout'),
            (FileNotFoundError('ssh missing'), 'Error connecting: ssh missing'),
        ):
            with self.subTest(diagnostic=diagnostic):
                self.run.side_effect = exc
                result = reports.workstations(['host'])
                self.assertIn(diagnostic, result)
                self.assertIn('| 🔴 host |', result)

    def test_long_report_and_quoted_paths(self):
        result = reports.workstations([{'url': 'host', 'name': 'Workstation',
                                       'disk': '/scratch space', 'tmpdisk': '/var space'}],
                                     short=False)
        self.assertIn('| 🟢 Workstation | 3% |', result)
        self.assertIn('### Workstation\n\n```text\n', result)
        self.assertIn(CPU, result)
        self.assertNotIn('Login message', result)
        command = self.run.call_args.args[0][2]
        self.assertIn("df -Ph '/scratch space'", command)
        self.assertIn("df -Ph '/var space'", command)

    def test_table_escapes_names_paths_and_multiline_errors(self):
        self.run.side_effect = subprocess.CalledProcessError(
            255, ['ssh', 'host'], stderr='first | error\nsecond error')
        result = reports.workstations([{'url': 'host', 'name': 'a|b_*'}])
        self.assertEqual(len(result.splitlines()), 3)
        self.assertIn(r'a\|b\_\*', result)
        self.assertIn(r'first \| error second error', result)

    def test_disk_paths_with_spaces(self):
        self.run.return_value = self.response(disks=DISKS.replace('/scratch', '/scratch space'))
        result = reports.workstations([{'url': 'host', 'disk': '/scratch space'}])
        self.assertIn('| 10% | 10% |', result)
        self.assertNotIn('/scratch', result)
        self.assertNotIn('/var', result)

    def test_compact_table_headers_and_percentages(self):
        self.run.return_value = self.response(cpu=CPU.replace('97.0 id', '50.0 id'),
                                             disks=DISKS.replace('10%', '95%'))
        result = reports.workstations(['host'])
        self.assertEqual(result.splitlines()[0],
                         '| Workstation | CPU | GPU | Disk | Temp |')
        self.assertIn('| 50% ⚠️ |', result)
        self.assertIn('95% ⚠️', result)
        self.assertIn('95% ⚠️', result)
        self.assertNotIn('Notes', result)

    def test_warning_symbol_boundaries(self):
        for disk, temp, cpu_idle, expected_disk, expected_temp, expected_cpu in (
            (89, 70, 79, '89%', '70%', '21%'),
            (90, 71, 78, '90% ⚠️', '71% ⚠️', '22% ⚠️'),
            (99, 99, 50, '99% ⚠️', '99% ⚠️', '50% ⚠️'),
        ):
            with self.subTest(disk=disk, temp=temp, idle=cpu_idle):
                disks = (DISKS.replace('10% /scratch', str(disk) + '% /scratch')
                         .replace('10% /var', str(temp) + '% /var'))
                self.run.return_value = self.response(
                    cpu=CPU.replace('97.0 id', str(cpu_idle) + ' id'), disks=disks)
                result = reports.workstations(['host'])
                self.assertIn('| ' + expected_cpu + ' |', result)
                self.assertIn('| ' + expected_disk + ' |', result)
                self.assertIn('| ' + expected_temp + ' |', result)

    def test_only_unavailable_gpu_has_warning(self):
        self.run.return_value = self.response('85\nN/A\n12\n')
        result = reports.workstations(['host'])
        self.assertIn('| 85%, unavailable ⚠️, 12% |', result)

    def test_long_report_fences_raw_backticks_after_all_rows(self):
        # Backticks in a mount point must not close the raw-output fence.
        self.run.return_value = self.response(disks=DISKS.replace('/var', '/var```'))
        result = reports.workstations(['one', 'two'], short=False)
        self.assertLess(result.index('| 🟢 two |'), result.index('### one'))
        self.assertIn('````text\n', result)
        self.assertIn('/var```\n````', result)

    def test_config_default(self):
        with patch.object(reports.coretools, 'config', return_value={'workstations': ['host']}):
            self.assertIn('| 🟢 host | 3% |', reports.workstations())


if __name__ == '__main__':
    unittest.main()
