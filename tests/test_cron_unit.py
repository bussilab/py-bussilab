"""Deterministic cron tests, independent of Screen and real scheduling waits."""
import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import yaml

from bussilab import cron


class TestCronUnit(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'cron.yml'
        self.write_config()
        self.output = io.StringIO()
        capture = redirect_stdout(self.output)
        capture.__enter__()
        self.addCleanup(capture.__exit__, None, None, None)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop('BUSSILAB_CRON_SCREEN_ARGS', None)

    def write_config(self, period=10, script='pass'):
        self.path.write_text(yaml.safe_dump({'cron': {
            'period': period, 'steps': [script]}}))

    def test_read_list_mapping_default_and_period_override(self):
        self.path.write_text('cron:\n  - pass\n')
        self.assertEqual(cron._load_config(str(self.path), None),
                         {'steps': ['pass'], 'period': 3600})
        self.write_config(period='bad')
        self.assertEqual(cron._load_config(str(self.path), 20)['period'], 20)
        with patch.object(cron.coretools, 'config', return_value={'cron': ['pass']}):
            self.assertEqual(cron._load_config('', None)['period'], 3600)

    def test_invalid_config_is_rejected_and_logged_at_startup(self):
        for contents in ('', '[]', 'other: value', 'cron: null', 'cron: {}',
                         'cron: {steps: pass}', 'cron: [',
                         'cron: {period: 0, steps: []}',
                         'cron: {period: -1, steps: []}',
                         'cron: {period: bad, steps: []}'):
            with self.subTest(contents=contents):
                self.path.write_text(contents)
                with self.assertRaises((ValueError, yaml.YAMLError)):
                    cron.cron(cron_file=str(self.path), quick_start=True, max_times=1)
        log = self.output.getvalue()
        self.assertIn(str(self.path), log)
        self.assertIn('ERROR reading cron configuration', log)
        self.assertIn('cannot start without a valid configuration', log)

    def test_invalid_explicit_period_is_rejected(self):
        for period in (0, -2, 1.5, True):
            with self.subTest(period=period), self.assertRaisesRegex(ValueError, 'positive integer'):
                cron._load_config(str(self.path), period)

    def test_reload_keeps_last_valid_config_on_read_and_schema_errors(self):
        previous = cron._load_config(str(self.path), None)
        for contents in ('cron: [', 'cron: {period: 0, steps: []}', '',
                         'cron: {steps: not-a-list}'):
            with self.subTest(contents=contents):
                self.path.write_text(contents)
                self.assertIs(cron._reload_config(str(self.path), None, previous), previous)
        self.path.unlink()
        self.assertIs(cron._reload_config(str(self.path), None, previous), previous)
        self.assertIn('FileNotFoundError', self.output.getvalue())
        self.assertIn('keeping the last valid configuration', self.output.getvalue())
        self.write_config(period=20, script='new_script')
        current = cron._reload_config(str(self.path), None, previous)
        self.assertEqual(current, {'period': 20, 'steps': ['new_script']})
        self.assertEqual(previous, {'period': 10, 'steps': ['pass']})

    def test_scheduler_survives_broken_edit_then_uses_corrected_config(self):
        edits = ['cron: [', 'cron: {period: 10, steps: [updated]}']

        def edit(_seconds):
            self.path.write_text(edits.pop(0))

        with patch.object(cron.time, 'sleep', side_effect=edit), \
             patch.object(cron, '_time_to_next_event', return_value=(1, 10)), \
             patch.object(cron, '_run') as run:
            cron.cron(cron_file=str(self.path), max_times=2)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].kwargs['config']['steps'], ['pass'])
        self.assertEqual(run.call_args_list[1].kwargs['config']['steps'], ['updated'])
        self.assertIn('keeping the last valid configuration', self.output.getvalue())

    def test_reload_recovery_runs_real_jobs_and_writes_errors_to_log(self):
        marker = Path(self.directory.name) / 'events'
        logfile = Path(self.directory.name) / 'cron.log'

        def script(label):
            return ('from pathlib import Path; '
                    'p = Path({!r}); '
                    'p.open("a").write({!r})').format(str(marker), label + "\n")

        self.write_config(script=script('old'))
        edits = iter(('broken', 'missing', 'fixed'))

        def edit(_seconds):
            action = next(edits)
            if action == 'broken':
                self.path.write_text('cron: [')
            elif action == 'missing':
                self.path.unlink()
            else:
                self.write_config(script=script('new'))

        # Replace cron's clock reference so subprocess.wait keeps real sleeps.
        with logfile.open('w') as log, redirect_stdout(log), \
             patch.object(cron, 'time') as clock, \
             patch.object(cron, '_time_to_next_event', return_value=(2, 10)):
            clock.sleep.side_effect = edit
            cron.cron(cron_file=str(self.path), max_times=3)
        self.assertEqual(marker.read_text().splitlines(), ['old', 'old', 'new'])
        log = logfile.read_text()
        self.assertIn('ParserError', log)
        self.assertIn('FileNotFoundError', log)
        self.assertIn(str(self.path), log)
        self.assertEqual(log.count('keeping the last valid configuration'), 2)

    def test_period_change_reschedules_without_mixing_event_and_period(self):
        edits = [20, 20]

        def edit(_seconds):
            self.write_config(period=edits.pop(0))

        with patch.object(cron.time, 'sleep', side_effect=edit), \
             patch.object(cron, '_time_to_next_event', side_effect=[(1, 10), (2, 20)]) as schedule, \
             patch.object(cron, '_run') as run:
            cron.cron(cron_file=str(self.path), max_times=1)
        self.assertEqual([call.args[0] for call in schedule.call_args_list], [10, 20])
        self.assertEqual(run.call_args.args[1:4], (20, 20, 0))
        self.assertEqual(run.call_args.kwargs['config']['period'], 20)
        self.assertIn('period changed', self.output.getvalue())

    def test_quick_start_and_iteration_limit(self):
        with patch.object(cron, '_run') as run, patch.object(cron.time, 'sleep') as sleep:
            cron.cron(cron_file=str(self.path), quick_start=True,
                      quick_start_skip_steps=2, quick_start_event=50, max_times=1)
        self.assertEqual(run.call_args.args[1:], (10, 50, 0, 2))
        sleep.assert_not_called()

    def test_nonzero_exit_timeout_and_launch_error_all_continue(self):
        failures = [subprocess.CalledProcessError(7, ['python']),
                    subprocess.TimeoutExpired(['python'], 5),
                    FileNotFoundError('bash not found')]
        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                config = {'steps': ['first', 'second']}
                with patch.object(cron, '_time_to_next_event', return_value=(10, 10)), \
                     patch.object(cron.subprocess, 'run', side_effect=[failure, None]) as run:
                    cron._run('', 10, 10, 0, config=config)
                self.assertEqual(run.call_count, 2)
                self.assertEqual(run.call_args.args[0][-1], 'second')
                self.assertTrue(run.call_args.kwargs['check'])
                self.assertNotIn('start_new_session', run.call_args.kwargs)
        log = self.output.getvalue()
        self.assertIn('event=10 step=0', log)
        self.assertIn('continuing to next step', log)
        self.assertIn('CalledProcessError', log)
        self.assertIn('TimeoutExpired', log)
        self.assertIn('Traceback', log)

    def test_bad_steps_do_not_block_independent_steps(self):
        bad_steps = [None, {'type': 'unknown'}, {'type': 'python'},
                     {'delay': '1', 'script': 'pass'},
                     {'skip': '0', 'script': 'pass'},
                     {'skip': 'missing_name', 'script': 'pass'}]
        for bad in bad_steps:
            with self.subTest(bad=bad):
                with patch.object(cron, '_time_to_next_event', return_value=(10, 10)), \
                     patch.object(cron.subprocess, 'run') as run:
                    cron._run('', 10, 10, 0, config={'steps': [bad, 'next']})
                run.assert_called_once()
                self.assertEqual(run.call_args.args[0][-1], 'next')

    def test_skip_delay_expressions_resume_and_no_config_mutation(self):
        config = {'steps': ['skipped', {'skip': '2*3', 'delay': '1', 'script': 'run'},
                            {'skip': '2*3', 'delay': '0', 'script': 'skip'},
                            {'type': 'bash', 'script': 'echo done'}]}
        original = copy.deepcopy(config)
        with patch.object(cron, '_time_to_next_event', return_value=(10, 10)), \
             patch.object(cron.subprocess, 'run') as run:
            cron._run('', 10, 10, 0, skip_steps=1, config=config)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0][-1], 'run')
        self.assertEqual(run.call_args.args[0][:4], ['bash', '--noprofile', '--norc', '-c'])
        self.assertEqual(config, original)

    def test_skip_delay_division_expressions_remain_supported(self):
        config = {'steps': [{'skip': '60/period', 'delay': '2/2', 'script': 'run'}]}
        with patch.object(cron, '_time_to_next_event', return_value=(10, 10)), \
             patch.object(cron.subprocess, 'run') as run:
            cron._run('', 10, 10, 0, config=config)
        run.assert_called_once()

    def test_reboot_preserves_remaining_iterations_event_and_resume_step(self):
        arguments = {'screen_cmd': 'screen', 'max_times': 5, 'cron_file': str(self.path)}
        with patch.dict(os.environ, {'BUSSILAB_CRON_SCREEN_ARGS':
                                     json.dumps({'arguments': arguments})}), \
             patch.object(cron, '_screen_version', return_value=(4, 9)), \
             patch.object(cron.subprocess, 'call', return_value=0), \
             patch.object(cron, 'cron') as launch:
            result = cron._reboot(iterations=2, skip_steps=3, event=40)
        self.assertIsInstance(result, cron._reboot_now)
        self.assertEqual(launch.call_args.kwargs['max_times'], 3)
        self.assertEqual(launch.call_args.kwargs['quick_start_event'], 40)
        self.assertEqual(launch.call_args.kwargs['quick_start_skip_steps'], 3)
        self.assertTrue(launch.call_args.kwargs['quick_start'])
        self.assertTrue(launch.call_args.kwargs['window'])

    def test_reboot_handoff_stops_current_event_and_preserves_position(self):
        with patch.object(cron, '_reboot', return_value=cron._reboot_now()) as reboot, \
             patch.object(cron.subprocess, 'run') as run:
            result = cron._run('', 10, 30, 2, config={'steps': [{'type': 'reboot'}, 'next']})
        self.assertIsInstance(result, cron._reboot_now)
        reboot.assert_called_once_with(iterations=2, skip_steps=1, event=30)
        run.assert_not_called()

    def test_reboot_and_selfupdate_failures_continue(self):
        for kind in ('reboot', 'selfupdate'):
            with self.subTest(kind=kind):
                with patch.object(cron, '_reboot', side_effect=RuntimeError('reboot failed')), \
                     patch.object(cron.pip, 'upgrade_self', side_effect=RuntimeError('update failed')), \
                     patch.object(cron, '_time_to_next_event', return_value=(10, 10)), \
                     patch.object(cron.subprocess, 'run') as run:
                    cron._run('', 10, 10, 0, config={'steps': [{'type': kind}, 'next']})
                run.assert_called_once()
                self.assertEqual(run.call_args.args[0][-1], 'next')

    def test_fractional_seconds_and_week_alignment(self):
        local = time.struct_time((2026, 10, 28, 12, 0, 0, 2, 301, -1))
        for timestamp, fraction in ((1000.0, 0), (1000.1, .1), (1000.005, .005),
                                    (1000.9999, .9999)):
            with self.subTest(timestamp=timestamp):
                with patch.object(cron.time, 'time', return_value=timestamp), \
                     patch.object(cron.time, 'localtime', return_value=local):
                    wait, event = cron._time_to_next_event(60)
                self.assertAlmostEqual(wait, 60-fraction)
                self.assertEqual(event, 2*86400 + 12*3600 + 60)

    def test_real_failed_and_timed_out_children_allow_next_step(self):
        marker = Path(self.directory.name) / 'ran'
        for first in ('raise SystemExit(7)', 'import time; time.sleep(10)'):
            with self.subTest(first=first):
                marker.unlink(missing_ok=True)
                script = 'from pathlib import Path; Path({!r}).write_text("ok")'.format(str(marker))
                with patch.object(cron, '_time_to_next_event', return_value=(1, 10)):
                    cron._run('', 10, 10, 0, config={'steps': [first, script]})
                self.assertEqual(marker.read_text(), 'ok')


if __name__ == '__main__':
    unittest.main()
