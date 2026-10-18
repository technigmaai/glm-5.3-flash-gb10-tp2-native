"""Check the public menu/CLI without starting model containers or downloads."""
import io
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import launcher


class Launcher(unittest.TestCase):
    def setUp(self):
        output = patch('sys.stdout', new_callable=io.StringIO)
        output.start()
        self.addCleanup(output.stop)

    def command(self, *args):
        return launcher.build_command(launcher.parser().parse_args(args))

    def test_start_selects_the_requested_recipe_and_preserves_approval_flags(self):
        command, _ = self.command('start', '--version', 'v0.6.0', '--dry-run')
        self.assertEqual(command[2:], ['start', '--profile', 'v0.6.0', '--dry-run'])
        self.assertTrue(command[1].endswith('/scripts/profilectl.py'))

    def test_setup_separates_image_version_from_serving_profile(self):
        command, _ = self.command('setup', '--worker', 'user@worker', '--worker-dir',
                                 '/worker directory', '--version', 'v0.6.0', '--serving-profile', 'c6')
        self.assertIn('technigmaai/glm-5.3-flash-gb10-tp2-native:v0.6.0', command)
        self.assertEqual(command[command.index('--profile') + 1], 'c6')
        self.assertEqual(command[command.index('--worker-dir') + 1], '/worker directory')
        self.assertNotIn('--apply', command)
        self.assertNotIn('--approved', command)

    def test_worker_logs_are_forwarded_without_a_shell_command_string(self):
        command, _ = self.command('logs', '--node', 'worker', '--tail', '40', '--follow')
        self.assertEqual(command[-5:], ['logs', 'worker', '--tail', '40', '--follow'])

    def test_empty_arguments_open_the_menu_instead_of_starting_a_model(self):
        with patch.object(launcher, 'menu', return_value=0) as menu:
            self.assertEqual(launcher.execute([]), 0)
            menu.assert_called_once_with(ROOT)

    def test_menu_start_targets_selected_version_only_after_confirmation(self):
        select = Mock(side_effect=[0, 5, 1, 8])
        run = Mock(return_value=0)
        launcher.menu(select=select, run=run, prompt=Mock(return_value=''))
        run.assert_called_once_with(['start', '--version', 'v0.6.0', '--approved'])

    def test_menu_cancel_never_invokes_a_backend(self):
        run = Mock()
        launcher.menu(select=Mock(side_effect=[0, 5, 0, 8]), run=run, prompt=Mock())
        run.assert_not_called()

    def test_back_from_version_selection_never_invokes_a_backend(self):
        run = Mock()
        launcher.menu(select=Mock(side_effect=[0, None, 8]), run=run, prompt=Mock())
        run.assert_not_called()

    def test_status_does_not_ask_for_an_image_version(self):
        run = Mock(return_value=0)
        launcher.menu(select=Mock(side_effect=[1, 8]), run=run, prompt=Mock())
        run.assert_called_once_with(['status'])

    def test_menu_setup_review_does_not_create_settings_or_download(self):
        run = Mock(return_value=0)
        launcher.menu(select=Mock(side_effect=[5, 0, 0, 8]), run=run,
                      prompt=Mock(side_effect=['user@worker', '', '']))
        run.assert_called_once_with(['setup', '--version', 'v0.1.0', '--worker', 'user@worker'])

    def test_menu_setup_apply_is_an_explicit_separate_selection(self):
        run = Mock(return_value=0)
        launcher.menu(select=Mock(side_effect=[5, 0, 1, 8]), run=run,
                      prompt=Mock(side_effect=['user@worker', '/worker checkout', '']))
        self.assertEqual(run.call_args.args[0], ['setup', '--version', 'v0.1.0', '--worker',
                         'user@worker', '--worker-dir', '/worker checkout', '--apply', '--prepare-assets'])

    def test_verification_can_be_cancelled_without_model_requests(self):
        run = Mock()
        launcher.menu(select=Mock(side_effect=[7, 2, 5, 0, 8]), run=run, prompt=Mock())
        run.assert_not_called()

    def test_invalid_version_and_conflicting_checks_fail_before_subprocess(self):
        with patch.object(launcher.subprocess, 'run') as run:
            with self.assertRaises(SystemExit):
                launcher.execute(['start', '--version', 'v0.7.0', '--approved'])
            with self.assertRaises(SystemExit):
                launcher.execute(['check', '--version', 'v0.6.0', '--all'])
            run.assert_not_called()

    def test_menu_requires_a_tty_and_explains_command_line_mode(self):
        with patch.object(launcher.sys.stdin, 'isatty', return_value=False):
            with self.assertRaisesRegex(ValueError, '--help'):
                launcher.choose('Choose', ['One'])

    def test_public_script_works_from_another_working_directory(self):
        result = subprocess.run(['bash', str(ROOT / 'launchers/launch.sh'), 'list', '--json'],
                                cwd='/', text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('v0.6.0', result.stdout)

    def test_unapproved_public_mutations_fail_without_cluster_settings(self):
        for action in ('start', 'restart', 'stop'):
            result = subprocess.run(['bash', str(ROOT / 'launchers/launch.sh'), action],
                                    cwd='/', text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('Supply --approved', result.stderr)

    def test_brand_is_ascii_and_has_the_requested_name(self):
        text = '\n'.join(launcher.BRAND)
        self.assertTrue(text.isascii())
        self.assertIn('Technigma AI', text)
        self.assertIn('DECODE. AUTOMATE. EVOLVE.', text)


if __name__ == '__main__':
    unittest.main()
