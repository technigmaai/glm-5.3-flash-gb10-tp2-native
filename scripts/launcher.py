#!/usr/bin/env python3
"""One terminal menu and command-line entry point for the native GB10 pair."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import textwrap

import setup
from runtime_profiles import catalog

ROOT = Path(__file__).resolve().parents[1]
BRAND = (
    'Technigma AI',
    'DECODE. AUTOMATE. EVOLVE.',
    'GLM-5.3-Flash | Native GB10 TP2',
)


def logo(width):
    path = ROOT / 'files/branding' / f'technigma-ai-{width}.txt'
    return tuple(path.read_text(encoding='ascii').strip('\n').splitlines())


def wordmark(width, height):
    for font in ('standard', 'small', 'mini'):
        path = ROOT / 'files/branding' / f'wordmark-{font}.txt'
        art = tuple(path.read_text(encoding='ascii').strip('\n').splitlines())
        # Leave space for the tagline, nine actions and keyboard hints.
        if max(map(len, art)) <= width and height >= len(art) + 18:
            return art
    return ()


def screen_layout(height, width):
    # Put the image beside the controls so a larger logo does not hide actions.
    art, column = (), 1
    for size in (52, 44, 34):
        candidate = logo(size)
        candidate_column = max(map(len, candidate)) + 4
        if height >= 20 and len(candidate) <= height - 3 and width - candidate_column - 2 >= 38:
            art, column = candidate, candidate_column
            break
    banner = wordmark(width - column - 1, height)
    title_row = len(banner) + 3 if banner else 4
    return art, column, title_row, title_row + 2


def menu_page(height, option_row, options, width):
    # Reserve description rows only when a choice needs them.
    reserve = 6 if any(len(option) > width - 2 for option in options) else 4
    return max(1, height - option_row - reserve)


def parser(root=ROOT):
    p = argparse.ArgumentParser(prog='./launchers/launch.sh', description=__doc__, epilog='Run without arguments for the arrow-key menu. Commands operate from the head node.')
    commands = p.add_subparsers(dest='command')
    commands.add_parser('menu', help='Open the interactive menu')
    commands.add_parser('list', help='List image versions and their contents').add_argument('--json', action='store_true')
    versions = list(catalog(root))
    for command in ('start', 'restart', 'stop', 'check', 'config', 'verify'):
        c = commands.add_parser(command)
        c.add_argument('--version', choices=versions, help='Fixed image release; independent of C4/C6 serving settings')
        if command in ('start', 'restart', 'stop'):
            c.add_argument('--approved', action='store_true')
            c.add_argument('--dry-run', action='store_true')
        if command == 'check':
            c.add_argument('--all', action='store_true', help='Check all installed releases on both nodes')
    commands.add_parser('status', help='Inspect both running ranks').add_argument('--json', action='store_true')
    c = commands.add_parser('setup', parents=[setup.build_parser(versioned=True, add_help=False)],
                            help='Plan a fresh installation or create node configurations')
    c.add_argument('--version', choices=versions, default='v0.1.0')
    c = commands.add_parser('logs', help='Read a model container log')
    c.add_argument('--node', choices=('head', 'worker'), default='head')
    c.add_argument('--tail', type=int, default=80)
    c.add_argument('--follow', action='store_true')
    c = commands.add_parser('sync', help='Sync tracked source while both checkouts are offline')
    g = c.add_mutually_exclusive_group(required=True)
    g.add_argument('--dry-run', action='store_true')
    g.add_argument('--approved', action='store_true')
    return p


def build_command(a, root=ROOT):
    root = Path(root)
    env = dict(os.environ)
    if a.command == 'setup':
        command = [sys.executable, str(root / 'scripts/setup.py'), '--image', catalog(root)[a.version]['image'],
                   '--profile', a.profile]
        for key, value in vars(a).items():
            if key in ('command', 'version', 'profile') or value is None or value is False:
                continue
            command.append('--' + key.replace('_', '-'))
            if value is not True:
                command.append(str(value))
        return command, env
    if a.command in ('logs', 'verify', 'sync'):
        command = ['bash', str(root / 'scripts/cluster.sh')]
        if a.command == 'logs':
            if a.tail < 0:
                raise ValueError('--tail must be nonnegative')
            command += ['logs', a.node, '--tail', str(a.tail)]
            if a.follow:
                command.append('--follow')
        elif a.command == 'verify':
            # Use the running version's complete node settings when available.
            from profilectl import Controller
            c = Controller(root)
            version = a.version or c.active(c.snapshots())
            if not version:
                raise ValueError('No recognized runtime is running; supply --version for explicit verification')
            env['GLM53_PROFILE'] = version
            command += ['verify']
        else:
            command += ['sync', '--approved' if a.approved else '--dry-run']
        return command, env
    command = [sys.executable, str(root / 'scripts/profilectl.py'), a.command]
    if getattr(a, 'version', None):
        command += ['--profile', a.version]
    for option in ('approved', 'dry_run', 'json', 'all'):
        if getattr(a, option, False):
            command.append('--' + option.replace('_', '-'))
    return command, env


def execute(argv, root=ROOT):
    p = parser(root)
    a = p.parse_args(argv)
    if not a.command or a.command == 'menu':
        return menu(root)
    if a.command == 'check' and a.all and a.version:
        p.error('--all and --version cannot be combined')
    command, env = build_command(a, root)
    return subprocess.run(command, env=env).returncode


def _curses_choice(screen, title, options):
    import curses
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    screen.keypad(True)
    index = 0
    while True:
        screen.erase()
        height, width = screen.getmaxyx()
        art, column, title_row, option_row = screen_layout(height, width)
        def line(y, text, style=0, x=None):
            x = column if x is None else x
            if 0 <= y < height - 1:
                try:
                    screen.addnstr(y, x, text, max(0, width - x - 1), style)
                except curses.error:
                    pass
        if art:
            for row, text in enumerate(art, 1):
                line(row, text, curses.A_BOLD, x=1)
        banner = wordmark(width - column - 1, height)
        if banner:
            for row, text in enumerate(banner):
                line(row, text, curses.A_BOLD)
            line(len(banner) + 1, BRAND[1])
            line(height - 3, BRAND[2], x=1)
        else:
            for row, text in enumerate(BRAND):
                line(row, text, curses.A_BOLD if row == 0 else 0)
        line(title_row, title)
        page = menu_page(height, option_row, options, width - column - 1)
        first = max(0, index - page + 1)
        for row, option in enumerate(options[first:first + page], option_row):
            selected = first + row - option_row == index
            line(row, ('> ' if selected else '  ') + option,
                 curses.A_REVERSE if selected else 0)
        if len(options[index]) > width - column - 3:
            detail = textwrap.wrap(options[index], max(1, width - column - 2))
            for row, text in enumerate(detail[:2], height - 5):
                line(row, text)
        line(height - 2, 'Up/Down or j/k: move | Enter: select | Esc/q: back', x=1)
        screen.refresh()
        key = screen.get_wch()
        if key in (curses.KEY_UP, 'k'):
            index = (index - 1) % len(options)
        elif key in (curses.KEY_DOWN, 'j'):
            index = (index + 1) % len(options)
        elif key in ('\n', '\r', curses.KEY_ENTER):
            return index
        elif key in ('\x1b', 'q'):
            return None


def choose(title, options):
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise ValueError('The menu needs an interactive terminal. Run launchers/launch.sh --help for command-line usage.')
    if os.environ.get('TERM', 'dumb') != 'dumb':
        try:
            import curses
        except ImportError:
            pass
        else:
            return curses.wrapper(_curses_choice, title, options)
    print('\n' + '\n'.join(logo(44)) + '\n\n' + '\n'.join(wordmark(200, 100)) +
          '\n\n' + '\n'.join(BRAND[1:]) + '\n\n' + title)
    for i, option in enumerate(options, 1):
        print(f'  {i}. {option}')
    while True:
        value = input('Selection (Enter to go back): ').strip()
        if not value:
            return None
        if value.isdigit() and 1 <= int(value) <= len(options):
            return int(value) - 1
        print('Choose one of the listed numbers.')


def menu(root=ROOT, select=None, run=None, prompt=None):
    select = select or choose
    run = run or (lambda argv: execute(argv, root))
    prompt = prompt or input
    actions = [('start', 'Start / switch image version'), ('status', 'Show cluster status'),
               ('restart', 'Restart the running version'), ('stop', 'Stop both ranks'),
               ('check', 'Check an image version'), ('setup', 'Set up a fresh installation'),
               ('logs', 'Read logs'), ('advanced', 'More operations'), ('quit', 'Quit')]
    def pick_version():
        versions = catalog(root)
        names = list(versions)
        index = select('Choose an image release', [v + '  |  ' + versions[v]['description'] for v in names])
        return names[index] if index is not None else None
    while True:
        index = select('Choose an operation', [label for _, label in actions])
        if index is None or actions[index][0] == 'quit':
            return 0
        action = actions[index][0]
        args = [action]
        try:
            if action in ('start', 'check', 'setup'):
                version = pick_version()
                if version is None:
                    continue
                args += ['--version', version]
            if action in ('start', 'restart', 'stop'):
                detail = f'Start {version}' if action == 'start' else action.capitalize() + ' the running version'
                decision = select(detail + ' — interrupts requests on both nodes', ['Cancel', 'Proceed'])
                if decision != 1:
                    continue
                args.append('--approved')
            elif action == 'setup':
                worker = prompt('Worker SSH destination (user@host): ').strip()
                if not worker:
                    continue
                args += ['--worker', worker]
                worker_dir = prompt('Worker checkout directory (Enter if same as head): ').strip()
                if worker_dir:
                    args += ['--worker-dir', worker_dir]
                mode = select('Fresh setup: existing .env files are protected',
                              ['Review the setup plan', 'Create configurations and download image/models', 'Back'])
                if mode is None or mode == 2:
                    continue
                if mode == 1:
                    args += ['--apply', '--prepare-assets']
            elif action == 'logs':
                node = select('Read logs from', ['Head', 'Worker'])
                if node is None:
                    continue
                args += ['--node', 'head' if node == 0 else 'worker', '--tail', '80']
            elif action == 'advanced':
                advanced = select('More operations', ['List image versions', 'Render a version configuration',
                                  'Run inference verification', 'Preview source synchronization', 'Back'])
                if advanced is None or advanced == 4:
                    continue
                args = [['list'], ['config'], ['verify'], ['sync', '--dry-run']][advanced]
                if advanced in (1, 2):
                    version = pick_version()
                    if version is None:
                        continue
                    args += ['--version', version]
                if advanced == 2 and select('Verification submits model requests', ['Cancel', 'Run verification']) != 1:
                    continue
            print('\n$ ./launchers/launch.sh ' + shlex.join(args), flush=True)
            result = run(args)
            if result:
                print(f'Command exited with status {result}. Read the error above before retrying.')
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            print(str(error), file=sys.stderr)
        prompt('\nPress Enter to return to the menu...')


def main(argv=None):
    try:
        return execute(sys.argv[1:] if argv is None else argv)
    except KeyboardInterrupt:
        print('\nInterrupted. Inspect status before retrying.', file=sys.stderr)
        return 130
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
