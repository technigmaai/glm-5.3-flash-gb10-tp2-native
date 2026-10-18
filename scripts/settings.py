"""Read generated, shell-quoted node settings without hostname assumptions."""
from pathlib import Path
import shlex
from runtime_profiles import catalog, env_file, selected


def load_settings(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    values = {}
    for line in env_file(root).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, value = line.split('=', 1)
        values[key] = ' '.join(shlex.split(value, comments=True))
    version = selected(root)
    if version:
        values['IMAGE'] = catalog(root)[version]['image']
    return values
