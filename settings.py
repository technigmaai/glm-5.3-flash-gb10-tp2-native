"""Read generated, shell-quoted node settings without hostname assumptions."""
from pathlib import Path
import shlex


def load_settings(root=None):
    root = Path(root) if root else Path(__file__).resolve().parent
    values = {}
    for line in (root / '.env').read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        key, value = line.split('=', 1)
        values[key] = ' '.join(shlex.split(value, comments=True))
    return values
