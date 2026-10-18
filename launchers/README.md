# One launcher for every version

```bash
./launchers/launch.sh
```

Run from the **head node**. The terminal menu uses Up/Down (or `j`/`k`) and Enter;
Esc or `q` goes back. A numbered menu is available on basic terminals. Start,
restart and stop show the operation before interrupting both model ranks.

Choose **Start / switch image version**, then select one of v0.1.0–v0.6.0.
The selection includes the exact image and matching repository source overlays.
See [release contents](../VERSIONS.md) before choosing an experimental version.

## Command-line use

| Task | Command from repository root |
|---|---|
| Fresh setup plan | `./launchers/launch.sh setup --worker user@worker --version v0.1.0` |
| List releases | `./launchers/launch.sh list` |
| Check one release | `./launchers/launch.sh check --version v0.6.0` |
| Preview a switch | `./launchers/launch.sh start --version v0.6.0 --dry-run` |
| Start or switch | `./launchers/launch.sh start --version v0.6.0 --approved` |
| Inspect both ranks | `./launchers/launch.sh status` |
| Read worker logs | `./launchers/launch.sh logs --node worker --tail 80` |
| Follow head logs | `./launchers/launch.sh logs --follow` |
| Restart the running version | `./launchers/launch.sh restart --approved` |
| Stop both ranks | `./launchers/launch.sh stop --approved` |
| Render a recipe | `./launchers/launch.sh config --version v0.6.0` |
| Run inference checks | `./launchers/launch.sh verify --version v0.6.0` |
| Preview source synchronization | `./launchers/launch.sh sync --dry-run` |

`./launchers/launch.sh --help` lists commands; append `--help` to a command for
its options. Use [the installation guide](../docs/DEPLOYMENT.md) for first setup.

The menu and command line use the same backend. They do not build or download
images during a switch, rewrite private node settings, or install a watchdog.
Startup is asynchronous; check API health after launching. Repeating Start for
an already running recipe leaves it in place; Restart applies source/settings
changes. Stop and Restart infer the recognized running version when none is supplied.

An explicit `--version` is clearest for command-line Start. If omitted, Start
uses `RUNTIME_VERSION` from the base `.env`, defaulting to v0.1.0. Check follows
the same default; `check --all` requires every image on both nodes.

The former root scripts and per-version shell scripts have been consolidated
into this file. Internal Python and shell helpers remain under `scripts/`.
