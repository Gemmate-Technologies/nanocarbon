# SPDX-License-Identifier: CC-BY-NC-SA-4.0
# Copyright (c) 2026 Gemmate Technologies s.r.l.
"""
python -m nanocarbon <verb> [args]

Verbs
-----
generate <config_file>
    Run a parallel batch from a Python config file.
    Template: utilities/batch_config.py

optimize <config_file>
    Run the PSO calibration from a Python config file.
    Template: utilities/calibrate_config.py

Post-anneal trajectory analysis is not a verb: use utilities/analyze_anneal.py.
"""

import sys

_VERBS = ('generate', 'optimize')


def main() -> None:
    """
    Dispatch the verb in sys.argv[1] to batch.main or calibrate.main; print the usage and
    exit on a missing or unknown verb.
    """
    if len(sys.argv) < 2 or sys.argv[1] not in _VERBS:
        sys.exit(
            f'Usage: python -m nanocarbon <{"  |  ".join(_VERBS)}> [args]\n'
            f'       python -m nanocarbon generate  --help\n'
            f'       python -m nanocarbon optimize  --help'
        )
    verb, argv = sys.argv[1], sys.argv[2:]
    if verb == 'generate':
        from .batch import main as _main
    elif verb == 'optimize':
        from .calibrate import main as _main
    else:
        raise RuntimeError("Verb not recognized")

    _main(argv)

if __name__ == '__main__':
    main()
