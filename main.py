"""Backward-compatible entry point: ``python main.py [options]``.

Equivalent to the ``skeleton-transfer`` command installed by ``pip install .``;
see ``skeleton_transfer/cli.py`` or ``python main.py --help``.
"""

from skeleton_transfer.cli import main

if __name__ == "__main__":
    main()
