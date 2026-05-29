"""Allow ``python -m scratchv`` to run the CLI."""

from scratchv.main import main
import sys

if __name__ == "__main__":
    sys.exit(main())
