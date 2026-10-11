"""Allow ``python -m metaproteomics``."""
import sys

from .pipeline import main

sys.exit(main())
