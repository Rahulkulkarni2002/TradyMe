"""Download 2015 → today of price history + news for the prediction model.
    python download_history.py            (everything; safe to stop and restart)
    python download_history.py --help     (options)"""
import sys

from research.download import main

sys.exit(main())
