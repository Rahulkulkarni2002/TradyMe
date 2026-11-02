import os
import sys
from pathlib import Path

os.environ["TRADYME_OFFLINE"] = "1"          # tests never touch the internet
os.environ["TRADYME_NO_LOOP"] = "1"          # no background auto-trader during tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
