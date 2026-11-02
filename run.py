"""Start TradyMe:   python run.py          (real free market data)
                   python run.py --demo   (simulated prices, no internet needed)
Then open http://localhost:8000 in your browser."""
import argparse
import os
import webbrowser
import threading

parser = argparse.ArgumentParser(description="TradyMe trading assistant")
parser.add_argument("--demo", action="store_true", help="use simulated prices (works offline)")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--no-browser", action="store_true")
args = parser.parse_args()

if args.demo:
    os.environ["TRADYME_OFFLINE"] = "1"

import uvicorn  # noqa: E402

if not args.no_browser:
    threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{args.port}")).start()
print(f"\n  📈 TradyMe running →  http://localhost:{args.port}   (Ctrl+C to stop)\n")
uvicorn.run("server.app:app", host="127.0.0.1", port=args.port, log_level="warning")
