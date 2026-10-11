"""
Windows production server (Gunicorn Windows par nahi chalta).

    python deploy/run_waitress.py            # http://127.0.0.1:8000
    python deploy/run_waitress.py --port 8080

HTTPS ke liye iske aage IIS (URL Rewrite + ARR reverse proxy) rakhein — dekhein DEPLOY.md.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

from waitress import serve  # noqa: E402

from config.wsgi import application  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"AutoInspect on http://{args.host}:{args.port}/")
    # channel_timeout lamba: AI inspection mein ek request ~1 minute le sakti hai
    serve(application, host=args.host, port=args.port, threads=4,
          channel_timeout=180, max_request_body_size=100 * 1024 * 1024,
          # IIS (isi machine par) jo X-Forwarded-Proto: https bhejta hai us par bharosa
          trusted_proxy="127.0.0.1",
          trusted_proxy_headers={"x-forwarded-proto", "x-forwarded-for", "x-forwarded-host"},
          clear_untrusted_proxy_headers=True)
