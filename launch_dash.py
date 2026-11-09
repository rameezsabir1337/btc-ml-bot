"""Daemonize the dashboard: double-fork so it survives the launching shell."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

LOG_PATH = Path(os.environ.get("DASH_LOG", ROOT / "data" / "dash.log"))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

HOST = os.environ.get("DASH_HOST", "0.0.0.0")
PORT = os.environ.get("DASH_PORT", "8050")

pid = os.fork()
if pid > 0:
    os._exit(0)
os.setsid()
pid2 = os.fork()
if pid2 > 0:
    os._exit(0)
fd = os.open(LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(fd, 1)
os.dup2(fd, 2)
os.execv(sys.executable,
         [sys.executable, "-m", "uvicorn", "dashboard.api:app",
          "--host", HOST, "--port", PORT])
