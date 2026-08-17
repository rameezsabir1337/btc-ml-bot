"""Daemonize the dashboard: double-fork so it survives the launching shell."""
import os
import sys

os.chdir("/home/rmz/hack/bot")
pid = os.fork()
if pid > 0:
    os._exit(0)
os.setsid()
pid2 = os.fork()
if pid2 > 0:
    os._exit(0)
fd = os.open("/tmp/dash.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(fd, 1)
os.dup2(fd, 2)
os.execv(sys.executable,
         [sys.executable, "-m", "uvicorn", "dashboard.api:app",
          "--host", "0.0.0.0", "--port", "8050"])
