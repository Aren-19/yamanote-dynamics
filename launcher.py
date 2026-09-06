"""Launch the Yamanote model as an application."""

from __future__ import annotations

import argparse
import atexit
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app.py"

if sys.platform == "win32":
    VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
else:
    VENV_PY = ROOT / ".venv" / "bin" / "python"

STARTUP_TIMEOUT_S = 90.0


def find_free_port(preferred: int | None = None) -> int:
    """An unused TCP port, preferring `preferred` if it happens to be free."""
    if preferred:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", preferred))
                return preferred
            except OSError:
                pass
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def interpreter() -> Path:
    """The interpreter to run Streamlit with."""
    if VENV_PY.exists():
        return VENV_PY
    return Path(sys.executable)


class LogDrain:
    """Continuously drain a child process's output on a background thread."""

    def __init__(self, proc: subprocess.Popen, keep: int = 400) -> None:
        self._lines: deque[str] = deque(maxlen=keep)
        self._thread = threading.Thread(target=self._pump, args=(proc,), daemon=True)
        self._thread.start()

    def _pump(self, proc: subprocess.Popen) -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            self._lines.append(line.rstrip())

    def tail(self, n: int = 40) -> str:
        return "\n".join(list(self._lines)[-n:])


def start_server(port: int) -> tuple[subprocess.Popen, LogDrain]:
    cmd = [
        str(interpreter()), "-m", "streamlit", "run", str(APP),
        "--server.port", str(port),
        # Loopback only; Streamlit otherwise binds every interface.
        "--server.address", "127.0.0.1",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "--server.fileWatcherType", "none",
    ]
    kwargs: dict = {}
    if sys.platform == "win32":
        # Keep the console from flashing up when launched from a shortcut.
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, **kwargs)
    return proc, LogDrain(proc)


def wait_until_serving(port: int, proc: subprocess.Popen, log: LogDrain) -> bool:
    """Poll until the server answers, or it dies, or we give up."""
    url = f"http://localhost:{port}"
    deadline = time.time() + STARTUP_TIMEOUT_S

    while time.time() < deadline:
        if proc.poll() is not None:
            print("  server exited before it started serving:", file=sys.stderr)
            print(log.tail(), file=sys.stderr)
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.4)

    print(f"  timed out after {STARTUP_TIMEOUT_S:.0f}s waiting for {url}",
          file=sys.stderr)
    print(log.tail(), file=sys.stderr)
    return False


def open_window(url: str, prefer_browser: bool, frameless: bool = False) -> None:
    """Open a native window if pywebview is present, else the browser."""
    if not prefer_browser:
        try:
            import webview  # type: ignore

            print("  opening desktop window (pywebview)")
            webview.create_window("Yamanote Line dynamics model", url,
                                  width=1520, height=980,
                                  min_size=(1100, 720),
                                  background_color="#0E1419",
                                  frameless=frameless,
                                  easy_drag=frameless)
            webview.start()
            return
        except ImportError:
            print("  pywebview not installed, opening in the browser instead.")
            print("  For a native window:  .venv\\Scripts\\python.exe -m pip "
                  "install pywebview")

    print(f"  opening {url}")
    webbrowser.open(url)
    print("\n  Server is running. Press Ctrl-C here to stop it.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  stopping.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the Yamanote model.")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--browser", action="store_true",
                        help="Force the browser even if pywebview is available.")
    parser.add_argument("--frameless", action="store_true",
                        help="Borderless window. Drops the OS close/minimise "
                             "buttons: drag anywhere to move, Alt+F4 to close.")
    args = parser.parse_args()

    if not APP.exists():
        print(f"  app.py not found at {APP}", file=sys.stderr)
        return 1
    if not VENV_PY.exists():
        print(f"  No virtual environment at {VENV_PY.parent.parent}.", file=sys.stderr)
        print("  Create it first:", file=sys.stderr)
        print("      python -m venv .venv", file=sys.stderr)
        print("      .venv\\Scripts\\python.exe -m pip install -e \".[viz]\"",
              file=sys.stderr)
        return 1

    port = find_free_port(args.port)
    if port != args.port:
        print(f"  port {args.port} busy, using {port}")

    print("  starting Yamanote model ...")
    proc, log = start_server(port)

    def cleanup() -> None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

    atexit.register(cleanup)

    if not wait_until_serving(port, proc, log):
        cleanup()
        return 1

    print("  ready: first load runs the full model, allow a few seconds")

    open_window(f"http://localhost:{port}", prefer_browser=args.browser,
                frameless=args.frameless)
    cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
