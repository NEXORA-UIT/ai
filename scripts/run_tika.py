"""Launcher script for Apache Tika local server.

Auto-discovers portable OpenJDK and Tika JAR under ~/Tools/ or accepts
overrides via CLI arguments and environment variables.
"""

import argparse
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9998
READINESS_TIMEOUT_SECONDS = 30.0


def find_java_executable(custom_path: str | None = None) -> Path | None:
    """Find Java executable via override, ~/Tools/Java, or system PATH."""
    if custom_path:
        p = Path(custom_path)
        if p.is_file() and os.access(p, os.X_OK):
            return p

    env_path = os.environ.get("TIKA_JAVA_EXECUTABLE")
    if env_path:
        p = Path(env_path)
        if p.is_file() and os.access(p, os.X_OK):
            return p

    tools_java = Path.home() / "Tools" / "Java"
    if tools_java.is_dir():
        # Search for java.exe (Windows) or java (POSIX) inside jdk-* directories
        candidates = list(tools_java.glob("**/bin/java.exe")) + list(
            tools_java.glob("**/bin/java")
        )
        for cand in candidates:
            if cand.is_file() and os.access(cand, os.X_OK):
                return cand

    which_java = shutil.which("java")
    if which_java:
        return Path(which_java)

    return None


def find_tika_jar(custom_path: str | None = None) -> Path | None:
    """Find Tika server JAR via override or ~/Tools/Tika."""
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return p

    env_jar = os.environ.get("TIKA_SERVER_JAR")
    if env_jar:
        p = Path(env_jar)
        if p.is_file():
            return p

    tools_tika = Path.home() / "Tools" / "Tika"
    if tools_tika.is_dir():
        jars = sorted(tools_tika.glob("tika-server-standard-*.jar"), reverse=True)
        if not jars:
            jars = sorted(tools_tika.glob("tika-server-*.jar"), reverse=True)
        if jars:
            return jars[0]

    return None


def probe_tika_endpoint(host: str, port: int, timeout: float = 1.5) -> bool:
    """Return True if an Apache Tika server is responding at host:port."""
    url = f"http://{host}:{port}/version"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                body = resp.read().decode("utf-8", errors="replace")
                return "Apache Tika" in body or "tika" in body.lower()
    except Exception:
        pass

    # Secondary check on /tika
    url_tika = f"http://{host}:{port}/tika"
    try:
        req = urllib.request.Request(url_tika, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        pass

    return False


def is_port_in_use(host: str, port: int) -> bool:
    """Check if a TCP port is currently listening."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def wait_for_ready(
    host: str, port: int, timeout: float = READINESS_TIMEOUT_SECONDS
) -> bool:
    """Poll Tika server until it responds or timeout expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if probe_tika_endpoint(host, port, timeout=1.0):
            return True
        time.sleep(0.5)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Apache Tika Server with portable Java."
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("TIKA_HOST", DEFAULT_HOST),
        help=f"Host to bind Tika server (default: {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("TIKA_PORT", DEFAULT_PORT)),
        help=f"Port to bind Tika server (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--java",
        default=None,
        help="Path to java executable (default: auto-detect under ~/Tools/Java).",
    )
    parser.add_argument(
        "--jar",
        default=None,
        help="Path to tika-server JAR (default: auto-detect under ~/Tools/Tika).",
    )
    args = parser.parse_args()

    # Pre-flight check: Is Tika already running?
    if probe_tika_endpoint(args.host, args.port):
        print(
            f"[INFO] Apache Tika is already running at http://{args.host}:{args.port}"
        )
        return 0

    # If not Tika, is the port occupied by something else?
    if is_port_in_use(args.host, args.port):
        print(
            f"[ERROR] Port {args.port} on {args.host} is already occupied by "
            "another service (not Apache Tika).",
            file=sys.stderr,
        )
        print(
            f"[ERROR] Please free port {args.port} or configure a different port.",
            file=sys.stderr,
        )
        return 1

    # Locate Java executable
    java_exe = find_java_executable(args.java)
    if not java_exe:
        print(
            "[ERROR] Java executable not found. Please install JDK to "
            "~/Tools/Java or set TIKA_JAVA_EXECUTABLE.",
            file=sys.stderr,
        )
        return 1

    # Locate Tika JAR
    tika_jar = find_tika_jar(args.jar)
    if not tika_jar:
        print(
            "[ERROR] Tika server JAR not found. Please copy tika-server-standard-*.jar "
            "to ~/Tools/Tika or set TIKA_SERVER_JAR.",
            file=sys.stderr,
        )
        return 1

    cmd = [
        str(java_exe),
        "-jar",
        str(tika_jar),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    print("[INFO] Starting Apache Tika server...")
    print(f"       Java: {java_exe}")
    print(f"       JAR:  {tika_jar}")
    print(f"       Host: {args.host}:{args.port}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        print(f"[ERROR] Failed to start Tika process: {exc}", file=sys.stderr)
        return 1

    # Wait for Tika server to be ready
    print("[INFO] Waiting for Tika server readiness...", end="", flush=True)
    ready = False
    deadline = time.monotonic() + READINESS_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            # Process exited prematurely
            print("\n[ERROR] Tika process exited unexpectedly.")
            out, _ = proc.communicate()
            if out:
                print(out, file=sys.stderr)
            return 1
        if probe_tika_endpoint(args.host, args.port, timeout=1.0):
            ready = True
            break
        print(".", end="", flush=True)
        time.sleep(0.5)

    if not ready:
        print("\n[ERROR] Timed out waiting for Apache Tika server to become ready.")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        return 1

    print(f"\n[INFO] Apache Tika server is READY at http://{args.host}:{args.port}")
    print(f"[INFO] Process PID: {proc.pid}")
    print("[INFO] Press Ctrl+C to stop Tika server.")

    try:
        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            if line:
                sys.stdout.write(line)
                sys.stdout.flush()
            elif proc.poll() is not None:
                break
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print(f"\n[INFO] Stopping Apache Tika server (PID {proc.pid})...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[INFO] Apache Tika server stopped successfully.")
        return 0

    return proc.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
