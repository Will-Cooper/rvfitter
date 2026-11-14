"""Smoke test for a frozen GUI: it must still be running after a few seconds (a crash at start-up fails)."""
import os
import subprocess
import sys

exe, wait = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 30.
proc = subprocess.Popen([exe], env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'},
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
try:
    out, _ = proc.communicate(timeout=wait)
except subprocess.TimeoutExpired:
    proc.kill()
    print(f'{exe}: still running after {wait:g} s: OK')
    sys.exit(0)
print(out)
sys.exit(f'{exe} exited after less than {wait:g} s with code {proc.returncode}')
