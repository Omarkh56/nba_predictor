"""Start a pipeline script once the network is up.

    python3 scripts/when_online.py nba_combined.py [args...]

launchd runs a job that was due while the Mac slept the moment it wakes, often
before Wi-Fi reconnects, so the job's first request fails to resolve its host
(seen on the 16:00 injury snapshot). This waits until one of the data hosts
resolves, checking every 15 seconds for up to 10 minutes, then replaces itself
with the script (same pid and exit code for launchd). If the network never
comes back, the script still runs, so its failure is logged in run_log.jsonl.
"""

import os
import socket
import sys
import time

HOSTS = ("site.api.espn.com", "stats.nba.com", "api.the-odds-api.com")
MAX_WAIT_SECONDS = 600
POLL_SECONDS = 15


def online() -> bool:
    for host in HOSTS:
        try:
            socket.getaddrinfo(host, 443)
            return True
        except OSError:
            continue
    return False


def main(argv) -> None:
    if not argv:
        raise SystemExit(__doc__)
    deadline = time.monotonic() + MAX_WAIT_SECONDS
    waited = False
    while not online() and time.monotonic() < deadline:
        if not waited:
            print("  network not up yet; waiting before starting", argv[0], flush=True)
            waited = True
        time.sleep(POLL_SECONDS)
    os.execv(sys.executable, [sys.executable, "-u", *argv])


if __name__ == "__main__":
    main(sys.argv[1:])
