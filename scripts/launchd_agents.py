"""Install, remove or list this project's macOS launchd agents.

    python3 scripts/launchd_agents.py install              # every agent below
    python3 scripts/launchd_agents.py install predict      # just one
    python3 scripts/launchd_agents.py uninstall injuries
    python3 scripts/launchd_agents.py status

Agents run with the interpreter that installs them, from the repo root, with
the same minimal environment as the closing agent, and log to
~/Library/Logs/nba-props/. The existing ...nba-props.closing agent (every 15
minutes) is left alone.

launchd calendar times are local. The times below are for Madrid, which is 6
hours ahead of ET for most of the season. Between the EU and US clock changes
(late Oct to early Nov, and in March) the gap is 5 hours, so runs land an hour
later in ET. A Mac that is asleep at a scheduled time runs the job once on wake.
"""

from __future__ import annotations

import getpass
import plistlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCH_AGENTS = Path.home() / "Library" / "LaunchAgents"
LOG_DIR = Path.home() / "Library" / "Logs" / "nba-props"
PREFIX = f"com.{getpass.getuser()}.nba-props"

AGENTS = {
    # Picks once a day at 16:30 ET, after evening props post and before 7 PM tips. One run a
    # day keeps Odds API credits down (a full slate is ~120); weekend games tipping before
    # 16:30 ET get no picks unless run by hand.
    "predict": {"args": ["nba_combined.py"], "local_times": [(22, 30)], "et": "16:30 ET"},
    "injuries": {"args": ["snapshot_injuries.py"],
                 "local_times": [(16, 0), (20, 0), (23, 0), (0, 30)],
                 "et": "10:00, 14:00, 17:00 and 18:30 ET"},
    # The local web dashboard (dashboard.py), kept running.
    "dashboard": {"args": ["dashboard.py"], "keep_alive": True, "et": "always on"},
}


def label(name: str) -> str:
    return f"{PREFIX}.{name}"


def plist_path(name: str) -> Path:
    return LAUNCH_AGENTS / f"{label(name)}.plist"


def build_plist(name: str) -> dict:
    spec = AGENTS[name]
    script = [str(ROOT / a) if a.endswith(".py") else a for a in spec["args"]]
    if not spec.get("keep_alive"):
        # Scheduled jobs wait for the network first: a job due while the Mac slept fires on wake.
        script = [str(ROOT / "scripts" / "when_online.py"), *script]
    plist = {
        "Label": label(name),
        "ProgramArguments": [sys.executable, "-u", *script],
        "WorkingDirectory": str(ROOT),
        "EnvironmentVariables": {"HOME": str(Path.home()),
                                 "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
                                 "PYTHONDONTWRITEBYTECODE": "1"},
        "ProcessType": "Background",
        "StandardOutPath": str(LOG_DIR / f"{name}.stdout.log"),
        "StandardErrorPath": str(LOG_DIR / f"{name}.stderr.log"),
    }
    if spec.get("keep_alive"):
        plist["RunAtLoad"] = True
        plist["KeepAlive"] = True
    else:
        plist["RunAtLoad"] = False
        plist["StartCalendarInterval"] = [{"Hour": h, "Minute": m} for h, m in spec["local_times"]]
    return plist


def _launchctl(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["launchctl", *args], capture_output=True, text=True)


def install(name: str) -> None:
    LAUNCH_AGENTS.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = plist_path(name)
    domain = f"gui/{subprocess.check_output(['id', '-u'], text=True).strip()}"
    _launchctl("bootout", f"{domain}/{label(name)}")  # reinstall cleanly; fails harmlessly if absent
    with open(path, "wb") as fh:
        plistlib.dump(build_plist(name), fh)
    result = _launchctl("bootstrap", domain, str(path))
    if result.returncode != 0:
        raise SystemExit(f"launchctl bootstrap failed for {label(name)}: {result.stderr.strip()}")
    print(f"  installed {label(name)}  ({AGENTS[name]['et']})")


def uninstall(name: str) -> None:
    domain = f"gui/{subprocess.check_output(['id', '-u'], text=True).strip()}"
    _launchctl("bootout", f"{domain}/{label(name)}")
    plist_path(name).unlink(missing_ok=True)
    print(f"  removed {label(name)}")


def status() -> None:
    listed = {line.split("\t")[-1]: line.split("\t")[:2]
              for line in _launchctl("list").stdout.splitlines()[1:] if PREFIX in line}
    for lbl, (pid, last_exit) in sorted(listed.items()):
        print(f"  {lbl:45s} pid {pid:>6s}  last exit {last_exit}")
    if not listed:
        print("  no nba-props agents loaded")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("install", "uninstall", "status"):
        print(__doc__)
        return 2
    if argv[0] == "status":
        status()
        return 0
    names = argv[1:] or list(AGENTS)
    unknown = [n for n in names if n not in AGENTS]
    if unknown:
        print(f"unknown agent(s): {unknown}; choose from {list(AGENTS)}")
        return 2
    for name in names:
        (install if argv[0] == "install" else uninstall)(name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
