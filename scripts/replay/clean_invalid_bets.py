#!/usr/bin/env python3
"""Remove historical rows marked MISS with no actual result before calibration.

The ignored bet_log.csv is backed up first. Voided rows are preserved in a
separate local CSV, never treated as a graded miss. Re-running is idempotent.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bet_log_utils import read_bet_rows, write_bet_rows  # noqa: E402


def clean(path=ROOT / "bet_log.csv", backup_dir=ROOT / "backups") -> dict:
    path, backup_dir = Path(path), Path(backup_dir)
    fields, rows = read_bet_rows(path)
    ungraded = [r for r in rows if not str(r.get("actual", "")).strip()]
    graded = [r for r in rows if str(r.get("actual", "")).strip()]
    if any(r.get("result") not in ("HIT", "MISS") for r in graded):
        raise ValueError("graded log contains unsupported results")
    backup_dir.mkdir(exist_ok=True, parents=True)
    backup = backup_dir / "bet_log.before-void-cleanup.csv"
    if ungraded:
        if not backup.exists():
            shutil.copy2(path, backup)
        write_bet_rows(backup_dir / "bet_log.void-or-ungraded.csv", fields, ungraded)
        write_bet_rows(path, fields, graded)
    return {"before": len(rows), "after": len(graded), "void_or_ungraded": len(ungraded),
            "backup": str(backup), "void_file": str(backup_dir / "bet_log.void-or-ungraded.csv")}


if __name__ == "__main__":
    print(json.dumps(clean(), indent=2))
