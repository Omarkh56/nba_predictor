"""Reproducibility metadata recorded with every replay: code commit, dirty flag, source fingerprint, artifacts."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Any, Dict, List


def _git(root: str, *args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_info(root: str) -> Dict[str, Any]:
    status = _git(root, "status", "--porcelain")
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "working_tree_dirty": bool(status),
        "dirty_files": [ln[3:] for ln in status.splitlines()][:40],
    }


def code_fingerprint(root: str) -> str:
    """sha256 over the CONTENT of every tracked/untracked-but-not-ignored .py and .json config file the replay
    depends on, so a run can be tied to exact source even when it was executed before the commit was made."""
    files = _git(root, "ls-files", "--cached", "--others", "--exclude-standard", "*.py").splitlines()
    files += ["scripts/replay/splits.json"] if os.path.exists(os.path.join(root, "scripts/replay/splits.json")) else []
    h = hashlib.sha256()
    for rel in sorted(set(files)):
        if rel.startswith("scripts/replay/output/") or rel.startswith("scripts/replay/reports/"):
            continue
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            h.update(rel.encode())
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()


def artifact_info(root: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in ("learned_params.json", "calibration.json", "residual_corrections.json"):
        p = os.path.join(root, name)
        if os.path.exists(p):
            with open(p, "rb") as fh:
                out[name] = {"sha256": hashlib.sha256(fh.read()).hexdigest()[:16]}
    lp = os.path.join(root, "learned_params.json")
    if os.path.exists(lp):
        with open(lp) as fh:
            d = json.load(fh)
        out["learned_params.json"].update(
            {k: d.get(k) for k in ("version", "trained_on", "validated_on", "tested_on")}
        )
    return out
