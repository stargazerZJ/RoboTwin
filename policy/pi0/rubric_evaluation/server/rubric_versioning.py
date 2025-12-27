from __future__ import annotations

import hashlib
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


_RUBRIC_SUMMARY_RE = re.compile(r"RUBRIC_SUMMARY.*?(?=\\n\\\"\\\"\\\"|\\n'''|\\Z)", re.DOTALL)


@dataclass(frozen=True)
class RubricVersion:
    version_id: str
    version_dir: Path
    rubric_path: Path
    created_at_unix: int
    summary: str
    sha256: str


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def extract_rubric_summary(rubric_source: str) -> str:
    m = _RUBRIC_SUMMARY_RE.search(rubric_source)
    if not m:
        return ""
    return m.group(0).strip()


def _ensure_version_dir(
    *,
    runs_root: Path,
    task_name: str,
    version_id: str,
    rubric_source: str,
    sha256: str,
    summary: str,
    created_at: int,
) -> RubricVersion:
    version_dir = runs_root / task_name / version_id
    version_dir.mkdir(parents=True, exist_ok=True)

    rubric_path = version_dir / "rubric.py"
    if not rubric_path.exists():
        rubric_path.write_text(rubric_source, encoding="utf-8")

    (version_dir / "videos").mkdir(parents=True, exist_ok=True)

    state_path = version_dir / "state.json"
    if not state_path.exists():
        state = {
            "version_id": version_id,
            "created_at_unix": created_at,
            "task_name": task_name,
            "rubric_sha256": sha256,
            "rubric_summary": summary,
            "evaluated_episode_ids": [],
        }
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    (version_dir / "metrics.jsonl").touch(exist_ok=True)

    return RubricVersion(
        version_id=version_id,
        version_dir=version_dir,
        rubric_path=rubric_path,
        created_at_unix=created_at,
        summary=summary,
        sha256=sha256,
    )


def create_new_version(
    *,
    runs_root: Path,
    current_rubric_path: Path,
    task_name: str,
) -> RubricVersion:
    """
    Snapshot the current rubric into a version folder.

    Behavior:
    - If the rubric code SHA already exists as the latest version, return that version (no new folder).
    - Otherwise create a new version folder named: <unix>_<sha8>.
    """
    runs_root.mkdir(parents=True, exist_ok=True)

    src = current_rubric_path.read_text(encoding="utf-8")
    sha = _sha256_bytes(src.encode("utf-8"))
    summary = extract_rubric_summary(src)

    # If latest version has same sha, reuse it
    versions = list_versions(runs_root, task_name)
    if versions and versions[0].sha256 == sha:
        return versions[0]

    created_at = int(time.time())
    version_id = f"{created_at}_{sha[:8]}"
    return _ensure_version_dir(
        runs_root=runs_root,
        task_name=task_name,
        version_id=version_id,
        rubric_source=src,
        sha256=sha,
        summary=summary,
        created_at=created_at,
    )


def load_state(version_dir: Path) -> Dict[str, Any]:
    p = version_dir / "state.json"
    return json.loads(p.read_text(encoding="utf-8"))


def save_state(version_dir: Path, state: Dict[str, Any]) -> None:
    p = version_dir / "state.json"
    p.write_text(json.dumps(state, indent=2), encoding="utf-8")


def list_versions(runs_root: Path, task_name: str) -> list[RubricVersion]:
    task_dir = runs_root / task_name
    if not task_dir.exists():
        return []
    versions: list[RubricVersion] = []
    for d in sorted(task_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        rubric_path = d / "rubric.py"
        if not rubric_path.exists():
            continue
        src = rubric_path.read_text(encoding="utf-8")
        sha = _sha256_bytes(src.encode("utf-8"))
        state = load_state(d)
        versions.append(
            RubricVersion(
                version_id=d.name,
                version_dir=d,
                rubric_path=rubric_path,
                created_at_unix=int(state.get("created_at_unix", 0)),
                summary=str(state.get("rubric_summary", "")),
                sha256=sha,
            )
        )
    return versions


def rollback_to_version(
    *,
    runs_root: Path,
    task_name: str,
    version_id: str,
    current_rubric_path: Path,
) -> Path:
    version_dir = runs_root / task_name / version_id
    rubric_path = version_dir / "rubric.py"
    if not rubric_path.exists():
        raise FileNotFoundError(f"Rubric version not found: {rubric_path}")
    shutil.copyfile(rubric_path, current_rubric_path)
    return version_dir