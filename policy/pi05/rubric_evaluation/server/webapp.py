from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from policy.pi05.rubric_evaluation.server.config import EvalConfig
from policy.pi05.rubric_evaluation.server.eval_manager import EvalManager


def create_app(*, cfg: EvalConfig, manager: EvalManager) -> FastAPI:
    app = FastAPI(title="Pi0 Rubric Evaluation", version="0.1")

    static_dir = Path("policy/pi05/rubric_evaluation/web/static")
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        index_path = Path("policy/pi05/rubric_evaluation/web/index.html")
        if not index_path.exists():
            return "<h1>Missing web UI</h1><p>Expected policy/pi05/rubric_evaluation/web/index.html</p>"
        return index_path.read_text(encoding="utf-8")

    @app.get("/api/status")
    def api_status() -> Dict[str, Any]:
        s = manager.status()
        cur = manager.current_version()
        return {
            "running": s.running,
            "model_name": cfg.model_name,
            "task_name": s.task_name,
            "task_config": cfg.task_config,
            "num_episodes": s.num_episodes,
            "evaluated": s.evaluated,
            "success": s.success,
            "success_rate": (s.success / s.evaluated) if s.evaluated else 0.0,
            "last_update_unix": s.last_update_unix,
            "current_version_id": s.current_version_id,
            "current_version_dir": str(cur.version_dir) if cur else None,
            "backends": s.backends,
        }

    @app.get("/api/versions")
    def api_versions() -> Dict[str, Any]:
        versions = manager.list_versions()
        out = []
        for v in versions:
            state_path = v.version_dir / "state.json"
            evaluated = 0
            if state_path.exists():
                try:
                    st = json.loads(state_path.read_text(encoding="utf-8"))
                    evaluated = len(st.get("evaluated_episode_ids", []))
                except Exception:
                    evaluated = 0
            out.append(
                {
                    "version_id": v.version_id,
                    "created_at_unix": v.created_at_unix,
                    "sha256": v.sha256,
                    "summary": v.summary,
                    "version_dir": str(v.version_dir),
                    "evaluated": evaluated,
                }
            )
        return {"versions": out}

    @app.post("/api/reload")
    def api_reload() -> Dict[str, Any]:
        v = manager.reload_rubric()
        return {"ok": True, "version_id": v.version_id, "version_dir": str(v.version_dir)}

    @app.post("/api/rollback/{version_id}")
    def api_rollback(version_id: str) -> Dict[str, Any]:
        v = manager.rollback(version_id)
        return {"ok": True, "version_id": v.version_id, "version_dir": str(v.version_dir)}

    @app.get("/api/episodes")
    def api_episodes(limit: int = 50) -> Dict[str, Any]:
        cur = manager.current_version()
        if cur is None:
            return {"episodes": []}
        metrics_path = cur.version_dir / "metrics.jsonl"
        if not metrics_path.exists():
            return {"episodes": []}
        lines = metrics_path.read_text(encoding="utf-8").splitlines()
        lines = [ln for ln in lines if ln.strip()]
        tail = lines[-limit:]
        episodes = []
        for ln in tail:
            try:
                episodes.append(json.loads(ln))
            except Exception:
                continue
        return {"episodes": episodes}

    @app.get("/runs/{model_name}/{task_name_config}/{version_id}/videos/{video_name}")
    def serve_video(model_name: str, task_name_config: str, version_id: str, video_name: str):
        # StaticFiles doesn't support dynamic roots easily; serve via file response
        # Directory structure: runs/{model_name}/{task_name}-{task_config}/{version_id}
        from fastapi.responses import FileResponse

        p = cfg.runs_root / model_name / task_name_config / version_id / "videos" / video_name
        if not p.exists():
            raise HTTPException(status_code=404, detail="Video not found")
        return FileResponse(str(p), media_type="video/mp4")

    @app.get("/runs/{model_name}/{task_name_config}/{version_id}/attention/{episode_dir}/{step_name}")
    def serve_attention(model_name: str, task_name_config: str, version_id: str, episode_dir: str, step_name: str):
        from fastapi.responses import FileResponse

        p = cfg.runs_root / model_name / task_name_config / version_id / "attention" / episode_dir / step_name
        if not p.exists():
            raise HTTPException(status_code=404, detail="Attention map not found")
        return FileResponse(str(p), media_type="image/png")

    return app