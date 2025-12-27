from __future__ import annotations

import json
import logging
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from policy.pi0.rubric_evaluation.server.backend_pool import Backend, make_backends, parse_gpus
from policy.pi0.rubric_evaluation.server.config import EvalConfig
from policy.pi0.rubric_evaluation.server.rubric_versioning import (
    RubricVersion,
    create_new_version,
    list_versions,
    load_state,
    rollback_to_version,
    save_state,
)


def _worker_process_main(
    backend_host: str,
    backend_port: int,
    gpu_id: int,
    task_name: str,
    task_config: str,
    pi0_step: int,
    max_steps: int,
    job_q: mp.Queue,
    result_q: mp.Queue,
    log_dir: str,
    cwd: str,
) -> None:
    """
    Worker process main loop. Runs in a separate process with its own CUDA context.
    This avoids SAPIEN ray tracing CUDA conflicts that occur with threading.
    """
    import os
    import sys
    import traceback

    # CRITICAL: Set working directory FIRST (rollout_worker uses relative paths)
    os.chdir(cwd)

    # CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE importing any CUDA libraries
    # This ensures SAPIEN runs on the correct GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Set up logging to file for this worker process
    log_file = Path(log_dir) / f"worker_{backend_port}.log"
    logging.basicConfig(
        level=logging.INFO,
        format=f"[{backend_port}] %(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    logging.info(f"Worker process starting for backend {backend_port} on GPU {gpu_id}")
    logging.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    logging.info(f"Python path: {sys.path[:3]}...")
    logging.info(f"Working dir: {Path.cwd()}")

    try:
        # Import here to avoid issues with multiprocessing spawn
        logging.info("Importing dependencies...")
        from envs.utils.create_actor import UnStableError
        from openpi_client import websocket_client_policy
        from policy.pi0.rubric_evaluation.server.rollout_worker import WorkerConfig, run_one_episode
        logging.info("Imports successful")

        worker_cfg = WorkerConfig(
            task_name=task_name,
            task_config=task_config,
            instruction_type="unseen",
            pi0_step=pi0_step,
            max_steps_fallback=max_steps,
        )
        logging.info(f"Worker config created: {worker_cfg}")

        # Create persistent websocket client
        client = None
        logging.info(f"Connecting to backend {backend_host}:{backend_port}...")
        client = websocket_client_policy.WebsocketClientPolicy(
            host=backend_host, port=backend_port
        )
        logging.info(f"Connected to backend {backend_port}")

        logging.info(f"Starting job loop")

        while True:
            try:
                job = job_q.get(timeout=1.0)
            except:
                continue

            if job is None:  # Poison pill to stop
                logging.info("Received stop signal")
                break

            # Job now includes rubric_path, out_dir, and version_id for dynamic version switching
            episode_id, seed, rubric_path, out_dir, version_id = job
            rubric_path_obj = Path(rubric_path)
            out_dir_obj = Path(out_dir)
            logging.info(f"Processing episode {episode_id} with seed {seed}, version={version_id}")

            try:
                rec = run_one_episode(
                    backend_host=backend_host,
                    backend_port=backend_port,
                    rubric_path=rubric_path_obj,
                    out_dir=out_dir_obj,
                    episode_id=episode_id,
                    seed=seed,
                    worker_cfg=worker_cfg,
                    client=client,
                )
                logging.info(f"Episode {episode_id} completed: success={rec.get('success')}")
            except UnStableError:
                logging.warning(f"Episode {episode_id} unstable, retrying with seed {seed + 1}")
                job_q.put((episode_id, seed + 1, rubric_path, out_dir, version_id))
                continue
            except Exception as e:
                logging.error(f"Episode {episode_id} failed: {e}")
                logging.error(traceback.format_exc())
                rec = {
                    "episode_id": int(episode_id),
                    "seed": int(seed),
                    "success": False,
                    "error": repr(e),
                    "backend": {"host": backend_host, "port": int(backend_port)},
                    "time_sec": 0.0,
                }

            # Include version_id so collector can discard stale results after reload
            result_q.put({"record": rec, "version_id": version_id, "out_dir": out_dir})

    except Exception as e:
        logging.error(f"Worker process fatal error: {e}")
        logging.error(traceback.format_exc())
        # Put error into result queue so parent knows
        result_q.put({
            "record": {
                "episode_id": -1,
                "seed": -1,
                "success": False,
                "error": f"Worker process crashed: {repr(e)}",
                "backend": {"host": backend_host, "port": int(backend_port)},
                "time_sec": 0.0,
            }
        })
    finally:
        if 'client' in dir() and client is not None:
            try:
                client.close()
            except Exception:
                pass
        logging.info(f"Worker process for backend {backend_port} exiting")


@dataclass
class ManagerStatus:
    running: bool
    current_version_id: str | None
    task_name: str
    num_episodes: int
    evaluated: int
    success: int
    last_update_unix: int
    backends: list[dict[str, Any]]


class EvalManager:
    def __init__(self, cfg: EvalConfig, *, current_rubric_path: Path) -> None:
        self._cfg = cfg
        self._current_rubric_path = current_rubric_path

        gpu_ids = parse_gpus(cfg.gpus)
        self._backends: list[Backend] = make_backends(cfg.host, cfg.base_port, gpu_ids)

        self._lock = threading.RLock()
        self._stop_event = threading.Event()

        self._current_version: RubricVersion | None = None
        self._threads: list[threading.Thread] = []
        self._job_q: queue.Queue[tuple[int, int]] = queue.Queue()  # (episode_id, seed)

        # Use multiprocessing queues for inter-process communication
        # (Threading causes CUDA context conflicts with SAPIEN ray tracing)
        self._mp_ctx = mp.get_context("spawn")  # spawn to avoid CUDA fork issues
        self._result_q: mp.Queue = self._mp_ctx.Queue()
        self._processes: list[mp.Process] = []

        self._success = 0
        self._evaluated = 0
        self._last_update_unix = int(time.time())

        # Thread to collect results from worker processes
        self._collector_thread: threading.Thread | None = None

    @property
    def cfg(self) -> EvalConfig:
        return self._cfg

    def list_versions(self) -> list[RubricVersion]:
        return list_versions(self._cfg.runs_root, self._cfg.task_name)

    def current_version(self) -> RubricVersion | None:
        with self._lock:
            return self._current_version

    def status(self) -> ManagerStatus:
        with self._lock:
            return ManagerStatus(
                running=not self._stop_event.is_set(),
                current_version_id=self._current_version.version_id if self._current_version else None,
                task_name=self._cfg.task_name,
                num_episodes=self._cfg.num_episodes,
                evaluated=self._evaluated,
                success=self._success,
                last_update_unix=self._last_update_unix,
                backends=[{"host": b.host, "port": b.port, "gpu_id": b.gpu_id} for b in self._backends],
            )

    def _load_or_init_version(self) -> RubricVersion:
        # If there is no version yet, create one from current rubric.
        versions = self.list_versions()
        if versions:
            # Use most recent as current
            return versions[0]
        return create_new_version(
            runs_root=self._cfg.runs_root,
            current_rubric_path=self._current_rubric_path,
            task_name=self._cfg.task_name,
        )

    def start(self) -> None:
        with self._lock:
            if self._processes:
                return
            self._current_version = self._load_or_init_version()
            self._stop_event.clear()

            # Start result collector thread
            self._collector_thread = threading.Thread(target=self._result_collector_loop, daemon=True)
            self._collector_thread.start()

            # Start worker processes FIRST (one per backend * workers_per_backend)
            # Each process has its own CUDA context, avoiding ray tracing conflicts
            v = self._current_version
            cwd = str(Path.cwd())  # Capture current working directory for spawned processes
            log_dir = str(v.version_dir)  # Fixed log directory at startup
            for backend in self._backends:
                for worker_id in range(self._cfg.rollout_workers_per_backend):
                    # Create per-process job queue
                    job_q = self._mp_ctx.Queue()
                    p = self._mp_ctx.Process(
                        target=_worker_process_main,
                        args=(
                            backend.host,
                            backend.port,
                            backend.gpu_id,  # Pass GPU ID so worker runs SAPIEN on correct GPU
                            self._cfg.task_name,
                            self._cfg.task_config,
                            self._cfg.pi0_step,
                            self._cfg.max_steps,
                            job_q,
                            self._result_q,
                            log_dir,  # Fixed log directory at startup
                            cwd,  # Pass working directory for relative path resolution
                        ),
                        daemon=True,
                    )
                    p.start()
                    self._processes.append((p, job_q))
                    logging.info(f"Started worker process for backend {backend.port} on GPU {backend.gpu_id}")

            # Now distribute jobs AFTER processes are created
            self._rebuild_jobs_locked()

    def stop(self) -> None:
        self._stop_event.set()

    def reload_rubric(self) -> RubricVersion:
        with self._lock:
            v = create_new_version(
                runs_root=self._cfg.runs_root,
                current_rubric_path=self._current_rubric_path,
                task_name=self._cfg.task_name,
            )
            self._current_version = v
            self._success = 0
            self._evaluated = 0
            self._rebuild_jobs_locked()
            return v

    def rollback(self, version_id: str) -> RubricVersion:
        with self._lock:
            version_dir = rollback_to_version(
                runs_root=self._cfg.runs_root,
                task_name=self._cfg.task_name,
                version_id=version_id,
                current_rubric_path=self._current_rubric_path,
            )
            # Set current version to that version and continue (skip already evaluated)
            versions = self.list_versions()
            v = next((x for x in versions if x.version_id == version_id), None)
            if v is None:
                # Construct minimal
                st = load_state(version_dir)
                v = RubricVersion(
                    version_id=version_id,
                    version_dir=version_dir,
                    rubric_path=version_dir / "rubric.py",
                    created_at_unix=int(st.get("created_at_unix", 0)),
                    summary=str(st.get("rubric_summary", "")),
                    sha256=str(st.get("rubric_sha256", "")),
                )
            self._current_version = v
            self._success = 0
            self._evaluated = 0
            self._rebuild_jobs_locked()
            return v

    def _rebuild_jobs_locked(self) -> None:
        assert self._current_version is not None
        # Drain queue
        while True:
            try:
                self._job_q.get_nowait()
            except queue.Empty:
                break

        state = load_state(self._current_version.version_dir)
        done_ids = set(state.get("evaluated_episode_ids", []))

        # Parse metrics.jsonl to find episodes with errors that need re-evaluation
        # Keep: episodes with proper evaluation (no "error" field or success=True/False with steps)
        # Re-evaluate: episodes with "error" field (ConnectionClosedError, FileNotFoundError, etc.)
        error_episode_ids: set[int] = set()
        good_records: list[str] = []  # Lines to keep in metrics.jsonl
        metrics_path = self._current_version.version_dir / "metrics.jsonl"

        if metrics_path.exists():
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    episode_id = rec.get("episode_id", -1)
                    # Check if this is an error record (has "error" field)
                    if "error" in rec:
                        error_episode_ids.add(episode_id)
                        logging.info(f"Episode {episode_id} will be re-evaluated (had error: {rec['error'][:50]}...)")
                    else:
                        # Keep this good record
                        good_records.append(line)
                except Exception:
                    continue

        # Rewrite metrics.jsonl with only good records (remove error records)
        if error_episode_ids:
            logging.info(f"Removing {len(error_episode_ids)} error episodes from metrics.jsonl")
            with metrics_path.open("w", encoding="utf-8") as f:
                for line in good_records:
                    f.write(line + "\n")

            # Update state to remove error episodes from done list
            done_ids -= error_episode_ids
            state["evaluated_episode_ids"] = sorted(done_ids)
            save_state(self._current_version.version_dir, state)
            logging.info(f"Updated state: {len(done_ids)} episodes remain evaluated")

        # Get current version's paths for job distribution
        rubric_path_str = str(self._current_version.rubric_path)
        out_dir_str = str(self._current_version.version_dir)
        version_id = self._current_version.version_id

        # Distribute jobs round-robin to worker processes
        pending_jobs = []
        for episode_id in range(self._cfg.num_episodes):
            if episode_id in done_ids:
                continue
            seed = self._cfg.seed_start + episode_id
            # Include rubric_path, out_dir, and version_id so workers use correct version after reload
            pending_jobs.append((episode_id, seed, rubric_path_str, out_dir_str, version_id))

        logging.info(f"Queueing {len(pending_jobs)} episodes for evaluation to {out_dir_str} (version={version_id})")

        # Distribute to workers round-robin
        for i, job in enumerate(pending_jobs):
            if self._processes:
                worker_idx = i % len(self._processes)
                _, job_q = self._processes[worker_idx]
                job_q.put(job)

        # Recompute counters from metrics.jsonl (only good records now)
        self._success = 0
        self._evaluated = 0
        if metrics_path.exists():
            for line in metrics_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    self._evaluated += 1
                    if rec.get("success", False):
                        self._success += 1
                except Exception:
                    continue

        self._last_update_unix = int(time.time())

    def _result_collector_loop(self) -> None:
        """Collect results from worker processes and update state."""
        while not self._stop_event.is_set():
            try:
                result = self._result_q.get(timeout=0.5)
            except:
                continue

            with self._lock:
                v = self._current_version
                if v is None:
                    continue

                rec = result["record"]
                episode_id = rec["episode_id"]
                result_version_id = result.get("version_id")
                result_out_dir = result.get("out_dir")

                # Discard stale results from previous rubric versions
                if result_version_id and result_version_id != v.version_id:
                    logging.warning(
                        f"Discarding stale result for episode {episode_id} "
                        f"(from version {result_version_id}, current is {v.version_id})"
                    )
                    continue

                # Write to the version's directory that matches the result
                # (use result_out_dir if available, otherwise current version)
                out_dir = Path(result_out_dir) if result_out_dir else v.version_dir

                # Append metrics
                self._append_metrics(out_dir, rec)
                self._mark_done(out_dir, episode_id)

                self._evaluated += 1
                if rec.get("success", False):
                    self._success += 1
                self._last_update_unix = int(time.time())

    def _append_metrics(self, version_dir: Path, rec: Dict[str, Any]) -> None:
        p = version_dir / "metrics.jsonl"
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _mark_done(self, version_dir: Path, episode_id: int) -> None:
        state = load_state(version_dir)
        done = set(state.get("evaluated_episode_ids", []))
        done.add(int(episode_id))
        state["evaluated_episode_ids"] = sorted(done)
        save_state(version_dir, state)
