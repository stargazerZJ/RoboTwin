# Rubric Evaluation Server (Pi0)

This directory contains a **fast iteration** evaluation system for `blocks_ranking_rgb` that:

- Keeps **policy inference warm** on multiple GPUs (websocket servers).
- Runs **rollouts in parallel** and writes **streaming results** (JSONL + MP4).
- Supports **rubric hot-reload** with **versioning** and **rollback**.
- Provides a **web UI** optimized for fast inspection on a 4K monitor.

## Quick start (tmux-friendly)

### 0) Activate env
Use the venv at [`policy/pi0/.venv`](policy/pi0/.venv:1).

### 1) Launch inference backends (one per GPU)
From repo root (recommended to run in a tmux window):

```bash
bash policy/pi0/rubric_evaluation/bin/launch_backends.sh \
  --train_config_name pi0_base_aloha_robotwin_lora \
  --model_name blocks_ranking_split \
  --checkpoint_id latest \
  --gpus all \
  --base_port 8000
```

This starts one websocket policy server per GPU using [`policy/pi0/scripts/serve_policy.py`](policy/pi0/scripts/serve_policy.py:1) and writes logs to:
- `policy/pi0/rubric_evaluation/backend_gpu*.log`

### 2) Start the evaluation manager + web UI
Run in another tmux window:

```bash
bash policy/pi0/rubric_evaluation/bin/start_server.sh \
  --task_name blocks_ranking_rgb \
  --task_config demo_randomized \
  --train_config_name pi0_base_aloha_robotwin_lora \
  --model_name blocks_ranking_split \
  --checkpoint_id latest \
  --gpus all \
  --base_port 8000 \
  --num_episodes 500 \
  --rollout_workers_per_backend 1
```

Open the UI at: `http://localhost:8899`

### 3) Edit rubric and reload (no backend restart)
Edit the rubric file:

- [`policy/pi0/rubric_evaluation/rubrics/blocks_ranking_rgb_rubric.py`](policy/pi0/rubric_evaluation/rubrics/blocks_ranking_rgb_rubric.py:1)

Then notify the server to snapshot + reload the rubric (creates a new version folder):

```bash
bash policy/pi0/rubric_evaluation/bin/reload_rubric.sh
```

### 4) Roll back to a previous rubric version
List versions in the UI (Versions panel) or on disk under:

- `policy/pi0/rubric_evaluation/runs/blocks_ranking_rgb/`

Then:

```bash
bash policy/pi0/rubric_evaluation/bin/rollback_rubric.sh --version <VERSION_ID>
```

Rollback overwrites the current rubric code with the selected version and evaluation continues, **skipping episodes already evaluated** for that rubric version.

---

## Folder layout

- `bin/` — bash scripts to launch backends, start server, reload, rollback.
- `rubrics/` — the *current* editable rubric file (what you edit).
- `runs/` — rubric version folders:
  - `rubric.py` (snapshotted rubric code)
  - `metrics.jsonl` (streaming per-episode results)
  - `videos/episode_XXXX.mp4`
  - `state.json` (resume bookkeeping)
- `web/` — static web UI (HTML/JS/CSS).
- `server/` — evaluation manager + web server (FastAPI).

---

## Web UI shortcuts

- `↑/↓`: select previous/next episode
- `space`: play/pause
- `j/k`: seek -/+ 1s
- `f`: frame-step (assumes 10 fps)
- `,` / `.`: speed down/up
- `l`: toggle loop
- `r`: reload rubric
- `v`: toggle versions panel

---

## Notes / gotchas

- Inference is the bottleneck; rollouts are sharded across GPUs by assigning each rollout worker to a backend port.
- Rubric changes affect prompts and therefore require re-running rollouts; this system re-runs only the episodes not yet evaluated for that rubric version.
- If you increase `--rollout_workers_per_backend`, you may saturate CPU or the simulator; start with 1.