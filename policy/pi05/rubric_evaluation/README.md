# Rubric Evaluation Server (Pi0)

This directory contains a **fast iteration** evaluation system that supports **multiple tasks** including:
- `blocks_ranking_rgb` - Arrange colored blocks in order (red-left, green-middle, blue-right)
- `put_object_cabinet` - Open cabinet drawer and place object inside

Features:
- Keeps **policy inference warm** on multiple GPUs (websocket servers).
- Runs **rollouts in parallel** and writes **streaming results** (JSONL + MP4).
- Supports **rubric hot-reload** with **versioning** and **rollback**.
- Provides a **web UI** optimized for fast inspection on a 4K monitor.
- **Task-agnostic**: easily add new tasks by creating a rubric file.
- **Hierarchical prompting**: supports subtask rubrics for studying decomposed prompting.

## Quick start (tmux-friendly)

### 0) Activate env
Use the venv at [`policy/pi05/.venv`](policy/pi05/.venv:1).

### 1) Launch inference backends (one per GPU)
From repo root (recommended to run in a tmux window):

```bash
bash policy/pi05/rubric_evaluation/bin/launch_backends.sh \
  --train_config_name pi0_base_aloha_robotwin_lora \
  --model_name blocks_ranking_split \
  --checkpoint_id latest \
  --gpus all \
  --base_port 8000
```

This starts one websocket policy server per GPU using [`policy/pi05/scripts/serve_policy.py`](policy/pi05/scripts/serve_policy.py:1) and writes logs to:
- `policy/pi05/rubric_evaluation/backend_gpu*.log`

### 2) Start the evaluation manager + web UI
Run in another tmux window:

**For blocks_ranking_rgb (baseline):**
```bash
bash policy/pi05/rubric_evaluation/bin/start_server.sh \
  --task_name blocks_ranking_rgb \
  --task_config demo_randomized \
  --train_config_name pi0_base_aloha_robotwin_lora \
  --model_name blocks_ranking_split \
  --checkpoint_id latest \
  --gpus all \
  --base_port 8000 \
  --num_episodes 500 \
  --rollout_workers_per_backend 1 \
  --rubric_variant baseline
```

**For put_object_cabinet (baseline - flat prompting):**
```bash
bash policy/pi05/rubric_evaluation/bin/start_server.sh \
  --task_name put_object_cabinet \
  --task_config demo_randomized \
  --train_config_name pi0_base_aloha_robotwin_lora \
  --model_name put_object_cabinet_model \
  --checkpoint_id latest \
  --gpus all \
  --base_port 8000 \
  --num_episodes 500 \
  --rubric_variant baseline
```

**For put_object_cabinet (subtask - hierarchical prompting):**
```bash
bash policy/pi05/rubric_evaluation/bin/start_server.sh \
  --task_name put_object_cabinet \
  --task_config demo_randomized \
  --train_config_name pi0_base_aloha_robotwin_lora \
  --model_name put_object_cabinet_model \
  --checkpoint_id latest \
  --gpus all \
  --base_port 8000 \
  --num_episodes 500 \
  --rubric_variant subtask
```

Open the UI at: `http://localhost:8899`

### 3) Edit rubric and reload (no backend restart)
Edit the rubric file for your task:

**Baseline rubrics (flat prompting):**
- [`policy/pi05/rubric_evaluation/rubrics/blocks_ranking_rgb_rubric.py`](policy/pi05/rubric_evaluation/rubrics/blocks_ranking_rgb_rubric.py:1) - for blocks_ranking_rgb
- [`policy/pi05/rubric_evaluation/rubrics/put_object_cabinet_rubric.py`](policy/pi05/rubric_evaluation/rubrics/put_object_cabinet_rubric.py:1) - for put_object_cabinet

**Subtask rubrics (hierarchical prompting):**
- [`policy/pi05/rubric_evaluation/rubrics/put_object_cabinet_subtask_rubric.py`](policy/pi05/rubric_evaluation/rubrics/put_object_cabinet_subtask_rubric.py:1) - hierarchical prompting for put_object_cabinet

Then notify the server to snapshot + reload the rubric (creates a new version folder):

```bash
bash policy/pi05/rubric_evaluation/bin/reload_rubric.sh
```

### 4) Roll back to a previous rubric version
List versions in the UI (Versions panel) or on disk under:

- `policy/pi05/rubric_evaluation/runs/{model_name}/{task_name}-{task_config}/`

Then:

```bash
bash policy/pi05/rubric_evaluation/bin/rollback_rubric.sh --version <VERSION_ID>
```

Rollback overwrites the current rubric code with the selected version and evaluation continues, **skipping episodes already evaluated** for that rubric version.

---

## Folder layout

- `bin/` — bash scripts to launch backends, start server, reload, rollback.
- `rubrics/` — editable rubric files:
  - `{task_name}_rubric.py` — baseline rubric (flat prompting)
  - `{task_name}_subtask_rubric.py` — subtask rubric (hierarchical prompting)
  - `blocks_ranking_rgb_rubric.py` — baseline rubric for blocks ranking task
  - `put_object_cabinet_rubric.py` — baseline rubric for put object in cabinet task
  - `put_object_cabinet_subtask_rubric.py` — subtask rubric for put object in cabinet task
- `runs/` — rubric version folders organized by model and task:
  - `{model_name}/{task_name}-{task_config}/{version_id}/`
    - `rubric.py` (snapshotted rubric code)
    - `metrics.jsonl` (streaming per-episode results)
    - `videos/episode_XXXX.mp4`
    - `state.json` (resume bookkeeping)
- `web/` — static web UI (HTML/JS/CSS).
- `server/` — evaluation manager + web server (FastAPI).

---

## Adding a new task

To add evaluation support for a new task:

1. **Create a rubric file** at `rubrics/{task_name}_rubric.py` following the pattern in existing rubrics.

2. **Required rubric interface:**
   ```python
   @dataclass
   class RubricConfig:
       # Task-specific configuration (tolerances, thresholds, etc.)
       pass

   @dataclass
   class RubricState:
       # Per-episode state (prompt, tracking variables)
       prompt: str = ""

   def reset() -> RubricState:
       """Called at episode start. Initialize state and select prompt."""
       pass

   def step(env: Any, observation: Dict[str, Any], state: RubricState, cfg: RubricConfig | None = None) -> Dict[str, Any]:
       """Called every step. Returns: {"prompt": str, "subtask_state": int, "done": bool, "debug": dict}"""
       pass
   ```

3. **Key implementation points:**
   - Use `env` to access simulator state (object poses, robot state)
   - `UNSEEN_PROMPT_TEMPLATES` should come from `description/task_instruction/{task_name}.json`
   - `done` should match the task's `check_success()` logic
   - `debug` dict is displayed in the web UI overlay

4. **Start the server** with `--task_name your_task_name --rubric_variant baseline`

---

## Hierarchical Subtask Rubrics

Subtask rubrics implement **hierarchical prompting** for studying whether decomposed instructions improve VLA performance.

### Motivation

VLA models trained on successful demonstrations cannot recover from failures. By decomposing a complex task into focused subtasks, we can:
- Provide more specific guidance at each stage
- Study which subtasks are most challenging
- Compare flat vs hierarchical prompting strategies

### put_object_cabinet Subtask Decomposition

The `put_object_cabinet` task is decomposed into three subtasks:

| Subtask | Description | Completion Condition | Prompt Focus |
|---------|-------------|---------------------|--------------|
| 0 | Grasp the object | Gripper near object AND closed | Object name + arm to use |
| 1 | Open the drawer | Drawer joint position > threshold | Arm to open drawer |
| 2 | Place object in drawer | Object at target + gripper open | Move object to drawer |

### Creating a Subtask Rubric

1. **Create** `rubrics/{task_name}_subtask_rubric.py`

2. **Implement state machine** in `RubricState`:
   ```python
   @dataclass
   class RubricState:
       subtask: int = 0  # Current subtask index
       prompt: str = ""
       initialized: bool = False
   ```

3. **Define subtask-specific prompts**:
   ```python
   SUBTASK_0_PROMPTS = ["Use your {arm} arm to grasp the {object}."]
   SUBTASK_1_PROMPTS = ["Open the drawer with your {arm} arm."]
   SUBTASK_2_PROMPTS = ["Place the {object} in the drawer."]
   ```

4. **Implement transitions** in `step()`:
   ```python
   if state.subtask == 0 and subtask_0_complete:
       state.subtask = 1
       state.prompt = generate_prompt(state, 1)
   elif state.subtask == 1 and subtask_1_complete:
       state.subtask = 2
       state.prompt = generate_prompt(state, 2)
   ```

5. **Start the server** with `--rubric_variant subtask`

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