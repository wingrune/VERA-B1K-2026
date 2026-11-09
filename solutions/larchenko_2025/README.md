# Larchenko 2025 VLA under the 2026 evaluator

The 1st-place solution of the 2025 BEHAVIOR Challenge (Ilia Larchenko et al., Pi0.5-based, Apache-2.0,
https://github.com/IliaLarchenko/behavior-1k-solution, report https://arxiv.org/abs/2512.06951), served behind a 2026 adapter.

| Solution id | Checkpoint (HF `IliaLarchenko/behavior_submission`) | 2025 tasks it was finetuned on (2025 ids = 2026 ids) |
|---|---|---|
| `larchenko_ckpt2` | `checkpoint_2` | 0, 1, 7, 8, 9, 12, 16, 17, 18, 20, 21, 22, 26, 30, 43, 45 |
| `larchenko_ckpt3` | `checkpoint_3` | 4, 27, 31, 32, 33, 35, 36, 37, 38, 39, 41, 46, 49 |
| `larchenko_ckpt4` | `checkpoint_4` | 40 |

Any of the three can be routed to any of the 50 tasks of 2025 (the task embedding exists in all of them); tasks new in 2026
(ids 50-99) are refused by the routing check. Get a checkpoint (weights stay outside git; the build copies them into the image):

    hf download IliaLarchenko/behavior_submission --include "checkpoint_2/*" --local-dir checkpoints/larchenko

and route tasks to it in `config/routing.csv`: `turning_on_radio,larchenko_ckpt2,checkpoints/larchenko/checkpoint_2`.

## Files

- `upstream/`: the solution source (src/b1k, its openpi fork, scripts) as used by our team, with the 2025-submission config
  `pi_behavior_2025_submission` and `convert_2025_action.py`. Changes against the team copy: training-only dataset and cache paths
  of the cluster replaced by `~/data/behavior` / `~/.cache`, team training launch scripts removed, `openpi` source set to the
  bundled fork.
- `adapter_2026.py`: the 2026 adapter (task id remap by name, 2025 stage table, action converter on, batch 1). Its docstring lists
  every difference and the known gap (base velocity frame).
- `backend.py`: one checkpoint per process, the evaluator protocol on the port the router gives it.
- `task_ids_2025.json`: the 2025 task index (BEHAVIOR-1K v3.7 `omnigibson/learning/utils/eval_utils.py`).
- `data/2026-challenge-demos/meta/tasks.jsonl`: that file of the HF dataset `behavior-1k/2026-challenge-demos` (same ids as
  `server/task_ids_2026.json`). `b1k.training.config` reads it at import; `backend.py` points `OPENPI_BEHAVIOR_DATA_ROOT` here.

## Environment

The image builds `/opt/venv-vla` from `upstream/openpi/uv.lock` (jax 0.5.3, torch 2.7.1, lerobot 0.3.4) and installs `b1k` on top
without re-resolving. The model loads in bf16 (about 12 GB GPU memory); JIT compilation makes the first action slow, the router
keeps the evaluator connection alive meanwhile.
