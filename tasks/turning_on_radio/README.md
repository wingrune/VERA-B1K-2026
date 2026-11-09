# Turning On Radio (BEHAVIOR task 0)

Legacy-evolved RGB-D planner, champion `cf23f1b5` from `evo-20261008-194556` (iteration 21).

- Source file SHA256: `1ae068860bf490bff11347f96913f4b5f9487dcf19be384bfaf56f7d64d65e5c`
- Prior measurement: H100 SR008 port, instances 301–310, SR 9/10, Q 0.90; one timeout.
- The policy consumes only RGB-D from the three cameras, camera relative poses, and proprioception, as recorded by its passport.
- Runtime dependencies NumPy and SciPy are already installed by `docker/Dockerfile`.
- This is the preserved `B1K_hacker` evolution artifact; it is not a `BIMBO_EvoX` fair-package import. The RTX result below is measured separately by the official `b1k_evo_submit` evaluator.

## Validation results

| Hardware and evaluator | Instances | Success rate | Mean final Q |
|---|---|---|---|
| H100 SR008, Isaac Lab port (recorded evolution result) | 301–310 | 9/10 (90%) | 0.90 |
| RTX 4090, native evaluator, 2026-10-16 | 301–310 | 8/10 (80%) | 0.80 |

The RTX run used `python -m omnigibson.eval.eval`, `RGBDFullResWrapper`, one rollout, one environment, a fresh evaluator
and planner process per episode, default timeouts, and `--write-video`. Independent workers used GPU0 and GPU1 on
`selectel-turing`; episodes were not repeated or selected after seeing their results. The ten JSON/video pairs passed
`scripts/check_results.py --metrics-only --strict` with zero errors and zero warnings.

- Policy image: `b1k-evo-submit:turning-on-radio-cf23f1b5`
- Image ID: `sha256:c823ffb904c6092d82280062bbdde2cf668c370a3192e85b4f799bedcbc7ed26`
- Validation run: `standard.public.VERA_team.team.20261015`
- SHA256 and per-episode metrics: server `results_radio/radio_validated.json`; worker logs:
  `~/.quarterdeck-jobs/radio-worker{0,1,2}-20261016.log`.
- Artifact destination: [JSON](https://huggingface.co/datasets/VERA-VLA/b1k_final_submission/tree/main/standard.public.VERA_team.team.20261015/json)
  and [videos](https://huggingface.co/datasets/VERA-VLA/b1k_final_submission/tree/main/standard.public.VERA_team.team.20261015/videos).
  Files are downloaded to the orchestration laptop, checked against server SHA256, and uploaded from the laptop.

| Instance | Success | Final Q | Steps |
|---:|:---:|---:|---:|
| 301 | yes | 1.00 | 1558 |
| 302 | yes | 1.00 | 1780 |
| 303 | yes | 1.00 | 1716 |
| 304 | yes | 1.00 | 868 |
| 305 | no | 0.00 | 3225 |
| 306 | yes | 1.00 | 1156 |
| 307 | yes | 1.00 | 835 |
| 308 | yes | 1.00 | 1734 |
| 309 | yes | 1.00 | 2054 |
| 310 | no | 0.00 | 3225 |

Both unsuccessful episodes reached the 3,225-step horizon. The H100 port measurement and the native RTX measurement
are separate evaluations; the native RTX results are the submission validation results.
