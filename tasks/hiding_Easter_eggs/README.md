# hiding_Easter_eggs (2026 task 6)

Classical controller on RGB-D + proprioception (no learned parts) from the b1k_rl_guys repository, served by `solutions/b1k_classical/backend.py`.

- Source: `tasks/hiding_Easter_eggs/experiments/candidate/v40` at `09f820ca82` (branch `main`), source fingerprint `962bdbe917eac723…` (21 files), in `controller/`
- Inputs: head and wrist RGB-D, proprioception, robot-relative camera poses; static priors in `controller/priors/` (robot model, a map of the task's scene, object models); no instance pose and no simulator state
- Design and task facts: `controller/README.md`; scored runs and reports: b1k_rl_guys `tasks/hiding_Easter_eggs/`
- Measured (native evaluator, fresh process per episode, instances 301-310): SR 4/10, mean Q 0.48; per instance 301: ok, 302: ok, 303: q 0.00, 304: q 0.00, 305: q 0.78, 306: ok, 307: q 0.00, 308: q 0.00, 309: ok, 310: q 0.00

Regenerate with `tools/import_b1k_classical.py` (the commands are in the repository README).

RTX smoke gate (repository README), 2026-10-16:

- Image `b1k-evo-submit:vera-brisket-eggs-8f116b9`, id `sha256:11fafc01056d136ec9771c06977e2bc6c793d7b4207d5a40913db0cbc613aa11`: `docker/Dockerfile` with the routing table reduced to cook_a_brisket and hiding_Easter_eggs, built with BuildKit
  through `docker compose build` (that host has no buildx). `docker/Dockerfile`, `server/`, `solutions/b1k_classical/` and this
  task's `controller/` and `solution.json` are unchanged since the build.
- RTX 4090 (AlexStar), BEHAVIOR-1K v3.9.3-post1: the evaluator step of `scripts/run_eval.sh` (what `B1K_EVALUATOR_CONTAINER=1`
  runs), instance 301 (index 0), fresh process, `--num-envs 1`, `--write-video`.
- Checks: `/healthz` OK; router `route 6 hiding_Easter_eggs -> b1k_classical`; `classical controller hiding_Easter_eggs (v40) loaded from
  /opt/submit/tasks/hiding_Easter_eggs/controller`, `b1k_classical ready`; evaluator `Connected to server!`; no worker subprocess (no cuRobo);
  `scripts/check_results.py --strict`: 0 errors, 0 warnings.
- Result: success true, Q 1.00, 4264 steps; the scored run of 301 above: success, 4810 steps.
- JSON, MP4 and logs (evaluator, container): `results_tmp/smoke_hiding_Easter_eggs/standard.public.VERA_team.team.20261016/` on AlexStar.
- The 301-310 results above come from the same controller code under the unmodified evaluator with the evaluator package's
  `WebsocketPolicyServer`, not from this image.
