# cook_a_brisket (2026 task 51)

Classical controller on RGB-D + proprioception (no learned parts) from the b1k_rl_guys repository, served by `solutions/b1k_classical/backend.py`.

- Source: `tasks/cook_a_brisket/experiments/candidate/v20_submit` at `09f820ca82` (branch `main`), source fingerprint `59d869b44ff62d15…` (16 files), in `controller/`
- Inputs: head and wrist RGB-D, proprioception, robot-relative camera poses; static priors in `controller/priors/` (robot model, a map of the task's scene, object models); no instance pose and no simulator state
- Design and task facts: `controller/README.md`; scored runs and reports: b1k_rl_guys `tasks/cook_a_brisket/`
- Measured (native evaluator, fresh process per episode, instances 301-310): SR 2/10, mean Q 0.27; per instance 301: q 0.00, 302: q 0.33, 303: ok, 304: q 0.00, 305: q 0.00, 306: ok, 307: q 0.00, 308: q 0.00, 309: q 0.00, 310: q 0.33

Regenerate with `tools/import_b1k_classical.py` (the commands are in the repository README).

RTX smoke gate (repository README), 2026-10-16:

- Image `b1k-evo-submit:vera-brisket-eggs-8f116b9`, id `sha256:11fafc01056d136ec9771c06977e2bc6c793d7b4207d5a40913db0cbc613aa11`: `docker/Dockerfile` with the routing table reduced to cook_a_brisket and hiding_Easter_eggs, built with BuildKit
  through `docker compose build` (that host has no buildx). `docker/Dockerfile`, `server/`, `solutions/b1k_classical/` and this
  task's `controller/` and `solution.json` are unchanged since the build.
- RTX 4090 (AlexStar), BEHAVIOR-1K v3.9.3-post1: the evaluator step of `scripts/run_eval.sh` (what `B1K_EVALUATOR_CONTAINER=1`
  runs), instance 301 (index 0), fresh process, `--num-envs 1`, `--write-video`.
- Checks: `/healthz` OK; router `route 51 cook_a_brisket -> b1k_classical`; `classical controller cook_a_brisket (v20_submit) loaded from
  /opt/submit/tasks/cook_a_brisket/controller`, `b1k_classical ready`; evaluator `Connected to server!`; no worker subprocess (no cuRobo);
  `scripts/check_results.py --strict`: 0 errors, 0 warnings.
- Result: success false, Q 0.00, 11003 steps (time limit); the scored run of 301 above: Q 0.00, 11003 steps.
- JSON, MP4 and logs (evaluator, container): `results_tmp/smoke_cook_a_brisket/standard.public.VERA_team.team.20261016/` on AlexStar.
- The 301-310 results above come from the same controller code under the unmodified evaluator with the evaluator package's
  `WebsocketPolicyServer`, not from this image.
