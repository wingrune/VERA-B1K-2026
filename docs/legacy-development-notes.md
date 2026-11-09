> Historical development notes from the original source snapshot. Commands, routing descriptions, and aggregate results below may be superseded. Start with the release README and reproduction guide.

# b1k_evo_submit

Submission package for the 2026 BEHAVIOR Challenge (BEHAVIOR-1K v3.9.3-post1, R1Pro, RGB + depth + proprioception).
One Docker image serves the evaluator's websocket policy on port 8000 and routes every task to its own solution:

- **planner**: task-specific motion planners from BIMBO_EvoX (fair planners: onboard RGB-D + proprioception only, GPU IK
  through cuRobo), one folder per task in `tasks/<task>/`;
- **b1k_classical**: classical controllers from the b1k_rl_guys repository (onboard RGB-D + proprioception, a whole-body IK and
  sampling planner of their own, static priors of the robot, the task's scene and its objects), one folder per task in
  `tasks/<task>/controller/`, served by `solutions/b1k_classical/backend.py`;
- **larchenko_ckpt2 / 3 / 4**: the 2025 winning VLA (Pi0.5-based) checkpoints behind a 2026 adapter, `solutions/larchenko_2025/`;
- **frozen_rgbd**: five unchanged classical RGB-D controllers in a separate pinned CPU environment (see below);
- **comet2**: the team's 2026 π0.5 checkpoint (group2), isolated OpenPI runtime; serves the seven group2 tasks; see [setup and checkpoint contract](solutions/comet2/README.md);
- **comet2_1**: the same runtime with the group1 checkpoint; serves its six group1 tasks;
- **comet3**: the same runtime with the all63 checkpoint; serves the tasks it was trained on that no other route serves, and any task outside the 100;
- any other model added to `config/solutions.json`.

Which solution serves which task is one CSV, `config/routing.csv`.

### comet2, comet2_1 and comet3

Each checkpoint serves the tasks it was trained on:

| Solution | Checkpoint (`checkpoints/<solution>`) | Tasks |
|---|---|---|
| `comet2` | group2_hour_a01 step 2498 | 66, 71, 79, 84, 88, 97, 98 |
| `comet2_1` | group1_7h_a01 step 20408 | 20, 26, 67, 81, 86, 91 |
| `comet3` | all63_1h_a01 step 2540 | 33 tasks, including 18, 39, 55 and 96, and the `*` row |

Tasks 18, 20, 39, 55, 67, 86 and 96 moved to comet2_1 or comet3 from Larchenko, planner and b1k_classical routes on
2026-10-17; their folders under `tasks/` stay, and the notes column of `config/routing.csv` records the previous route. The
router sends any task id without a row of its own, such as a task outside these 100, to the `*` route.
`tests/test_routing.py` fails if a task is left unrouted, because an unrouted task scores 0. All three run
`solutions/comet2/backend.py`, which takes the task name as its prompt and knows all 100 tasks
(`solutions/comet2/assets/task_ids.json`); a task id outside them gets an empty prompt.
Weights are supplied separately: place each checkpoint root (containing `params/` and `assets/`) at `checkpoints/comet2`,
`checkpoints/comet2_1` and `checkpoints/comet3`, or set the paths in the CSV rows. No weights are stored in Git. The ordinary
`scripts/build_docker.py` includes the isolated runtime only when a comet checkpoint is routed and copies
each distinct checkpoint once using a named build context. `scripts/run_eval.sh` preserves
the 20-action replay contract automatically. See `solutions/comet2/README.md` for build,
validation and provenance; evaluation results for these checkpoints are not claimed here.

## Tasks in this repository

| 2026 id | Task | Solution | Measured (H100 port, instances 301-310) |
|---:|---|---|---|
| 0 | turning_on_radio | planner (evolved, program cf23f1b5) | SR 9/10 (Q 0.90) |
| 35 | attach_a_camera_to_a_tripod | planner (evolved, program d20842d2) | SR 10/10 (Q 1.00) |
| 83 | store_honey | planner (evolved, program ffacbbe6) | SR 7/10 (Q 0.70) |
| 92 | installing_a_scanner | planner (evolved, program 7dd85299) | SR 4/10 (Q 0.60) |
| 69 | vacuuming_floors | planner (evolved, program 434d955b) | SR 3/10 (Q 0.30) |
| 42 | chop_an_onion | planner (evolved, program 00403145) | SR 1/10 (Q 0.33) |
| 55 | re_shelving_library_books | planner (evolved, program c243b165) | SR 0/10 (Q 0.33) |
| 44 | chopping_wood | planner (evolved, program b6615d11) | no H100 301-310 run; evolution RTX 301-305 SR 3/5 (Q 0.75) |
| 40 | make_microwave_popcorn | planner (evolved, program 740988aa) | SR 6/10 (Q 0.60) |
| 34 | hanging_pictures | planner (evolved, program b07a9f5b) | no H100 301-310 run; evolution RTX 301-310 SR 1/10 (Q 0.10) |
| 63 | installing_smoke_detectors | planner (evolved, program dfd49599) | SR 6/10 (Q 0.60) |
| 96 | cook_broccolini | planner (evolved, program cb2a002c) | SR 0/10 (Q 0.15) |
| 39 | spraying_fruit_trees | planner (evolved, program 0698e8f4) | H100 port 301-305 SR 0/5 (Q 0.10); no 301-310 run |
| 67 | boxing_food_after_dinner | planner (evolved, program 2bde53a4) | SR 0/10 (Q 0.08) |
| 77 | installing_a_modem | b1k_classical (modem_v10) | native RTX 301-310 SR 8/10 (Q 0.85) |
| 20 | sorting_vegetables | planner (evolved, program f31b2286) | SR 0/10 (Q 0.13) |

### Turning On Radio (task 0)

Planner `cf23f1b5` is packaged in `tasks/turning_on_radio/` and routed to `solutions/planner/backend.py` in
`config/routing.csv`. It uses NumPy and SciPy, installed in the planner Docker venv. Its recorded H100-port result is
SR 9/10 (Q 0.90). Native RTX 4090 validation completed on 2026-10-16: SR 8/10 (Q 0.80). Instances 305 and 310 reached
the 3,225-step horizon without success. All ten JSON/video pairs passed `scripts/check_results.py --metrics-only --strict`.

### Store Honey (task 83)

Planner `ffacbbe6` is packaged in `tasks/store_honey/` and routed to `solutions/planner/backend.py`. Its recorded H100-port
result is SR 7/10 (Q 0.70). Native RTX 4090 validation completed on 2026-10-16: SR 3/10 (Q 0.30). Successes were instances
301 (5,265 steps), 308 (4,515) and 310 (6,243); the other seven reached the 10,151-step horizon. All ten JSON/video pairs
passed `scripts/check_results.py --metrics-only --strict` and are in
`VERA-VLA/b1k_final_submission` at `standard.public.VERA_team.team.20261015` (HF commit `82301cba5c2abd44393aac7cfacb624eb2a6a06f`).

### Vacuuming Floors (task 69)

Planner `434d955b` is packaged in `tasks/vacuuming_floors/` and routed to `solutions/planner/backend.py`. Its recorded H100-port
result is SR 3/10 (Q 0.30). Native RTX 4090 validation completed on 2026-10-16: SR 0/10 (Q 0.00). All ten instances reached
3,618 steps without success. All ten JSON/video pairs passed `scripts/check_results.py --metrics-only --strict` and are in
`VERA-VLA/b1k_final_submission` at `standard.public.VERA_team.team.20261015` (HF commit `b0e59a14be9e6a39731c41a75b7f747d50872085`).

### Chop an Onion (task 42)

Planner `00403145` is packaged in `tasks/chop_an_onion/` and routed to `solutions/planner/backend.py`. Its recorded H100-port
result is SR 1/10 (Q 0.33). Native RTX 4090 validation completed on 2026-10-17: SR 0/10 (Q 0.15). All ten instances reached
the 9,600-step horizon without success. Per-instance Q: 0.00, 0.25, 0.00, 0.75, 0.00, 0.00, 0.50, 0.00, 0.00, 0.00.
All ten JSON/video pairs passed `scripts/check_results.py --metrics-only --strict` and are in
`VERA-VLA/b1k_final_submission` at `standard.public.VERA_team.team.20261015` (HF commit `4ccb33cb0f27092e29b3cbb550bdfe69035690f4`).

### Chopping Wood (task 44)

Planner `b6615d11` is packaged in `tasks/chopping_wood/` and routed to `solutions/planner/backend.py`. Its recorded H100 result is no H100 301-310 run; evolution RTX 301-305 SR 3/5 (Q 0.75). Native RTX 4090 validation: SR 4/10 (Q 0.60). Per-instance Q: 1.00, 0.00, 1.00, 0.00, 0.75, 0.50, 1.00, 1.00, 0.75, 0.00. All ten JSON/video pairs passed `scripts/check_results.py --metrics-only --strict` and are in
`VERA-VLA/b1k_final_submission` at `standard.public.VERA_team.team.20261015` (HF commit `f436dab73e5e60f87218425f364527ca9528a21f`).

### re Shelving Library Books (task 55)

Planner `c243b165` is packaged in `tasks/re_shelving_library_books/` and routed to `solutions/planner/backend.py`. Its recorded H100 result is SR 0/10 (Q 0.33). Native RTX 4090 validation: SR 0/10 (Q 0.07). Per-instance Q: 0.00, 0.00, 0.00, 0.33, 0.00, 0.33, 0.00, 0.00, 0.00, 0.00. All ten JSON/video pairs passed `scripts/check_results.py --metrics-only --strict` and are in
`VERA-VLA/b1k_final_submission` at `standard.public.VERA_team.team.20261015` (HF commit `9aa5d0fa81ed7d7f56b42f8dadc4af3f2f991e8f`).

### Cook Broccolini (task 96)

Planner `cb2a002c` is packaged in `tasks/cook_broccolini/` and routed to `solutions/planner/backend.py`. Its recorded H100 result is SR 0/10 (Q 0.15). Native RTX 4090 validation: SR 0/10 (Q 0.00). Per-instance Q: 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00. All ten JSON/video pairs passed `scripts/check_results.py --metrics-only --strict` and are in
`VERA-VLA/b1k_final_submission` at `standard.public.VERA_team.team.20261015` (HF commit `ff683a7f8053481f30aa43011835224097091a6a`).

### Hanging Pictures (task 34)

Planner `b07a9f5b` is packaged in `tasks/hanging_pictures/` and routed to `solutions/planner/backend.py`. Its recorded H100 result is no H100 301-310 run; evolution RTX 301-310 SR 1/10 (Q 0.10). Native RTX 4090 validation: SR 0/10 (Q 0.00). Per-instance Q: 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00. All ten JSON/video pairs passed `scripts/check_results.py --metrics-only --strict` and are in
`VERA-VLA/b1k_final_submission` at `standard.public.VERA_team.team.20261015` (HF commit `162b355aaccc3eab82e0ca6fc8a7e56a1ff5bac5`).

### Spraying Fruit Trees (task 39)

Planner `0698e8f4` is packaged in `tasks/spraying_fruit_trees/` and routed to `solutions/planner/backend.py`. Its recorded H100 result is H100 port 301-305 SR 0/5 (Q 0.10); no 301-310 run. Native RTX 4090 validation: SR 0/10 (Q 0.00). Per-instance Q: 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00. All ten JSON/video pairs passed `scripts/check_results.py --metrics-only --strict` and are in
`VERA-VLA/b1k_final_submission` at `standard.public.VERA_team.team.20261015` (HF commit `894977f02983a4bde965acd679db09477772c4a2`).

### Boxing Food After Dinner (task 67)

Planner `2bde53a4` is packaged in `tasks/boxing_food_after_dinner/` and routed to `solutions/planner/backend.py`. Its recorded H100 result is SR 0/10 (Q 0.08). Native RTX 4090 validation: SR 0/10 (Q 0.05). Per-instance Q: 0.17, 0.00, 0.00, 0.17, 0.00, 0.00, 0.00, 0.17, 0.00, 0.00. All ten JSON/video pairs passed `scripts/check_results.py --metrics-only --strict` and are in
`VERA-VLA/b1k_final_submission` at `standard.public.VERA_team.team.20261015` (HF commit `9c81e6e2e8f06c4359dd3f808e5ba9437e29611f`).

### Installing a Scanner (task 92)

Planner `7dd85299` is packaged in `tasks/installing_a_scanner/` and routed to `solutions/planner/backend.py`. Its recorded H100 result is SR 4/10 (Q 0.60). Native RTX 4090 validation: SR 3/10 (Q 0.40). Per-instance Q: 0.00, 0.50, 1.00, 0.00, 0.00, 0.00, 1.00, 0.50, 1.00, 0.00. All ten JSON/video pairs passed `scripts/check_results.py --metrics-only --strict` and are in
`VERA-VLA/b1k_final_submission` at `standard.public.VERA_team.team.20261015` (HF commit `5f6b975fe8c9b7757ce858b03529774e17d7bf34`).

### RTX-validated

| 2026 id | Task | Solution | Measured (RTX 4090 unless noted, native evaluator, fresh process per episode, instances 301-310) |
|---:|---|---|---|
| 0 | turning_on_radio | planner (evolved, program cf23f1b5) | SR 8/10 (Q 0.80), validation run `standard.public.VERA_team.team.20261015` |
| 35 | attach_a_camera_to_a_tripod | planner (evolved, program d20842d2) | SR 6/10 (Q 0.60), validation run `standard.public.VERA_team.team.20261016` |
| 83 | store_honey | planner (evolved, program ffacbbe6) | SR 3/10 (Q 0.30), validation run `standard.public.VERA_team.team.20261015` |
| 69 | vacuuming_floors | planner (evolved, program 434d955b) | SR 0/10 (Q 0.00), validation run `standard.public.VERA_team.team.20261015` |
| 42 | chop_an_onion | planner (evolved, program 00403145) | SR 0/10 (Q 0.15), validation run `standard.public.VERA_team.team.20261015` |
| 92 | installing_a_scanner | planner (evolved, program 7dd85299) | SR 3/10 (Q 0.40), validation run `standard.public.VERA_team.team.20261015` |
| 67 | boxing_food_after_dinner | planner (evolved, program 2bde53a4) | SR 0/10 (Q 0.05), validation run `standard.public.VERA_team.team.20261015` |
| 39 | spraying_fruit_trees | planner (evolved, program 0698e8f4) | SR 0/10 (Q 0.00), validation run `standard.public.VERA_team.team.20261015` |
| 34 | hanging_pictures | planner (evolved, program b07a9f5b) | SR 0/10 (Q 0.00), validation run `standard.public.VERA_team.team.20261015` |
| 96 | cook_broccolini | planner (evolved, program cb2a002c) | SR 0/10 (Q 0.00), validation run `standard.public.VERA_team.team.20261015` |
| 55 | re_shelving_library_books | planner (evolved, program c243b165) | SR 0/10 (Q 0.07), validation run `standard.public.VERA_team.team.20261015` |
| 44 | chopping_wood | planner (evolved, program b6615d11) | SR 4/10 (Q 0.60), validation run `standard.public.VERA_team.team.20261015` |
| 89 | installing_a_fax_machine | b1k_classical (m05) | SR 10/10 (Q 1.00) |
| 94 | dispose_of_batteries | b1k_classical (b01) | SR 10/10 (Q 1.00) |
| 1 | picking_up_trash | b1k_classical (t01) | SR 9/10 (Q 0.97) |
| 13 | loading_the_car | b1k_classical (c01) | SR 10/10 (Q 1.00) |
| 65 | unloading_the_car | b1k_classical (u01) | SR 9/10 (Q 0.90) |
| 75 | put_together_a_basic_pruning_kit | b1k_classical (p01) | SR 9/10 (Q 0.93) |
| 2 | putting_away_Halloween_decorations | b1k_classical (h01) | SR 9/10 (Q 0.97) |
| 3 | cleaning_up_plates_and_food | b1k_classical (f01) | SR 7/10 (Q 0.79) |
| 57 | sweeping_garage | b1k_classical (s03) | SR 6/10 (Q 0.70) |
| 37 | clean_a_trumpet | b1k_classical (t01) | SR 6/10 (Q 0.60) |
| 60 | scrubbing_bathroom_floor | b1k_classical (b01) | SR 2/10 (Q 0.20) |
| 80 | clean_up_broken_glass | b1k_classical (g01) | SR 10/10 (Q 1.00) |
| 76 | dispose_of_glass | b1k_classical (w01) | SR 2/10 (Q 0.65) |
| 54 | putting_away_toys | b1k_classical (y01) | SR 4/10 (Q 0.79) |
| 85 | putting_dirty_dishes_in_sink | b1k_classical (k02) | SR 2/10 (Q 0.57) |
| 5 | setting_mousetraps | b1k_classical (v01) | SR 0/10 (Q 0.97) |
| 52 | sorting_bottles_cans_and_paper | b1k_classical (o01) | SR 0/10 (Q 0.09) |
| 25 | clearing_food_from_table_into_fridge | b1k_classical (z02) | SR 0/10 (Q 0.36) |
| 53 | tidying_living_room | b1k_classical (v02) | SR 0/10 (Q 0.23) |
| 79 | polishing_shoes | b1k_classical (h02) | SR 0/10 (Q 0.27) |
| 86 | make_gift_bags_for_baby_showers | b1k_classical (m03) | SR 0/10 (Q 0.18) |
| 82 | store_batteries | b1k_classical (x02) | SR 6/10 (Q 0.80) |
| 11 | putting_dishes_away_after_cleaning | b1k_classical (a01) | SR 0/10 (Q 0.19) |
| 51 | cook_a_brisket | b1k_classical (v20_submit) | SR 2/10 (Q 0.27) |
| 6 | hiding_Easter_eggs | b1k_classical (v40) | SR 4/10 (Q 0.48) |
| 56 | make_rose_centerpieces | b1k_classical (M15) | SR 0/10 (Q 0.40), RTX PRO 6000 Blackwell |

BIMBO fair planners are included when they have at least one full success, or a partial score where no other solution has one
(sorting_vegetables). Oracle planners read the simulator state and are not allowed in the challenge. H100-port measurements
and RTX-validated native evaluator results are shown separately. The challenge score comes only from `scripts/run_eval.sh`
on the native evaluator. Each `tasks/<task>/README.md` and `solution.json` record the commit, evolved program and per-seed
results. Task 35 uses BIMBO_EvoX commit `0ed6e8236c16e9b598d71e827014302c18488018`; its RTX run passed strict result
validation and has one JSON and one MP4 per instance, with SHA256 entries in `artifacts.sha256`. Submission files always go
into HF folder `standard.public.VERA_team.team.20261015`, independent of the validation run's date.

### Submitted to Hugging Face

Native evaluator results currently in the HF submission folder, instances 301-310, one rollout each (21 of 100 tasks;
missing tasks score 0, so the mean Q over 100 tasks is 0.083):

| Task | Policy | Result |
|---|---|---|
| turning_on_radio | planner (evolved, program cf23f1b5) | SR 8/10 (Q 0.80) |
| moving_boxes_to_storage | larchenko_ckpt2 | SR 7/10 (Q 0.75) |
| attach_a_camera_to_a_tripod | planner (evolved, program d20842d2) | SR 6/10 (Q 0.60) |
| cook_hot_dogs | larchenko_ckpt2 | SR 6/10 (Q 0.60) |
| putting_shoes_on_rack | larchenko_ckpt2 | SR 0/10 (Q 0.60) |
| preparing_lunch_box | larchenko_ckpt2 | SR 1/10 (Q 0.58) |
| slicing_vegetables | larchenko_ckpt2 | SR 3/10 (Q 0.49) |
| hiding_Easter_eggs | b1k_classical (v40) | SR 4/10 (Q 0.48) |
| bringing_water | larchenko_ckpt2 | SR 4/10 (Q 0.47) |
| make_rose_centerpieces | b1k_classical (M15) | SR 0/10 (Q 0.40) |
| clean_boxing_gloves | larchenko_ckpt3 | SR 3/10 (Q 0.35) |
| collecting_childrens_toys | larchenko_ckpt2 | SR 1/10 (Q 0.34) |
| putting_up_Christmas_decorations_inside | larchenko_ckpt2 | SR 0/10 (Q 0.32) |
| cook_a_brisket | b1k_classical (v20_submit) | SR 2/10 (Q 0.27) |
| setting_the_fire | larchenko_ckpt2 | SR 1/10 (Q 0.25) |
| tidying_bedroom | larchenko_ckpt2 | SR 0/10 (Q 0.23) |
| picking_up_toys | larchenko_ckpt2 | SR 0/10 (Q 0.20) |
| wash_a_baseball_cap | larchenko_ckpt3 | SR 2/10 (Q 0.20) |
| rearranging_kitchen_furniture | larchenko_ckpt2 | SR 1/10 (Q 0.18) |
| cook_bacon | larchenko_ckpt3 | SR 1/10 (Q 0.16) |
| sorting_vegetables | planner (evolved, program f31b2286) | SR 0/10 (Q 0.07) |

- [JSON results](https://huggingface.co/datasets/VERA-VLA/b1k_final_submission/tree/main/standard.public.VERA_team.team.20261015/json)
- [Episode videos](https://huggingface.co/datasets/VERA-VLA/b1k_final_submission/tree/main/standard.public.VERA_team.team.20261015/videos)

### Frozen RGB-D classical controllers

These five controllers use `frozen_rgbd`, a separate CPU backend with the exact evaluated dependency pins. Their controller files remain unchanged. Existing BIMBO, b1k_classical and VLA routes retain their environments.

| Task ID | Task | Candidate | Native mean Q | Success |
|---:|---|---|---:|---:|
| 87 | collecting_aluminum_cans | a023 | 0.850000 | 3/10 |
| 73 | organizing_school_stuff | s91_integrated_contact_recovery | 0.800000 | 3/10 |
| 59 | organizing_art_supplies | source87 | 0.660000 | 3/10 |
| 90 | composting_waste | c05 | 0.650000 | 3/10 |
| 23 | boxing_books_up_for_storage | books_b01 | 0.183333 | 0/10 |

All numbers above are original native 301–310 results, not integration replay scores. See each task's solution.json for the original image and per-instance metrics. The new combined image requires its own packaging checks; these results do not assert that it was the originally evaluated image.

Source checks: `python3 tools/check_frozen_rgbd.py`. Protocol parity: `tools/replay_frozen_rgbd.py` on saved permitted observations. The independent reference folder must match the original fingerprint. Example (in an environment with the frozen dependency pins):

```bash
python3 scripts/build_docker.py --routing config/routing.frozen_rgbd.csv --tag b1k-evo-submit:five-frozen-review
python3 tools/replay_frozen_rgbd.py --task organizing_school_stuff --reference-root /reference/s91 \
  --observation /fixtures/school.npz --observation-origin 'School s91 instance 301 recorded frame' \
  --port 8000 --steps 80 --output /checks/school.json
```

Replay checks exact actions, episode resets and protocol error recovery on repeated recorded frames. It does not measure new SR/Q or replace the native smoke gate for a new evaluation image. Do not rerun or select individual scored episodes when packaging.

HF artifacts for these tasks join the existing `standard.public.VERA_team.team.20261015/{json,videos}` directories. The current generic `push_hf.sh` uses another layout; use the reviewed prepared upload manifest for this collection.

## Working in this repository

All changes go to the `main` branch: commit on `main` and push it (`scripts/git_push.sh main`). Do not create other branches.

## Layout

```
config/routing.csv           the routing table: all 100 tasks, shared in git; edit your task's row
checkpoints/                 model weights (contents gitignored), e.g. checkpoints/larchenko/checkpoint_2
config/solutions.json        solution registry: backend command, one process per task or per checkpoint, fresh process per episode
server/router.py             the image's policy server: obs["task_id"] -> routing -> backend process (one at a time), passthrough
server/protocol.py           evaluator wire protocol (msgpack + numpy, /healthz), policy calls off the event loop
server/routing.py            routing table check: python3 server/routing.py config/routing.csv
solutions/planner/           backend of the BIMBO planners
solutions/b1k_classical/     backend of the b1k_rl_guys classical controllers (tasks/<task>/controller/)
solutions/frozen_rgbd/       CPU adapter for the five frozen RGB-D controllers, separate dependency pins
solutions/larchenko_2025/    vendored 2025 solution, 2026 adapter, backend (see its README)
tasks/<task>/                one planner per task: root/ (planner tree at its evaluated commit), solution.json, README.md
docker/Dockerfile            base image; scripts/build_docker.py adds the checkpoints and the routing
eval/                        bundled r1pro.yaml and RGBDFullResWrapper (unchanged copies, part of every results package)
scripts/                     build_docker.py, run_eval.sh, test_task.sh, promote_task.py, check_results.py, push_hf.sh, git_push.sh
.env.example                 token names; copy to .env (gitignored, not in the image)
results/, results_tmp/       local only (gitignored): one folder per task, results/<task>/{json,videos,logs}; results_tmp/ holds test runs
tools/import_bimbo_planner.py  (re)import a planner from BIMBO_EvoX; tools/smoke_client.py checks a running image
tools/import_b1k_classical.py  (re)import a classical controller from b1k_rl_guys
tests/                       python3 -m unittest discover tests
```

## 1. Route tasks

```bash
hf download IliaLarchenko/behavior_submission --include "checkpoint_2/*" "checkpoint_3/*" --local-dir checkpoints/larchenko
python3 server/routing.py config/routing.csv      # checks tasks, solutions, checkpoint folders; prints the table
```

`config/routing.csv` is committed and shared by the team. It is plain CSV with no comment lines, so GitHub renders it as a
table. To add a solution, edit your task's row, run the check below and commit on `main`. It lists all 100 tasks, prefilled from the team task tracker (Approach column) with the solutions this
repository has: `planner` for the five tasks above, `larchenko_ckpt2` for the 12 tasks and `larchenko_ckpt3` for the 3 tasks the
tracker assigns to those checkpoints (each is the checkpoint finetuned on that task), `none` elsewhere. The `notes` column keeps
the tracker's approach and SR / Q on indices 0-9 (and marks "Classical" tasks whose code is not in this repository).

Rows are `task,solution,checkpoint`. `solution` is `planner` (needs `tasks/<task>/`), `larchenko_ckpt2|3|4` (2025 tasks only,
needs a checkpoint folder with `params/` and `assets/`), another key of `config/solutions.json`, or `none`. `checkpoint` is a
path relative to the repository: all weights go into `checkpoints/` (the folder is in git, its contents are gitignored; see
`checkpoints/README.md`). Keep paths relative so the shared table works on every machine.

## 2. Build the image

```bash
python3 scripts/build_docker.py --tag b1k-evo-submit:latest        # --dry-run prints the docker command only
docker run --gpus all --rm -p 8000:8000 b1k-evo-submit:latest     # curl localhost:8000/healthz -> OK
```

The build checks the routing, copies each routed checkpoint from `checkpoints/` into `/opt/checkpoints/` through a named build
context (`checkpoints/` itself is outside the main build context, so weights are copied once), bakes the routing table into
`/opt/submit/config/routing.csv` and builds the VLA venv only when a VLA is routed. One 24 GB GPU is enough: the router runs
one backend at a time and switches when the task changes. Planners get a fresh process per episode, as they were evaluated.
With `-e PRELOAD_TASK=<task>`, the router starts that task's backend before the first request.

### Mandatory planner image and RTX smoke gate

Run this gate for every new or changed planner before the full 301-310 validation. Test the exact source tree, routing table,
Dockerfile and immutable image tag that will be used for the full run; do not smoke-test a different checkout or `latest` image.

1. Inspect the whole selected `tasks/<task>/root/` tree, `solution.json`, and planner entrypoint. Check static and dynamic imports,
   subprocesses, environment variables, data/model paths, case-sensitive package paths, and any files loaded relative to the planner.
   Confirm every referenced file is in this package or an explicitly mounted evaluator asset.
2. Check every runtime dependency against the planner venv in `docker/Dockerfile` (and the venv used by any worker subprocess).
   Host-installed packages do not count. Add missing packages and required environment defaults to the Dockerfile or solution config,
   then rebuild the immutable image. For example, a cuRobo planner needs cuRobo, its Torch/CUDA and Warp stack, SciPy when imported,
   and a valid `BIMBO_IK_PYTHON` pointing to the cuRobo-capable planner Python.
3. Build from this repository root with a task-scoped routing table and a unique tag. Start that exact image on an RTX GPU and run
   one native BEHAVIOR-1K v3.9.3-post1 evaluator episode (instance 301, index 0, fresh process, `--num-envs 1`, `--write-video`) using the
   same mounts and environment as the full validation. Use a separate smoke results directory.
4. Before approving the image for the ten-seed run, verify the image health check, router's task-to-planner route, planner imports
   and startup, required worker readiness (including cuRobo GPU IK), evaluator's successful WebSocket connection, and the resulting
   JSON/MP4 with `scripts/check_results.py --strict`. Record the image tag/digest, hardware, instance, logs, success and Q in the
   task README. Fix any missing package, environment variable or path in this repository and repeat the smoke gate.

The RTX smoke is required even when the planner passed on H100 or in the evolution runner: it proves the packaged planner starts
from this directory in this Docker image and connects to the actual evaluator.

## 3. Evaluate and collect results

```bash
B1K=~/BEHAVIOR-1K B1K_PYTHON=python scripts/run_eval.sh store_honey attach_a_camera_to_a_tripod
```

Run this from the repository root. By default it selects only the requested tasks' routes, builds a task-scoped planner Docker
image from this checkout (including the dependencies in `docker/Dockerfile`), starts it with GPU access, waits for `/healthz`,
runs the native evaluator, writes container logs, and removes the container it started. `B1K_PYTHON` is the host Python with
BEHAVIOR-1K v3.9.3-post1 / OmniGibson installed; planner dependencies and planner paths are provided by the Docker image. Each task must
have a non-`none` route in `config/routing.csv`; otherwise the command stops before building. Use `TAG=...` to name the image,
`ROUTING=...` for a different routing table, and `PORT=...` if port 8000 is occupied.

Before a full run for a new or changed planner, complete the mandatory one-episode RTX smoke gate above from this same checkout
and image. Then run the official evaluator (v3.9.3-post1, `RGBDFullResWrapper`, bundled robot config, `--write-video`, default timeouts,
one rollout, `--num-envs 1`) for instances 301-310. By default they run sequentially. For throughput, up to three independent
simulator+planner pairs may run on each GPU; give every pair a unique policy port, container name, runtime/cache directory and
results directory, monitor VRAM, and never reuse an instance index. Each task is its own folder, `results/<task>/{json,videos,logs}`, with
that task's README, wrapper and robot config. Instances that already have a json are skipped (resume); nothing is
re-run.

When the evaluator itself is already running inside an evaluation container and the policy Docker is already up, use
`B1K_EVALUATOR_CONTAINER=1` to bypass Docker orchestration:

```bash
B1K_EVALUATOR_CONTAINER=1 HOST=127.0.0.1 PORT=8000 scripts/run_eval.sh attach_a_camera_to_a_tripod
```

### Test changed code on a task (results_tmp) and promote it

```bash
B1K=~/BEHAVIOR-1K scripts/test_task.sh 83                                                          # task id or name, several allowed
python3 scripts/promote_task.py 83                                                                 # --dry-run, --force
```

`test_task.sh` builds the image from the working tree (`NO_BUILD=1` reuses it, `PORT=` picks the host port), starts it, evaluates
the tasks on instances 301-310 into `results_tmp/<task>/{json,videos,logs}` (gitignored, same layout as `results/`), stops the
container and checks each task folder. `promote_task.py` compares that task with `results/<task>/` and, if the new mean Q is higher
(equal Q: higher SR) and the new set covers the same instances, replaces all of the task's json, videos and logs in
`results/<task>/`. It always moves a task as a whole: the rules forbid keeping the best of repeated runs per instance, so promote
a task after its code changed, not to keep the luckier of two runs of the same code.

`B1K` must be the BEHAVIOR-1K v3.9.3-post1 checkout and `B1K_PYTHON` the python of the env OmniGibson is installed in (needed when
`python` is not on `PATH`); `run_eval.sh` stops if either is missing. Example where the checkout is `~/b1k/BEHAVIOR-1K` with its venv
next to it, reusing an already built image:

```bash
B1K=~/b1k/BEHAVIOR-1K B1K_PYTHON=~/b1k/venv/bin/python NO_BUILD=1 scripts/test_task.sh attach_a_camera_to_a_tripod
```

## 4. Check results

```bash
python3 scripts/check_results.py results/store_honey    # or results/ for every task folder
                                                        # --metrics-only: json + videos only; --strict: warnings fail
```

Checks that the folder name is a 2026 task, file names (that task x public instance x rollout 0), every json against the
evaluator's schema and its file name, q in [0, 1] and success consistency, an MP4 per rollout (ffprobe when installed), and the
package files. It then prints per-task SR / Q and the overall score by the organizers' formula. Exit code 1 on errors.

## 5. Push results and the image to Hugging Face, code to GitHub

Tokens live in `.env` in the repository root. It is gitignored and excluded from the Docker build context, so it never reaches a
push or the image:

```bash
cp .env.example .env && chmod 600 .env     # fill HF_TOKEN (write), HF_REPO (e.g. my-org/b1k-2026-submission), GITHUB_TOKEN
scripts/push_hf.sh --image b1k-evo-submit:latest results/store_honey   # or results/ ; --repo <id> overrides HF_REPO
scripts/git_push.sh                                                # current branch to origin with GITHUB_TOKEN
scripts/push_ghcr.sh b1k-evo-submit:submit-<commit>                # image to ghcr.io/alstar8/b1k-evo-submit with GHCR_TOKEN
```

The challenge portal asks for a Docker image URI (a registry reference such as `ghcr.io/alstar8/b1k-evo-submit:<tag>`) and
optionally its `sha256` digest; `push_ghcr.sh` prints both. A new GHCR package is private: make it public once in its package
settings. `build_docker.py` copies each checkpoint folder in layers of at most 5 GB, so no layer exceeds the registry limit (GHCR: 10 GB).

`push_hf.sh` creates the dataset repo (private) if missing, checks and uploads each task folder (json, videos, evaluator logs,
README, wrapper, robot config) to `results/<task>/`, and uploads the image as `docker/<tag>.tar.gz` (`docker save | gzip`) with load
and run instructions. Results stay out of git; Hugging Face holds the shared copy. The video link for one task is
`https://huggingface.co/datasets/<repo>/tree/main/results/<task>/videos`. `git_push.sh` gives git the token for that one push only
(not stored in `.git/config` or the remote URL) and refuses to push if `.env`, results or checkpoint files are tracked. Values
already set in the environment take precedence over `.env`.

## Adding solutions

- A BIMBO planner: `python3 tools/import_bimbo_planner.py --bimbo <BIMBO_EvoX> --task-dir <id>_<name> --package <fair|fair_ag>
  --commit <commit> [--program evo/runs/<run>/programs/<id>.json --sha <bundle sha256>] [--env K=V]`. The planner tree comes from
  `git archive` of that commit; the evolved modules are checked against the bundle sha256. The four tasks above:

  ```bash
  B=<path to the BIMBO_EvoX checkout>
  python3 tools/import_bimbo_planner.py --bimbo $B --task-dir 35_attach_a_camera_to_a_tripod --package fair --commit 0ed6e82 --program evo/runs/fair-35-r1/programs/d20842d2.json --sha d4b2cae1ad46b9ea0a1cfd8cf2845541361ce003c820b54a49e4fa6fe1035698
  python3 tools/import_bimbo_planner.py --bimbo $B --task-dir 83_store_honey --package fair_ag --commit 106e2be5314eb896d6fa2f224c8666032906dbf5 --program evo/runs/fair-83-r1/programs/ffacbbe6.json --sha fbf438cf79e3af73ee0d076a8eb39d1e3989981a659399a4c0c6641c187d9a13
  python3 tools/import_bimbo_planner.py --bimbo $B --task-dir 98_installing_scanner --package fair_ag --commit 32da8e4035 --program evo/runs/fair-98-r3/programs/7dd85299.json --sha d2174ab2fa78eeb91d2ed574dcd6718510f6a6e8a3456df18f7be689d421c242
  python3 tools/import_bimbo_planner.py --bimbo $B --task-dir 69_vacuuming_floors --package fair --commit 106e2be5314eb896d6fa2f224c8666032906dbf5 --program evo/runs/fair-69-r1/programs/434d955b.json --sha 502bd01ed73a4af5d3284772ee1cfb99792baa6e20f71fd4cc70e00fb1283e8d
  python3 tools/import_bimbo_planner.py --bimbo $B --task-dir 42_chop_an_onion --package fair --commit 106e2be5314eb896d6fa2f224c8666032906dbf5 --program evo/runs/fair-42-r3/programs/00403145.json --sha 65a4628605f0343fbd84310820782238b4185cfa43812a575230e98c3a10ef3c
  python3 tools/import_bimbo_planner.py --bimbo $B --task-dir 55_re_shelving_library_books --package fair_ag --commit 106e2be5314eb896d6fa2f224c8666032906dbf5 --program evo/runs/fair-55-r1/programs/c243b165.json --sha 7f75e2fd5ded0df9e438d3f4062d4b80ba8755c38dff7a27b2bddc8a4a88e1f5
  python3 tools/import_bimbo_planner.py --bimbo $B --task-dir 20_sorting_vegetables --package fair --commit 94db055f --program evo/runs/fair-20-r1/programs/f31b2286.json --sha 4d16dc04504177e4b6c31f9182cfdbe5e8336228fb6913c8a68e4bce1f2a0687
  ```

- A b1k_rl_guys classical controller: `python3 tools/import_b1k_classical.py --b1k <b1k_rl_guys checkout> --task <task>
  --candidate <name> --commit <commit> [--fingerprint <sha256 of the scored runs>] [--results <folder with json/>]`. The
  controller folder `tasks/<task>/experiments/candidate/<name>` comes from `git archive` of that commit into
  `tasks/<task>/controller/`; its source fingerprint (the one b1k_rl_guys records in every scored launch) is checked and kept
  in `solution.json`. The first eleven b1k_classical tasks above, from b1k_rl_guys `fax-manip` at `9d5cb69`:

  ```bash
  B=<path to the b1k_rl_guys checkout>; C=9d5cb69e2e3c015b9532c8f5f2396abe4dec27d6
  for tc in installing_a_fax_machine:m05 dispose_of_batteries:b01 picking_up_trash:t01 loading_the_car:c01 unloading_the_car:u01 \
            put_together_a_basic_pruning_kit:p01 putting_away_Halloween_decorations:h01 cleaning_up_plates_and_food:f01 \
            sweeping_garage:s03 clean_a_trumpet:t01 scrubbing_bathroom_floor:b01; do
    python3 tools/import_b1k_classical.py --b1k $B --task ${tc%%:*} --candidate ${tc##*:} --commit $C
  done
  ```

  Tasks added since, from the b1k_rl_guys branches and commits shown below:

  ```bash
  python3 tools/import_b1k_classical.py --b1k $B --branch pick-and-place --task clean_up_broken_glass --candidate g01 --commit 4448e41fdcc50c8ecfa378d993c0a38f0bf64b17
  python3 tools/import_b1k_classical.py --b1k $B --branch pick-and-place --task dispose_of_glass --candidate w01 --commit 1fb32b358396e7bd82b54774eb540cdf305c9ac7
  python3 tools/import_b1k_classical.py --b1k $B --branch pick-and-place --task putting_away_toys --candidate y01 --commit 3be4f6ae472ba3d0a4411a1a28e5802e5c7c14b7
  python3 tools/import_b1k_classical.py --b1k $B --branch pick-and-place --task putting_dirty_dishes_in_sink --candidate k01 --commit 349e9231edd0cf17e4f138d6b0434e812e034b00
  python3 tools/import_b1k_classical.py --b1k $B --branch pick-and-place --task putting_dirty_dishes_in_sink --candidate k02 --commit 90b1ed1d1770fa73fb5f626bbac036e4027484ad
  python3 tools/import_b1k_classical.py --b1k $B --branch pick-and-place --task setting_mousetraps --candidate v01 --commit 4b0dafd31065da41925bef716e07b4b64bfb3aca
  # (then add instances 302 and 307, crashed in the scored run, as q 0 to its solution.json)
  python3 tools/import_b1k_classical.py --b1k $B --branch pick-and-place --task sorting_bottles_cans_and_paper --candidate o01 --commit ede53534c6ffb87cba37611a35c00e80eff19429
  python3 tools/import_b1k_classical.py --b1k $B --branch pick-and-place --task clearing_food_from_table_into_fridge --candidate z01 --commit 4e67f1817ea9705de4a0313e16be64c51fbb54fb
  python3 tools/import_b1k_classical.py --b1k $B --branch pick-and-place --task clearing_food_from_table_into_fridge --candidate z02 --commit fdab48263355dd864714a2fa66e7434dd8f9ca98
  python3 tools/import_b1k_classical.py --b1k $B --branch pick-and-place --task tidying_living_room --candidate v01 --commit 9ecd841487af7b0d8162289cfab0ccbf5ec22862
  python3 tools/import_b1k_classical.py --b1k $B --branch pick-and-place --task tidying_living_room --candidate v02 --commit 84354487f4d9806f5013dfda8bc780261d983952
  python3 tools/import_b1k_classical.py --b1k $B --branch pick-and-place --task polishing_shoes --candidate h02 --commit 54d6cd5fac876f781e95f3fd65c8d4f882ff5bdd
  python3 tools/import_b1k_classical.py --b1k $B --branch pick-and-place --task make_gift_bags_for_baby_showers --candidate m02 --commit 3af2081e2114cc5a75c787b0b447874ceeb2bcb4
  python3 tools/import_b1k_classical.py --b1k $B --branch pick-and-place --task store_batteries --candidate x01 --commit e190f5c860c593a359690fb4303f85b91bea6d03
  python3 tools/import_b1k_classical.py --b1k $B --branch pick-and-place --task store_batteries --candidate x02 --commit 63d6172b9ea2fc8d3f61602f5c60d16080813751
  python3 tools/import_b1k_classical.py --b1k $B --branch pick-and-place --task make_gift_bags_for_baby_showers --candidate m03 --commit 30eae0165077f1824361f02df64a5d51544fdb97
  python3 tools/import_b1k_classical.py --b1k $B --branch pick-and-place --task putting_dishes_away_after_cleaning --candidate a01 --commit 7906a45943e81f4a153140469c8ad8361bb64e2f
  python3 tools/import_b1k_classical.py --b1k $B --branch main --task cook_a_brisket --candidate v20_submit --commit 09f820ca821dab56932047c0cf3945118464e6db
  python3 tools/import_b1k_classical.py --b1k $B --branch main --task hiding_Easter_eggs --candidate v40 --commit 09f820ca821dab56932047c0cf3945118464e6db
  python3 tools/import_b1k_classical.py --b1k $B --branch pick-and-place-wingrune --task make_rose_centerpieces --candidate M15 --commit 2b611cb8623e1fca9c00ed2fda9e6616fc336d33 --source-path tasks/make_rose_centerpieces/m15_validation_20261016/controller --adapter rose_m15
  ```

  Rose M15 uses `--source-path` for its frozen package and `--adapter rose_m15` for the h32 Policy API in the common
  classical backend. Its import keeps runtime code and static priors, excludes reports and unused variants, and records
  the resulting package fingerprint. Add `--results <native results folder with json/>` to retain the measured per-instance
  metrics when regenerating `solution.json`.

- Another model: add an entry to `config/solutions.json` (backend command with `{repo} {task} {port} {checkpoint}`). The backend
  serves the evaluator protocol on `{port}`; `server.protocol.serve(policy, port)` with `policy.reset()` / `policy.act(obs)` is
  enough. Then route tasks to it in the CSV. A new Python environment goes into `docker/Dockerfile`.

## Rules this package follows

- Observations: only what `RGBDFullResWrapper` sends (RGB, depth, proprioception, camera poses relative to the base, task id).
  Planners use only static priors of the manipulated object assets (collision geometry, marker frames, in their own frame; see
  each `data/PROVENANCE.json`), no instance pose and no simulator state.
- Robot: bundled `r1pro.yaml` unchanged (`grasping_mode: assisted`), 23-d actions.
- One rollout per instance, default timeouts, no editing of json or videos, no cherry-picking (`run_eval.sh` never re-runs a
  finished instance).
