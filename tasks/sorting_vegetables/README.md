# sorting_vegetables (2026 task 20)

Motion planner on RGB-D + proprioception (BIMBO_EvoX fair planner), served by `solutions/planner/backend.py`.

- Source: BIMBO_EvoX `94db055f25`, `tasks/20_sorting_vegetables/planners/fair`
- Package in this folder: `root/tasks/20_sorting_vegetables/planners/fair` (plus the shared `core/skills_lib`, `skills/gpu-ik` and the IK worker of the same commit)
- Evolution: run `fair-20-r1`, program `f31b2286` (iteration 442), bundle sha256 `4d16dc04504177e4...`
- Measured (H100 port, seeds 301-310): SR 0/10, Q 0.23 0.31 0.00 0.15 0.00 0.15 0.31 0.15 0.00 0.00
- Note: replaces 108c64cb (iteration 150, GAPF 0.1216, H100 mean Q 0.108); f31b2286 GAPF 0.1791, never below 108c64cb on a seed both scored (308: 0.15 vs 0.08); seeds 303 and 305 were infra in evolution and are recorded as 0
- RTX 4090, native evaluator (BEHAVIOR-1K v3.9.3-post1, RGBDFullResWrapper, bundled r1pro.yaml, default horizon 17855 steps), image `b1k-evo-submit:submit-90d14f3` (`sha256:9e8c1e7dd895c2e20001c3816ba34ffbe01a859acfbb850d7dcbb0c4421ea7ec`), instances 301-310: SR 0/10, mean Q 0.069. Per-instance Q: 301=0.00, 302=0.00, 303=0.08, 304=0.08, 305=0.00, 306=0.00, 307=0.31, 308=0.08, 309=0.15, 310=0.00; every episode ran to the horizon. All 10 JSON and 10 MP4 passed `scripts/check_results.py --strict`. These are the 301-310 results in the Hugging Face submission folder (HF commit `ae26fa853ba4ac900ce56b143ca28aa30fb4f789`); they replaced the native results of 99e5cd85 (mean Q 0.085), a program that is not in the image.
- Note: each episode starts a fresh cuRobo IK worker; with an empty Warp kernel cache its warm-up took ~380 s on the RTX 4090 (the evaluator waits for the first action, so no simulator steps are lost).

Regenerate with `tools/import_bimbo_planner.py` (the command is in the repository README).
