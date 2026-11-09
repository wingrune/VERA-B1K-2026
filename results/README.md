Results live here locally and are never committed (gitignored). One folder per task:

```
results/<task>/json/<task>_<instance>_0.json
results/<task>/videos/<task>_<instance>_0.mp4
results/<task>/logs/<task>_<instance>.log
```

plus `README.md`, the wrapper `.py` and the robot config in that same task folder. `results_tmp/<task>/` has the same layout for test runs. `scripts/run_eval.sh` writes it, `scripts/check_results.py results/<task>` (or `results/` for every task) checks it, `scripts/push_hf.sh` uploads it.
