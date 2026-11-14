# Reproduce the surface-code burst-alert study

## Archived results

From the repository root, run `.venv/bin/python scripts/verify_package.py` to validate the five data sets. The four experiment directories contain the original simulation and verification code from the source versions listed in the [README](../README.md). Each is kept self-contained so it uses the same decoder helpers as its archived run. Inputs to v0.8.0 include the frozen rolling-wait design and per-commit counts used to construct the random-wait baseline.

| Run | Burst-free trials | Burst trials | Trigger threshold |
| --- | ---: | ---: | ---: |
| Distance 7, single commit, 42 rounds | 24,000,000 | 840,000 | 28 |
| Distance 7, rolling wait, 84 rounds | 6,000,000 | 2,520,000 | 32 |
| Distance 7, matched wait, 84 rounds | 3,000,000 | 2,520,000 | 32 |
| Distance 9, robustness, 84 rounds | 1,800,000 | 5,040,000 | 46 / 48 |
| Distance 9, replication, 84 rounds | 1,800,000 | 5,040,000 | 46 / 48 |

Thresholds of 46 and 48 apply to the fixed and variable delay laws, respectively. The rolling-wait run also evaluates the transferred threshold of 28. Within each distance-9 run, the two delay conditions share the sampled syndromes, the global and local conditions share the burst-free trials, and the burst trials split equally across spatial scopes. The distance-9 diagnostic uses an additional 1,008,000 burst trials per scope.

| File | What it records |
| --- | --- |
| `run.json` | Settings, decision schedule, calibration metadata, software versions, and seed plan |
| `paired_moments.csv` | Sample counts and the first and second paired moment sums by simulation job and metric |
| `summary.csv` | Failure rates, gains, wait metrics, and pointwise intervals by analysis cohort |
| `commit_waits.csv`, `delay_counts.csv`, `delay_distribution.csv` | Commit-level waits and added-round distributions for the rolling-wait and matched-wait experiments |
| `trigger_capture.csv`, `trigger_capture_by_onset.csv` | Distance-9 trigger diagnostic, aggregate and by onset |

In the CSVs, `ordinary` means burst-free, `jitter` denotes a discrete uniform delay of 12–16 rounds, and `both_vs_lookahead` denotes the matched comparison that adds late alert evidence. Positive values of this contrast mean fewer logical failures. Multiply its `population` estimate by one million for failures per million trials or its `burst` estimate by 100 for percentage points.

The independent replication directory contains the three files returned by the local distance-9 run. The trigger diagnostic files accompany the original distance-9 run. Its independent diagnostic root seed is 2026111314.

## Figures

```bash
.venv/bin/python scripts/build_figures.py
```

| Output | Inputs |
| --- | --- |
| `figure_1.png` | Round-28 example with the fixed 14-round reporting delay |
| `figure_2.png` | Population rows for `selective_vs_random`, `selective_vs_syndrome`, and `both_vs_lookahead` from [distance-7 matched-wait summary](../experiments/matched_wait/results/summary.csv) |
| `figure_3.png` | Population rows for `both_vs_lookahead` under all four scope and delay combinations from the [original](../experiments/distance9/results/summary.csv) and the [replication](../experiments/distance9/replication/summary.csv) |

The plotted error bars are pointwise 95% intervals from the paired analysis. Figure 3 uses a separate horizontal scale for each scope. The PNGs bundled in [figures](../figures) are the figures in the manuscript.

## Rerun the simulations

These commands run the paper-scale experiments locally on the CPU. Install the environment as shown in the [README](../README.md) and run from the repository root. The commands use the archived seeds, calibration sizes, test sizes, and batch sizes. Choose a fresh output directory for each rerun. Lower `--workers` to fit available CPU cores or memory.

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
qec_python=".venv/bin/python"

"$qec_python" experiments/single_commit/run_selective_wait.py \
  --seed 2026111310 --workers 8 --output reproduced/single_commit
"$qec_python" experiments/single_commit/verify_selective_wait.py reproduced/single_commit

"$qec_python" experiments/rolling_wait/run_rolling_wait.py \
  --seed 2026111311 --workers 8 --output reproduced/rolling_wait
"$qec_python" experiments/rolling_wait/verify_rolling_wait.py reproduced/rolling_wait

"$qec_python" experiments/matched_wait/run_matched_wait.py \
  --seed 2026111312 --workers 6 --output reproduced/matched_wait
"$qec_python" experiments/matched_wait/verify_matched_wait.py reproduced/matched_wait

"$qec_python" experiments/distance9/run_robustness.py \
  --seed 2026111313 --workers 8 --output reproduced/distance9
"$qec_python" experiments/distance9/diagnose_trigger.py reproduced/distance9 \
  --seed 2026111314 --workers 2
"$qec_python" experiments/distance9/verify_robustness.py reproduced/distance9

"$qec_python" experiments/distance9/run_robustness.py \
  --seed 2026111410 --workers 8 --output reproduced/distance9_replication
"$qec_python" experiments/distance9/verify_robustness.py reproduced/distance9_replication
```

To continue an interrupted distance-9 run, repeat its command with the same seed, output directory, and sampling settings, adding `--resume`. The commands retain the simulator's archived batch sizes. Stim's seeded samples can vary with simulator version, platform, or batch size. The archived paired moments preserve the exact inputs to the manuscript's statistical analysis.
