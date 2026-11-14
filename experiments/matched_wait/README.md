# Delayed burst alerts at a matched wait budget

This study compares ways to spend a three-round extension at a sliding-window decoding commit. It reuses the threshold of 32 detection events from the 84-round rolling study (v0.7.0) and evaluates it on a fresh set of simulated trials.

## Results

| Paired comparison | Fewer logical failures per million memory trials (95% interval) |
|---|---:|
| Selective alert waiting over random alert waiting | 0.582 (0.288 to 0.876) |
| Selective alert waiting over selective syndrome-only lookahead | 0.681 (0.351 to 1.012) |
| Selective alert waiting over selective lookahead with pre-deadline alerts | 0.571 (0.278 to 0.865) |
| Selective lookahead with all available alerts over selective lookahead with pre-deadline alerts | 0.571 (0.278 to 0.865) |
| Selective lookahead with all available alerts over selective alert waiting | 0.000 (−0.006 to 0.006) |

The realized mean added delay was 0.02730 rounds per memory trial for each selective policy (0.02699 to 0.02761) and 0.02713 for random waiting (0.02681 to 0.02745). The random schedule's design expectation was 0.02740 rounds. The comparison with syndrome lookahead matches waits on each trial. The random control matches the mean budget over the assumed mixture. On burst trials the selective trigger averaged 2.188 added rounds and random waiting averaged 0.027, which shows how the trigger allocates the budget to rare events.

In the new test, selective alert waiting delayed 0.835% of burst-free trials (95% interval 0.825% to 0.845%). Relative to the fixed deadline with alerts, it reduced logical failures by 0.571 per million (0.278 to 0.865). Its gain exceeded 80% of the always-wait gain by 0.130 per million (0.062 to 0.198).

The combined policy reduced logical failures on burst trials by 0.137 percentage points (0.069 to 0.205) when the true alert was due within the extension, compared with selective lookahead using pre-deadline alerts. For later arrivals the reduction was 0.0002 percentage points (−0.0002 to 0.0007). The absolute population logical failure rate was about 8.62% for all policies at the displayed precision. The paired differences resolve effects that are much smaller than the individual failure rates.

## Policies

| Policy | Wait decisions | Syndrome data | Alert evidence |
|---|---|---|---|
| Fixed deadline with alerts | No extensions | Nominal window | Through the nominal deadline |
| Always wait for alerts | All nine eligible commits | Nominal window | Through the extended deadline |
| Random wait for alerts | A random draw at each eligible commit | Nominal window | Through the chosen deadline |
| Selective wait for alerts | Frozen trigger | Nominal window | Through the chosen deadline |
| Fixed syndrome-only | No extensions | Nominal window | None |
| Selective syndrome-only | Frozen trigger | Through the chosen deadline | None |
| Selective lookahead with pre-deadline alerts | Frozen trigger | Through the chosen deadline | Through the nominal deadline |
| Selective lookahead with all available alerts | Frozen trigger | Through the chosen deadline | Through the chosen deadline |

The four selective policies have identical wait decisions on each trial. The random policy draws independently of the syndromes, burst timing, and sensor. Its nine per-commit probabilities are frozen from the v0.7.0 results, weighted for a 0.1% chance of a burst per trial. They imply 0.027404 added rounds per memory trial in expectation. Added delay sums extensions across a memory trial. The nominal commit boundaries and the scheduled final readout apply to every policy.

The comparison with random waiting measures the value of allocating the alert budget using syndromes. The comparison with syndrome-only lookahead measures the full policy difference at identical waits. The comparison with lookahead with pre-deadline alerts holds that earlier sensor evidence fixed. Comparing the two lookahead policies measures the incremental value of late alerts given the extra syndrome data.

## Simulation and inference

The model uses a rotated distance-7 logical X memory with 84 syndrome rounds. The circuit error parameter is 0.003 in the background and 0.020 during a three-round global burst. Given a burst, its onset is uniform over rounds 8–70. A true alert is generated with probability 0.90 and arrives 14 rounds after the onset. The independent false-alert probability is 0.0001 per other timestamp. Nominal deadlines run from round 21 to 77 in steps of seven for the eligible windows.

The trigger uses four-slice detection counts for the three onsets whose alerts could arrive during the extension. The code reconstructs the v0.7.0 graph-selection tables from their saved calibration seeds and batch size. Each syndrome lookahead policy can use the extra measurements both to select a noise hypothesis and to decode the current window. The same correction and boundary-carry rules apply to every policy.

Each onset contributes 40,000 test trials and the burst-free stratum contributes 3 million, totaling 5.52 million paired trials. All policies receive the same syndromes and logical labels. The alert policies also share the generated alert bits. For burst-free trials the decoder integrates the no-false-alert case and a sampled nonempty false-alert mask using their known probability weights. Separate RNG streams control each trial's syndrome, alert, and random wait. The test seeds are disjoint from v0.7.0.

Positive paired differences denote fewer logical failures for the second policy in the comparison. Population estimates weight burst-free trials by 0.999 and each onset by 0.001/63. The reported 95% intervals are pointwise normal approximations from paired, stratified test-shot variance conditional on the calibration. A zero-width interval can occur when a stratum has no observed paired changes. The analysis measures added syndrome rounds in one simulated noise setting.

## Reproduce

Use Python 3.12, 3.13, or 3.14 with the pinned dependencies. The packaged results were generated and checked with Python 3.12.14 on Linux. Use `python3.14` on your Mac if that is how you invoke your installation.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest -v
.venv/bin/python verify_matched_wait.py results

OPENBLAS_NUM_THREADS=1 .venv/bin/python run_matched_wait.py \
  --seed 2026111312 --workers 6 --output rerun
.venv/bin/python verify_matched_wait.py rerun
.venv/bin/python plot_matched_wait.py rerun rerun/matched_wait_comparison.png
```

The simulation runs on the CPU. The packaged run took about six minutes with six workers. Lower `--workers` if your machine has less memory. Choose a fresh output directory. Stim's seeded samples can vary with platform, simulator version, or batch size.

| File | Contents |
|---|---|
| `priority1_design.json` and `priority1_commit_waits.csv` | Frozen schedule, calibration, trigger threshold, and wait counts used to set the random baseline |
| `results/run.json` | Test settings, seed plan, random wait probabilities, and software versions |
| `results/summary.csv` | Absolute failure rates, paired differences, and delays by cohort, arrival phase, deadline, and onset |
| `results/paired_moments.csv` | Trial counts and first and second moment sums by chunk |
| `results/delay_distribution.csv`, `results/delay_counts.csv`, `results/commit_waits.csv` | Distribution of summed extensions and wait counts by commit |
| `results/matched_wait_comparison.png` | Paired contrasts and realized wait budgets |
