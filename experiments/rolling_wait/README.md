# Selective waiting across an 84-round surface-code memory

This experiment applies the single-commit burst-alert policy from v0.6.0 at every eligible commit in a longer distance-7 memory. It checks how often waits accumulate in a burst-free trial and whether the logical error reduction holds when burst timing varies across the stream.

## Results

| Policy | Burst-free trials with any extra wait | Fewer logical failures per million memory trials (95% interval) |
|---|---:|---:|
| Fixed deadline | 0% | Reference |
| Always wait | 100% | 0.513 (0.217 to 0.808) |
| Selective, original threshold of 28 | 5.396% | 0.513 (0.217 to 0.808) |
| Selective, recalibrated threshold of 32 | 0.838% | 0.537 (0.244 to 0.831) |

The recalibrated policy waited on 50,302 of 6 million burst-free trials (95% interval 0.831% to 0.846%), below the 1% full-trial limit. Its gain exceeded 80% of the always-wait gain by 0.127 fewer failures per million trials (paired 95% interval 0.061 to 0.193). The direct comparison with always waiting is 0.025 fewer failures per million (−0.012 to 0.062). The population estimates assume a burst in 0.1% of memory trials.

| Policy | Absolute logical failure rate without a burst | Absolute logical failure rate with a burst |
|---|---:|---:|
| Fixed deadline | 8.5732% | 38.6636% |
| Always wait | 8.5732% | 38.6082% |
| Selective, original threshold | 8.5732% | 38.6082% |
| Selective, recalibrated threshold | 8.5732% | 38.6082% |

Use the paired estimates for policy differences. The recalibrated policy reduced failures on burst trials by 0.0554 percentage points (0.0261 to 0.0847). Its estimated change on burst-free trials was an increase of 0.0167 failures per million (0.0009 to 0.0326). The absolute estimates in the table use all 2.52 million burst trials and a prespecified 500,000-trial subset of burst-free trials. The paired differences and wait distributions use all 8.52 million trials.

## Where the gain occurs

The arrival phase is the number of rounds from the last nominal deadline strictly before a burst's scheduled true-alert arrival. Each phase has nine onset times and 360,000 burst trials. A true alert is generated with probability 0.90 at the scheduled time.

| Scheduled arrival | Recalibrated policy's logical failure reduction on burst trials (percentage points, 95% interval) | Burst trials with any extra wait |
|---|---:|---:|
| 1–3 rounds after the nominal deadline | 0.1290 (0.0606 to 0.1973) | 96.29% |
| 4–7 rounds after the nominal deadline | 0.0002 (−0.0003 to 0.0007) | 54.25% |

The second group can trigger waits because a burst can raise detection counts in adjacent candidate onset intervals. Its true alert falls outside the extra three rounds. In an exploratory check excluding the seven onset times associated with the original round-28 deadline, the reduction on the remaining burst trials was 0.0632 percentage points (0.0321 to 0.0942). Individual deadlines have wider intervals. All timing breakdowns use pointwise intervals.

On burst-free trials, the recalibrated policy added three rounds at one commit in 0.8353% of trials and six rounds across two commits in 0.00305% of trials. The mean sum of added rounds was 0.0252 on burst-free trials and 0.0274 in the assumed mixture. Always waiting adds three rounds at each of nine commits, for a sum of 27 rounds. These totals measure the sum of decision deadline extensions. Each trial ends at the scheduled readout after round 84.

## Design

- The rotated distance-7 logical X memory uses 84 syndrome rounds, a circuit noise parameter of 0.003, and a three-round global burst with parameter 0.020. The single burst onset is uniform over rounds 8–70. This yields nine complete seven-round onset blocks and covers all seven arrival phases equally.
- The same 14-round alert delay, 90% sensitivity, and independent 0.0001 false-alarm probability per other candidate timestamp apply to all policies. Candidate timestamps in this study range from round 8 to round 70. Every alert can arrive by the scheduled final readout.
- Windows commit seven rounds at a time and use seven rounds of syndrome lookahead. Nominal deadlines from round 21 through round 77, in steps of seven, are eligible for an extension because candidate alerts can arrive in the following three rounds. For each of these windows the fixed policy commits at the nominal deadline, the always-wait policy commits three rounds later, and the two selective policies decide whether to wait at the nominal deadline. The syndrome windows, correction boundaries, and subsequent nominal deadlines are shared.
- The trigger examines the three onsets whose alerts could arrive during an extension. It takes the maximum four-slice detection count for those onsets. The original threshold is fixed at 28. On 200,000 separate burst-free calibration trials, threshold 32 was the smallest integer that triggered at any eligible commit on at most 0.9% of trials. Threshold 31 triggered on 1.351% and threshold 32 on 0.827%. This leaves a margin below the 1% evaluation limit. Both thresholds were frozen before the test.
- The matching-graph likelihood ratios use a separate set of 60,000 calibration trials per hypothesis, including the background. At each deadline, the decoder uses syndrome data through the nominal window and the alert bits available by its chosen deadline. Matching decisions carry boundary corrections forward through the stream. A later decision remains earlier than the following nominal deadline.

The result applies to one simulated distance, at most one synthetic global burst per trial, and a deterministic alert delay. The comparison isolates the effect of admitting later alerts with fixed syndrome geometry. Delay is measured in syndrome rounds.

## Paired estimation

Each onset contributes 40,000 test trials. The burst-free group contributes 6 million. All four policies share the underlying syndromes, logical labels, and alert masks. On burst-free trials the code evaluates the no-false-alert case and a conditionally sampled nonempty false-alert mask, using their exact weights. This improves precision when a change depends on a rare false alert. The full graph sequence is decoded whenever policies choose different graphs. For paired differences only, matching sequences are known to give identical logical outcomes and can be skipped. The absolute failure sample is selected by chunk number in the seed plan before simulation.

For a policy with failure indicator $L$, the gain is $L_{\mathrm{fixed}}-L$. The recovery test uses the selective gain minus 0.8 times the always-wait gain, evaluated on the same trials. Positive values favor selective waiting relative to the recovery target. Population estimates weight the burst-free stratum by 0.999 and each onset by $0.001/63$. The normal-approximation pointwise 95% intervals use stratified test-shot variance conditional on calibration. Zero-width intervals identify strata with no observed paired changes and describe the sample.

## Reproduce

From the extracted folder, use Python 3.12, 3.13, or 3.14 with the pinned dependencies. The packaged simulation was run and checked on Linux with Python 3.12.14. If your Mac exposes Python 3.14 as `python3.14`, use that command to create the virtual environment.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest -v
.venv/bin/python verify_rolling_wait.py results

OPENBLAS_NUM_THREADS=1 .venv/bin/python run_rolling_wait.py \
  --seed 2026111311 --workers 8 --output rerun
.venv/bin/python verify_rolling_wait.py rerun
.venv/bin/python plot_rolling_wait.py rerun rerun/rolling_wait_comparison.png
```

The simulation uses the CPU. The packaged run took about three minutes on eight workers. You can lower `--workers` for a smaller machine. Each test chunk keeps its own seeds and size. Stim's seeded samples can vary across software versions, platforms, or batch sizes. Choose a fresh output directory for each run.

| File | Contents |
|---|---|
| `results/summary.csv` | Absolute failure rates, paired gains, recovery margins, and waits by cohort, arrival phase, nominal deadline, and onset |
| `results/delay_distribution.csv` | Probability of each sum of added rounds, including zero |
| `results/run.json` | Settings, seed streams, deadlines, and trigger calibration histogram |
| `results/paired_moments.csv` | First and second moments and trial counts for each test chunk |
| `results/delay_counts.csv`, `results/commit_waits.csv` | Raw delay counts by chunk and wait counts by commit |
| `results/rolling_wait_comparison.png` | Population gains, burst-free waits, and the breakdown by arrival phase |
