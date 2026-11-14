# Selective waiting for delayed surface-code burst alerts

The experiment tests whether a syndrome trigger can preserve the logical error reduction from a three-round deadline extension while using that extension on at most 1% of ordinary trials. All three policies use the same test trials.

## Results

| Policy | Fewer logical failures per million trials relative to the fixed deadline (95% interval) | Ordinary trials with a deadline extension |
|---|---:|---:|
| Fixed deadline | Reference | 0% |
| Always wait | 0.798 (0.284 to 1.312) | 100% |
| Selective wait | 0.802 (0.288 to 1.316) | 0.614% |

The selective policy extended the deadline on 147,406 of 24 million ordinary trials (95% interval 0.611% to 0.617%). Its gain exceeded 80% of the always-wait gain by 0.164 fewer failures per million trials (paired 95% interval 0.060 to 0.268). Directly comparing selective with always waiting gives a difference of 0.004 fewer failures per million (−0.016 to 0.024). The population estimates assume a 0.1% burst probability per memory trial.

## Design

- The simulated rotated surface code has distance 7 and 42 syndrome rounds in the logical X memory circuit. The background circuit noise parameter is 0.003 and the burst parameter is 0.020 for three consecutive rounds. A burst starts uniformly among rounds 15 through 21.
- The sensor reports each candidate timestamp after a 14-round delay. Its sensitivity is 90% at the true onset. The independent false-alarm probability is 0.0001 per other candidate timestamp.
- The target window uses syndrome data from rounds 15 through 28 and commits through round 21. The fixed policy makes this decision at round 28. The always-wait policy accepts alert evidence through round 31. The selective policy chooses between those deadlines at round 28. The window geometry is shared across all policies to isolate the availability of alert evidence. Only alerts for onsets 15, 16, and 17 can arrive during the added three rounds.
- The trigger sums detection events over rounds 15–18, 16–19, and 17–20, and takes the largest of the three sums. It waits when this score is at least 28. That was the smallest integer threshold giving at most 0.9% waits on 200,000 separate ordinary calibration trials. The observed calibration rate was 0.611%. The 0.9% calibration budget leaves room for sampling variation below the 1% evaluation target.

The simulation measures a deadline extension at one commit under a synthetic noise and alert model. Waiting time is expressed in syndrome rounds. Hardware decoding time is outside the experiment.

## Paired estimates

The matching-graph scores use 80,000 calibration trials per noise hypothesis, separate from trigger calibration and testing. The test uses 24 million ordinary trials and 120,000 burst trials at each of seven onset times, totaling 24.84 million. The policies share the sampled syndromes, logical outcomes, and alert masks. For ordinary trials, the no-false-alert case and a conditionally sampled nonempty false-alert mask are combined with their exact stratum weights.

If $D$ is the fixed-policy logical failure indicator minus the always-wait indicator and $I$ indicates whether the trigger waits, then $ID$ is the selective gain. The direct test of the 80% criterion uses $ID - 0.8D$. The reported intervals are approximate pointwise 95% intervals from paired test-shot variance conditional on both calibrations. Population estimates weight ordinary trials by 0.999 and each onset stratum by $0.001/7$.

## Reproduce

Run these commands from the extracted folder with Python 3.12, 3.13, or 3.14. The packaged results were generated and checked with Python 3.12.14 on Linux. The simulation runs on the CPU.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest -v
.venv/bin/python verify_selective_wait.py results

OPENBLAS_NUM_THREADS=1 .venv/bin/python run_selective_wait.py \
  --seed 2026111310 --workers 8 --output rerun
.venv/bin/python verify_selective_wait.py rerun
.venv/bin/python plot_selective_wait.py rerun rerun/selective_wait_comparison.png
```

You can lower `--workers` for a smaller machine. Each chunk keeps its own seed and size. Stim's seeded output can vary across platforms and simulator versions. Use a fresh output directory for each run.

| File | Contents |
|---|---|
| `results/summary.csv` | 15 estimates for ordinary, burst, and population-weighted comparisons |
| `results/paired_moments.csv` | Trial counts and paired moment sums by onset, chunk, and metric |
| `results/run.json` | Circuit and alert settings, calibration histogram, commit schedules, software versions, and seeds |
| `results/selective_wait_comparison.png` | Population gain and ordinary wait rate with 95% intervals |
