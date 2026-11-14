# Distance-nine robustness of selective waiting for burst alerts

This study crosses burst scope (global or localized) with alert timing (fixed or variable). The primary paired comparison adds late alerts to a decoder that already uses additional syndrome lookahead and the alerts available at the nominal deadline. Both policies wait on exactly the same trials and at the same commits.

## Results

| Burst scope | Alert delay | Fewer failures per million memory trials (95% interval) | Fewer failures on burst trials, percentage points (95% interval) | Trigger capture among true alerts that could arrive during an extension |
|---|---|---:|---:|---:|
| Global | 14 rounds | 0.919 (0.607 to 1.230) | 0.0922 (0.0611 to 0.1234) | 99.90% |
| Global | 12–16 rounds | 0.896 (0.604 to 1.187) | 0.0896 (0.0605 to 0.1187) | 99.82% |
| Localized | 14 rounds | −0.005 (−0.029 to 0.020) | 0.00091 (−0.00086 to 0.00268) | 3.49% |
| Localized | 12–16 rounds | −0.009 (−0.028 to 0.011) | 0.00052 (−0.00087 to 0.00190) | 2.28% |

The gain for global bursts persisted with jitter. On the paired global-burst samples, the difference in gain between jitter and fixed delay was −0.0027 percentage points (−0.0282 to 0.0229). The localized-burst intervals center close to zero. Under variable delay, always waiting still reduced localized-burst failures by 0.0112 percentage points (0.0076 to 0.0149) relative to the fixed deadline. Selective alert waiting fell short of the 80% recovery target for localized bursts under both timing models. The burst-free local-decoder estimate is based on a handful of changes under the weighted false-alert draws, so its small negative population point estimate warrants caution.

The independent trigger diagnostic uses 1.008 million burst trials per scope. It averages exactly over the reporting delays and conditions on a generated true alert being able to arrive during a potential extension. Capture means the trigger opens that particular extension. The burst-free probability of any added wait was 0.676% (0.664% to 0.688%) with fixed delay and 0.612% (0.600% to 0.623%) with jitter.

## Design

| Setting | Simulation |
|---|---|
| Memory | Rotated distance-9 logical X memory, 84 syndrome rounds |
| Noise | Circuit-level depolarization and check readout noise, 0.003 in the background and 0.020 during the burst at affected locations |
| Burst | Three rounds, onset uniform over rounds 8–70 conditional on a burst |
| Global | All circuit locations that would receive the v0.8 burst |
| Localized | One of four equally likely 25-qubit patches, including data and check qubits |
| Alert timing | Fixed at 14 rounds, or independently uniform over 12, 13, 14, 15, and 16 rounds after each candidate timestamp |
| Sensor | 90% probability of an alert for the true onset, and independent false-alarm probability 0.0001 per other timestamp |
| Decoder windows | Seven committed rounds, seven nominal rounds of lookahead, three extra rounds on a wait |
| Trial mixture | 0.1% burst probability per memory trial |

The four patches are squares in Stim's layout, each with Chebyshev radius 3 in its coordinate units. Each contains 25 of the 161 active data and check qubits. A single-qubit location uses the elevated rate if its qubit is in the patch. A two-qubit location uses the elevated rate if either participant is in the patch. Each patch is stationary during the three-round burst. The localized burst raises the rate at 369 circuit locations and the global burst at 2,181, so the spatial comparison also changes the total fault exposure. Localized impacts and spatial spreading have been observed in superconducting hardware. The simulated stationary patch is a controlled model of spatial concentration. [McEwen et al., Nature Physics 18, 107–111](https://doi.org/10.1038/s41567-021-01432-8)

The decoder knows the burst scope and delay distribution for its run. A positive alert reports only the candidate onset. For the localized model the onset likelihoods are fitted to the mixture over patches. The matching graph for each onset uses the four-patch mean error probability on each edge. This edge approximation allows matching with a single graph for an unknown patch. The same decoder assumptions apply to all compared policies within that scenario. The likelihood contribution of silence at time $t$ uses the fraction $F(t-o)$ of allowed delays elapsed since onset $o$. Its log likelihood ratio is $\log[(1-0.90F)/(1-0.0001F)]$. A received positive alert has likelihood ratio $0.90/0.0001$ because its delay distribution is identical in the two hypotheses. With jitter, three of the 315 onset-delay pairs arrive after readout and remain censored.

The syndrome trigger examines four rounds of detection counts for every onset that could produce an alert during an extension. Each tested delay model uses a threshold chosen on 250,000 separate burst-free calibration trials. The smallest integer threshold satisfying a 0.9% full-trial wait budget was 46 for fixed delay and 48 for jitter. Each threshold is shared by the global and local conditions. Positive flags, syndromes, and logical labels are shared between fixed and variable delay for each burst scope. The jitter draws are independent across timestamps. Burst-free trials are shared across scopes. Both scope-specific decoders can process those same samples.

## Policies and estimates

| Policy | When the deadline is extended | Evidence admitted during the extension |
|---|---|---|
| Fixed | Never | None |
| Always wait for alerts | At each eligible nominal deadline | Late alerts, with the original syndrome window |
| Selective alerts | When the frozen trigger fires | Late alerts, with the original syndrome window |
| Selective lookahead | When the frozen trigger fires | Extra syndrome data, with alert evidence held at the nominal deadline |
| Selective lookahead and alerts | When the frozen trigger fires | Extra syndrome data and late alerts |

Potential extensions are three rounds long and finish before the next nominal deadline. Nine nominal deadlines are eligible, from round 21 through round 77 in steps of seven. The schedule also carries a round-14 slot with no possible arrivals and a trigger score of zero. The underlying code continues to round 84 under all policies. The time metric is the sum of the added decision rounds over the trial. The primary comparison is selective lookahead and alerts versus selective lookahead. Positive values indicate that the added alert reduces logical failure at the same waits and with the same syndrome data. The package also includes the alert-only comparison, always-wait reference and the earlier 80% recovery-margin calculation.

There are 40,000 test trials at each of 63 onset times for each scope, with 10,000 trials at each local patch position per onset. The 1.8 million burst-free trials give 6.84 million distinct syndrome trials across both scopes. The two delay conditions are paired re-evaluations of those syndromes. For burst-free trials, the simulator combines the no-false-alert case with an independently sampled nonempty false-alert pattern, weighted by its exact probability.

The intervals are pointwise normal approximations to stratified paired test-shot variation, conditional on calibration. Test-shot variation is combined over independently seeded onsets. Local shots are balanced across patch positions within each onset. Population estimates weight the burst-free stratum by 0.999 and each onset by $0.001/63$. The same 0.1% prior is used for both spatial models. By-onset and arrival-band estimates are available in the CSV. The arrival bands classify onsets by their **fixed** 14-round arrival, so they can be compared on the same onset sets when jitter is introduced.

## Reproduce

The packaged run uses Python 3.12.14 on Linux. Set up the environment with Python 3.12, 3.13, or 3.14. If your Mac exposes Python 3.14 under `python3.14`, use that command to create the virtual environment.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest -v
.venv/bin/python verify_robustness.py results

OPENBLAS_NUM_THREADS=1 .venv/bin/python run_robustness.py --workers 8 --output rerun
OPENBLAS_NUM_THREADS=1 .venv/bin/python diagnose_trigger.py rerun --workers 2
.venv/bin/python verify_robustness.py rerun
.venv/bin/python plot_robustness.py rerun rerun/robustness_comparison.png
```

Use fewer workers if memory is limited. A partial run can be resumed using the same command and `--resume`. Completed chunks are written atomically. Stim's seeded samples can vary by platform, version, and batch size. The trigger diagnostic used root seed 2026111314 and the primary experiment used 2026111313.

| File | Contents |
|---|---|
| `results/summary.csv` | Conditional and population failure rates, paired contrasts, wait rates, and timing sensitivity |
| `results/paired_moments.csv` | Sample counts and paired moments by onset and test chunk |
| `results/run.json` | Noise and delay parameters, calibration histograms, thresholds, and seed plan |
| `results/trigger_capture.csv`, `results/trigger_capture_by_onset.csv` | Independent syndrome-only diagnostic of waits at the commit where the true alert would arrive |
| `results/robustness_comparison.png` | Main matched-wait estimates for all four scenarios |
