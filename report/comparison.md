# Staged matched comparison

72 runs completed: three methods, four geometry settings, three problem instances and two search seeds. Each run had a cap of 500 evaluations including identical initial designs. Settings were fixed before the study. The full 2,100-run study remains pending.

## Final regret

The table reports 10 raised to mean log10(max(regret, 1e-12)): a geometric mean with a numerical floor. Each cell has six runs from three instances.

| κ | Orientation | ARD BO | Isotropic BO | CMA-ES |
|---|---|---|---|---|
| 1 | aligned | 0.00030602 | 0.0004546 | 6.0172e-08 |
| 1 | rotated | 0.00018168 | 0.00026208 | 1.5309e-07 |
| 10000 | aligned | 2.0383e-06 | 2.2841e-05 | 3.8709e-06 |
| 10000 | rotated | 2.2905e-05 | 5.2886e-05 | 4.1058e-06 |

## Paired effects at 500 calls

Positive method effects favour CMA-ES; positive rotation effects indicate a rotation penalty. Interaction is the rotation penalty at κ=10000 minus its value at κ=1. Intervals are pointwise 95% percentile bootstrap intervals over whole instances (10,000 draws, seed 20261103). Only three instances are available: these are exploratory intervals, not confirmatory population evidence. Individual instance means are in the CSV and plots.

| Contrast | κ | Orientation | Method | Mean | 95% interval | Instances / pairs |
|---|---|---|---|---|---|---|
| interaction | 10000 | paired | ard | 1.277 | [1.180, 1.337] | 3 / 6 |
| interaction | 10000 | paired | cmaes | -0.380 | [-1.241, 0.360] | 3 / 6 |
| interaction | 10000 | paired | isotropic | 0.604 | [0.037, 0.897] | 3 / 6 |
| method | 1 | aligned | ard | 3.706 | [3.357, 4.113] | 3 / 6 |
| method | 1 | aligned | isotropic | 3.878 | [3.271, 4.775] | 3 / 6 |
| method | 1 | rotated | ard | 3.074 | [2.388, 3.554] | 3 / 6 |
| method | 1 | rotated | isotropic | 3.233 | [2.521, 3.716] | 3 / 6 |
| method | 10000 | aligned | ard | -0.279 | [-0.579, -0.013] | 3 / 6 |
| method | 10000 | aligned | isotropic | 0.771 | [0.539, 0.932] | 3 / 6 |
| method | 10000 | rotated | ard | 0.747 | [-0.004, 1.283] | 3 / 6 |
| method | 10000 | rotated | isotropic | 1.110 | [0.712, 1.618] | 3 / 6 |
| rotation | 1 | paired | ard | -0.226 | [-0.441, -0.025] | 3 / 6 |
| rotation | 1 | paired | cmaes | 0.406 | [0.342, 0.528] | 3 / 6 |
| rotation | 1 | paired | isotropic | -0.239 | [-0.725, 0.720] | 3 / 6 |
| rotation | 10000 | paired | ard | 1.051 | [0.896, 1.289] | 3 / 6 |
| rotation | 10000 | paired | cmaes | 0.026 | [-0.894, 0.887] | 3 / 6 |
| rotation | 10000 | paired | isotropic | 0.365 | [0.152, 0.757] | 3 / 6 |

## Interpretation

At 500 calls, CMA-ES had much lower geometric-mean regret on both sphere cases.
ARD BO was best on the aligned ill-conditioned problem (about 1.9 times lower
regret than CMA-ES), but its regret increased about 11.2 times after rotation.
CMA-ES had lower average regret on that rotated problem, although the paired
ARD-minus-CMA interval includes zero. The convergence curves show why budget
matters: BO often improves rapidly early, while CMA-ES continues improving later.

The increase in ARD's rotation penalty from κ=1 to κ=10000 was 1.277 log10 units,
with an exploratory interval [1.180, 1.337]. This is consistent with the proposed
mechanism: separate coordinate length scales can represent aligned anisotropy,
but cannot directly represent an arbitrarily rotated metric. CMA-ES's corresponding
interaction interval includes zero. This supports investigating the explanation
further; it does not establish a universal difference between the algorithms.

The isotropic control also had a positive interaction (0.604), despite its GP's
rotation-invariant distance. Therefore the GP representation alone does not explain
all observed effects: finite acquisition optimisation, sampling and the small set
of paired instances also matter. Nonzero sphere-control contrasts should not be
mistaken for proof that theoretical distributional invariance fails.

Normalisation fixes the expected loss of a uniform point, but not difficulty at
every target accuracy. Increasing κ redistributes relative curvature; when aligned,
ARD can exploit the dominant directions. Thus greater conditioning need not worsen
every method's finite-budget regret on this normalised family.

Frequent zero-EI fallbacks and GP fitting warnings are a substantial limitation of
this BO implementation. The observed plateaus coincide with fallback behaviour,
but these diagnostics alone do not distinguish posterior overconfidence from
numerical acquisition issues. The acquisition diagnostic below investigates this limitation before any
generalisation about BO stagnation. Settings were not changed
after examining these results.

In the paired-effect figures, blue dots show the aggregate mean, blue lines the
95% bootstrap interval, and orange dots the individual instance means. With only
three instances, these intervals offer limited evidence beyond the observed cases.

## Runtime

| Method | Median seconds | IQR seconds |
|---|---|---|
| ard | 43.43 | [32.94, 63.06] |
| isotropic | 73.36 | [15.03, 138.6] |
| cmaes | 0.01825 | [0.01808, 0.01841] |

Serial execution used one BLAS thread. Timing includes initial evaluations, fitting/sampling, acquisition or covariance updates, constraint handling and bookkeeping; it excludes imports, problem/design generation, serialization and analysis. The tiny CMA-ES timings are machine-specific.

## Diagnostics

| BO kernel | Zero-EI fallbacks / proposals | Fits with warnings |
|---|---|---|
| ard | 5109 / 11736 | 11642 |
| isotropic | 4896 / 11736 | 11155 |

The fallback counts describe this implementation; they do not establish that all BO methods behave similarly. Raw GP warnings and SLSQP diagnostics remain in each run file. All objectives here are noiseless and single-basin. Dimension and nonquadratic transfer tests remain pending.

Figures: [convergence](../results/comparison/staged/convergence_d5_beta0.png), [method effects](../results/comparison/staged/method_effects.png), [rotation effects](../results/comparison/staged/rotation_effects.png), [interaction](../results/comparison/staged/interaction_effects.png), [runtime](../results/comparison/staged/runtime.png). PDF versions accompany each PNG.

## BO acquisition diagnostic follow-up

We ran 32 controlled experiments: four acquisition variants, both GP kernels, κ=1/10000 and both orientations, with d=5, β=0, instance/run 0. Each used 100 evaluations including the same 11 initial points. GP fitting and all model hyperparameter rules stayed fixed. The variants are baseline EI/global candidates, log-EI/global candidates, EI/mixed candidates, and log-EI/mixed candidates. Mixed pools contain 384 global candidates plus 64 feasible novel perturbations at each radius 0.01 and 0.1 around the observed incumbent. All use ten SLSQP starts. These variants remain opt-in.

All 32 runs completed. All eight baseline trajectories exactly reproduced the first 100 saved evaluations. Original comparison artifacts and the baseline source snapshot retained their checksums. The diagnostic uses fresh, compatible output files; it does not replace the 72-run study.

| Kernel | κ | Orientation | Baseline | Log-EI | Local | Both |
|---|---|---|---|---|---|---|
| ard | 1 | aligned | 0.000381 | 3.137e-05 | 1.981e-06 | 3.055e-06 |
| ard | 1 | rotated | 0.000155 | 0.000227 | 1.655e-06 | 2.15e-06 |
| ard | 10000 | aligned | 4.277e-06 | 1.036e-07 | 8.122e-07 | 1.331e-07 |
| ard | 10000 | rotated | 0.0002542 | 0.0001602 | 0.0004971 | 4.265e-05 |
| isotropic | 1 | aligned | 0.0008963 | 5.295e-06 | 3.691e-06 | 9.659e-07 |
| isotropic | 1 | rotated | 0.000102 | 4.119e-05 | 4.49e-06 | 3.641e-06 |
| isotropic | 10000 | aligned | 0.0005785 | 0.0002582 | 0.000103 | 0.0003472 |
| isotropic | 10000 | rotated | 0.0006201 | 0.0004284 | 0.000352 | 0.0002955 |

### What the frozen-model audit establishes

Saved GP states were reconstructed using the original warm-start sequence and RNG before proposals 50, 100 and the first fallback where present. Each variant was given the same frozen model and a clone of the candidate RNG. Extra local samples replace the last 128 global candidates. Offline true-loss checks are diagnostic only and never enter GP training.

At evaluation 90 in the aligned ARD sphere run, all 512 global candidates had positive predictive standard deviation, yet ordinary EI was zero everywhere. Maximum log-EI was −799.08: finite mathematically, but below the representable EI range in double precision. Mixed candidates instead included 128 points with positive ordinary EI.

| Variant at that frozen state | Selected ordinary EI | Offline candidate loss |
|---|---|---|
| baseline | 0 | 1.2703 |
| logei | 7.0757e-276 | 0.78793 |
| local | 0.00036428 | 1.9438e-06 |
| logei_local | 0.00036496 | 7.8691e-07 |

The incumbent loss was 0.000381038. For this sphere, F(x)=||x−x*||²/C with C=0.776786; its strictly improving region has radius 0.017204 and lies inside the unit ball. A fresh uniform pool of 512 points hits that region with probability 1−(1−r⁵)^512≈7.72e-07. This explains why global sampling offers poor local resolution. Local optimisation could still reach the region from outside, but zero/tiny EI and numerical local-search failures obstruct that route. The mixed-pool method uses only the observed incumbent, not x*.

### Interpretation and limits

Across the eight 100-call runs per variant, the baseline used 44 random fallbacks;
each new variant used zero. Failed SLSQP exits were 1,138 out of 6,680 attempted
refinements for the baseline, versus 238, 75 and 57 out of 7,120 for log-EI,
local and combined variants. A failed refinement does not invalidate a run:
the best feasible candidate pool is retained. These counts are diagnostics,
not a substitute for objective regret. Total search times summed over the eight
runs were 125.6, 303.7, 213.3 and 291.5 seconds respectively; acquisition quality
improvements can cost more computation. Frozen-model replay time is excluded.

Both numerical underflow and candidate coverage contribute. A finite log-EI value corrects the arithmetic representation; it does not guarantee that SLSQP finds a good acquisition maximum. In the frozen example, mixed candidates expose useful EI under the unchanged GP, demonstrating that the original surrogate was not wholly uninformative.

Effects on live optimisation are not uniformly favourable. Log-EI alone worsened the rotated ARD sphere case, and local candidates alone worsened the rotated ill-conditioned ARD case. These counterexamples are retained. One instance and one seed per setting support a mechanism investigation, not confidence intervals or a universal winner. Baseline/model settings were not retuned after observing outcomes.

The earlier BO-versus-CMA results remain a record of the original implementation. These acquisition effects mean that its plateaus should not be generalised to BO as a method. A revised, frozen implementation needed a new matched comparison before replacing those conclusions; the held-out study below supplies that comparison at 100 calls. No variant is promoted automatically.

Log-EI follows [Ament et al. (2023)](https://arxiv.org/abs/2310.20708). Tests compare the stable expression with an independent numerical integral, including extreme tails, zero uncertainty and branch boundaries.

Artifacts: [convergence](../results/bo_diagnostics/convergence.png), [fallback counts](../results/bo_diagnostics/fallbacks.png), [live runs](../results/bo_diagnostics/runs.csv), [frozen-model audit](../results/bo_diagnostics/frozen.csv). PDF versions accompany the figures.

## Revised BO: held-out comparison at 100 calls

The combined `logei_local` variant was frozen for both BO kernels after the acquisition diagnostic; CMA-ES settings stayed unchanged. We completed 72 fresh runs on instances 3–5, search seeds 0–1, d=5, β=0, κ=1/10000 and both orientations. Each used exactly 100 objective calls including the 11 shared initial observations. There were no failures, missing results or early stops. These instances were unused for variant selection, but belong to the same problem family. No tuning used their outcomes.

This dataset is separate from the original 500-call comparison. Comparing cohorts directly would confound the acquisition change with the changed instances. The earlier controlled diagnostic isolates acquisition changes; this held-out study tests the selected implementation against unchanged CMA-ES on new instances.

### Regret and paired effects

Geometric-mean regret across six runs per cell (three instances, two search seeds):

| κ | Orientation | Revised ARD BO | Revised isotropic BO | CMA-ES |
|---|---|---|---|---|
| 1 | aligned | 2.741e-06 | 2.503e-06 | 0.01334 |
| 1 | rotated | 1.654e-06 | 2.311e-06 | 0.01134 |
| 10000 | aligned | 8.658e-07 | 0.0006063 | 0.002452 |
| 10000 | rotated | 0.0003822 | 0.0005057 | 0.004387 |

Both revised BO variants have lower geometric-mean regret than CMA-ES at 100 calls in all four conditions. Paired differences use log10(max(regret, 1e-12)); negative BO-minus-CMA values favour BO. All intervals below are exploratory 95% instance-bootstrap intervals: 10,000 resamples, seed 20261103. Search seeds remain together within each instance. There are only three independent instances, not six, and intervals are not adjusted for multiple contrasts.

| κ | Orientation | ARD minus CMA [interval] | Isotropic minus CMA [interval] |
|---|---|---|---|
| 1 | aligned | −3.687 [−4.060, −3.279] | −3.727 [−3.950, −3.425] |
| 1 | rotated | −3.836 [−4.205, −3.280] | −3.691 [−3.830, −3.423] |
| 10000 | aligned | −3.452 [−3.632, −3.115] | −0.607 [−0.742, −0.379] |
| 10000 | rotated | −1.060 [−1.557, −0.508] | −0.938 [−1.188, −0.482] |

The geometry effect persists after the acquisition revision. Positive rotation effects mean higher regret after rotation; the interaction is the rotation effect at κ=10000 minus that at κ=1.

| Method | Rotation effect, κ=1 | Rotation effect, κ=10000 | Interaction |
|---|---|---|---|
| Revised ARD BO | −0.219 [−0.709, 0.052] | 2.645 [2.496, 2.861] | 2.864 [2.526, 3.205] |
| Revised isotropic BO | −0.035 [−0.177, 0.083] | −0.079 [−0.132, −0.045] | −0.044 [−0.141, 0.132] |
| CMA-ES | −0.070 [−0.391, 0.217] | 0.253 [−0.029, 0.443] | 0.323 [0.226, 0.381] |

ARD's ill-conditioned rotation penalty corresponds to approximately 441 times higher geometric-mean regret. The much larger ARD interaction and near-zero isotropic interaction support the coordinate-metric explanation on these cases. However, CMA-ES also has a positive interaction whose interval excludes zero. This counterexample prevents a claim that only ARD responds to rotation. With three instances and subsequent random proposals not coupled by the hidden rotation, a nonzero contrast does not establish failure of theoretical distributional invariance. All three sphere-control rotation intervals include zero.

### Runtime and numerical diagnostics

| Method | Median search seconds | IQR seconds |
|---|---|---|
| Revised ARD BO | 35.87 | [27.04, 42.07] |
| Revised isotropic BO | 36.37 | [29.53, 42.77] |
| CMA-ES | 0.004356 | [0.004251, 0.004477] |

These serial measurements include objective evaluation and optimiser overhead, but exclude imports, problem/design generation, serialization and analysis. The BO evaluation-efficiency advantage costs substantially more computation on this cheap objective. This is not an equal-wall-time comparison, and short CMA-ES times are machine-specific.

| BO kernel | Random fallbacks / proposals | Failed SLSQP exits / refinements | Fits with warnings / fits |
|---|---|---|---|
| ARD | 0 / 2136 | 187 / 21360 | 2006 / 2136 |
| Isotropic | 0 / 2136 | 153 / 21360 | 1579 / 2136 |

Feasible pool candidates remain available after failed refinements. GP warnings remain common even though random fallbacks disappeared, so these results do not remove all numerical or model-fitting limitations. CMA-ES rejected 86 outside-ball proposals across its 24 runs; their time was included without objective charges. Every run stopped at its budget.

The figures label the revised methods explicitly in convergence/runtime plots. In paired-effect plots, `ard` and `isotropic` refer to the combined `logei_local` variant. Orange dots show individual instance means, blue dots the aggregate means and blue lines the exploratory intervals.

Artifacts: [convergence](../results/heldout_comparison/heldout/convergence_d5_beta0.png), [method effects](../results/heldout_comparison/heldout/method_effects.png), [rotation effects](../results/heldout_comparison/heldout/rotation_effects.png), [interaction](../results/heldout_comparison/heldout/interaction_effects.png), [runtime](../results/heldout_comparison/heldout/runtime.png), [checkpoint CSV](../results/heldout_comparison/heldout/checkpoints.csv), [summary](../results/heldout_comparison/heldout/summary.json). PDF versions accompany the figures.

## Separate 500-call timing pilot

All three fresh runs completed 500 calls on d=5, β=0, κ=10000, rotated, instance 3, search seed 0. Their first 100 input/output pairs exactly match the corresponding held-out runs. There are no bootstrap intervals: this is one run per method, separate from the repeated 100-call comparison.

| Method | Regret at 100 | Regret at 500 | Elapsed seconds at 100 | Elapsed seconds at 500 |
|---|---|---|---|---|
| Revised ARD BO | 0.0005763 | 1.107e-07 | 27.44 | 319.43 |
| Revised isotropic BO | 0.0008949 | 5.204e-07 | 24.21 | 280.87 |
| CMA-ES | 0.003778 | 6.301e-05 | 0.004934 | 0.01944 |

The time columns are cumulative measurements within each 500-call run, rather than timings taken from the separate 100-call executions. Five times the objective budget required about 11.6 times the elapsed time for each BO kernel and 3.94 times for CMA-ES in this case. ARD spent 56.88 seconds fitting GPs and 262.54 seconds on acquisition search; isotropic BO spent 22.83 and 258.02 seconds respectively. Objective evaluations cost less than 0.01 seconds per run. These measurements illustrate overhead growth, but cannot reliably project costs across dimensions, conditioning levels or all 2,100 planned runs.

Both revised BO variants retain lower regret than CMA-ES at 500 calls in this case. This is not a replicated 500-call ranking, and must not be pooled with the original comparison or the held-out 100-call dataset. The curves still plateau late despite no random fallbacks, so eliminating fallbacks does not guarantee continued progress.

Each BO pilot made 489 acquisition proposals and 4,890 SLSQP refinements. ARD recorded 57 failed refinement exits and warnings in all 489 GP fits; isotropic BO recorded 94 failed exits and warnings in 450 fits. Neither used a random fallback. CMA-ES rejected 12 outside-ball proposals. All three reached the budget without solver failure.

The full 2,100-run matrix, including dimension and nonquadratic checks, remains pending. The current evidence supports an implementation-specific account: improving acquisition arithmetic and candidate coverage helps BO use a small evaluation budget, while coordinate-sensitive modelling still matters on rotated anisotropic landscapes and computation costs remain substantial.

Artifacts: [pilot convergence](../results/heldout_comparison/timing/convergence_d5_beta0.png), [pilot runtime](../results/heldout_comparison/timing/runtime.png), [pilot checkpoints](../results/heldout_comparison/timing/checkpoints.csv), [pilot summary](../results/heldout_comparison/timing/summary.json). Raw records and separate manifests are in `results/heldout_comparison/`; `source_snapshot.txt` records the executed code and `preserved_checksums.json` protects the 136 original comparison/diagnostic artifacts.
