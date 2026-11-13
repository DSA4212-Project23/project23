# CMA-ES pilot

Historical implementation pilot, preceding the completed repeated studies in
[comparison.md](comparison.md). These pilot runs are excluded from the report’s
main comparisons.

Four runs: d=5, beta=0, instance=0, run=0; each used exactly 500 calls,
including 11 shared initial points. Settings were fixed before running.
Each completed 61 populations of eight and one final evaluated point without
an update. All runs completed successfully; none stopped early.

| κ | Orientation | CMA at 50 | CMA at 500 | ARD BO at 50 | Isotropic BO at 50 | CMA total seconds | Rejected proposals |
|---|---|---|---|---|---|---|---|---|
| 1 | aligned | 0.0365865 | 1.68296e-08 | 0.000381038 | 0.00089634 | 0.0174 | 0 |
| 1 | rotated | 0.0388153 | 2.65688e-08 | 0.00015499 | 0.000101987 | 0.0172 | 0 |
| 10000 | aligned | 0.00669307 | 6.54627e-06 | 8.70488e-06 | 0.00244428 | 0.0173 | 6 |
| 10000 | rotated | 0.0081294 | 2.6538e-07 | 0.00232436 | 0.00780719 | 0.0173 | 14 |

CMA-ES had higher regret than both BO variants at the matched 50-call checkpoint
in these four cases. CMA-ES improved substantially by 500 calls, but those final
values cannot be compared fairly with BO's 50-call values. The subsequent 500-call comparison used matched budgets and repeated
problem instances/search seeds; see [comparison.md](comparison.md).

The rotated ill-conditioned CMA run finished better than the aligned run; this
single pair does not establish that rotations help or refute distributional
rotation invariance. Equal integer seeds do not make trajectories rotated copies.

Timing covers initial evaluations, optimizer setup, sampling, rejection, covariance
updates and bookkeeping. It excludes imports, problem/design construction and
serialization. Runs used OPENBLAS_NUM_THREADS=1; these very short timings are
machine-specific and not stable runtime estimates. Version information and all
checkpoints/observations are stored in each JSON under results/cmaes/.

Constraint rejection is rotation-symmetric but changes the unconstrained sampling
distribution. Starting from the best initial point uses less of the initial data
than BO's surrogate; this is an explicit policy, not hidden objective information.
The completed repeated comparisons and their limitations are documented in
[comparison.md](comparison.md).
