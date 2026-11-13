# Bayesian optimisation pilot

Historical implementation pilot, preceding the completed studies in
[comparison.md](comparison.md). Its 50-call results are excluded from the
report’s main comparisons.

Eight runs; dimension 5, beta 0, instance 0, run 0, 50 calls including 11 initial points. Executed 6 November 2026.

| Kernel | Conditioning | Orientation | Final regret | Seconds | Failed SLSQP refinements | Fits with warnings |
|---|---:|---|---:|---:|---:|---:|
| ard | 10000 | aligned | 8.7048761e-06 | 8.24 | 1 / 390 | 39 / 39 |
| ard | 10000 | rotated | 0.0023243636 | 7.22 | 1 / 390 | 39 / 39 |
| ard | 1 | aligned | 0.00038103787 | 3.01 | 156 / 390 | 35 / 39 |
| ard | 1 | rotated | 0.00015498969 | 4.73 | 128 / 390 | 31 / 39 |
| isotropic | 10000 | aligned | 0.0024442804 | 6.40 | 0 / 390 | 15 / 39 |
| isotropic | 10000 | rotated | 0.0078071897 | 7.00 | 0 / 390 | 14 / 39 |
| isotropic | 1 | aligned | 0.00089633993 | 3.68 | 146 / 390 | 20 / 39 |
| isotropic | 1 | rotated | 0.00010198692 | 3.50 | 139 / 390 | 19 / 39 |

Every run completed 50 calls. No zero-EI random fallbacks were used. GP warnings include length scales reaching the upper bound; warning text is stored in each JSON. Failed refinements are counted even if their final point is feasible and useful. The original random pool always remains eligible.

The ARD result on the aligned ill-conditioned problem was better than its rotated counterpart in this pilot. One instance and one seed cannot establish a systematic rotation penalty or superiority over another method. CMA-ES was evaluated subsequently; see [cmaes_pilot.md](cmaes_pilot.md) and
the repeated studies in [comparison.md](comparison.md).

The isotropic surrogate is rotationally invariant, but acquisition sampling and numerical local searches are not coupled by rotation; individual paired search traces can differ. Surrogate invariance was checked separately using rotated training/query points.

Timing starts immediately before evaluating the initial design and includes objective calls, GP fitting, acquisition selection and history collection. It excludes problem generation, initial-point generation, JSON serialization and plots. Pilots were run sequentially with OPENBLAS_NUM_THREADS=1. Timings depend on the machine and concurrent workload.

Versions: python 3.14.7, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2.

To reproduce a row, use `.venv/bin/python code/prepare_benchmark.py --bo --budget 50 --conditioning K --orientation O --kernel M --output OUTPUT`, replacing `K`, `O`, `M` with its settings and `OUTPUT` with a fresh directory. Dimension, beta, instance and run use their defaults.

Full observations, timings, checkpoints, seeds and warnings are in [results/bo](../results/bo/).
