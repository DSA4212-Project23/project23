# Controlled valley geometry: rotation × conditioning

## Research question

Does coupling coordinates change search difficulty, and does this effect increase
with conditioning? The hypothesis is that representations favouring individual
coordinate directions incur a larger rotation penalty as conditioning grows.
This is a testable hypothesis, not an established comparison result. No optimisers
have yet been run. Conclusions will depend on the specific implementations tested.

The construction follows the emphasis on interpretable difficulty factors in
[COCO's benchmark design](https://coco-platform.org/testsuites/bbob/overview.html).
It is a custom benchmark, not an implementation of the official BBOB functions.
[Hansen's tutorial](https://arxiv.org/abs/1604.00772) motivates checking rotation
invariance with correspondingly transformed initial information.

## Objective and exact solution

For dimension d ≥ 2, minimise over the unit Euclidean ball:

\[
 f(x)=C^{-1}\sum_{i=1}^d\lambda_i(z_i^2+\beta z_i^4),\qquad
 z=Q^T x-a,\qquad \|x\|_2\leq1,
 \qquad\lambda_i=\kappa^{(i-1)/(d-1)}.
\]

Here Q is identity or a seeded random rotation, κ ≥ 1, β ≥ 0, and a is a seeded
uniform direction with length 0.25. The unique optimum is x* = Qa, with f* = 0:
each positive-weight term is nonnegative and vanishes exactly when z = 0.
The optimum is strictly interior, 0.75 units from the boundary.

At the optimum, H = (2/C) Q diag(λ) Qᵀ, so its condition number is exactly κ.
For β = 0 this Hessian is constant. For β = 1, curvature changes with position;
κ specifies **local conditioning at the optimum**, not global conditioning.
Both versions are strictly convex and have one basin. In two dimensions,
quadratic contour-axis length ratio is √κ, not κ.

Normalise by the uniform-ball expected unnormalised loss:

\[
 C=\sum_i\lambda_i\left[\frac1{d+2}+a_i^2+
 \beta\left(\frac3{(d+2)(d+4)}+\frac{6a_i^2}{d+2}+a_i^4\right)\right].
\]

This follows from E[Xᵢ²] = 1/(d+2), E[Xᵢ⁴] = 3/((d+2)(d+4)), zero odd moments,
and rotational symmetry. Thus E[f(X)] = 1 for a uniformly sampled feasible X.
Normalisation preserves the optimum and Hessian condition number. It controls
mean random-point scale, not the entire loss distribution or absolute curvature.

## Controls and experiment matrix

| Study | d | β | κ | Orientations |
|---|---|---|---|---|
| Main | 5 | 0 | 1, 100, 10000 | Aligned, rotated |
| Dimension check | 10 | 0 | 100, 10000 | Aligned, rotated |
| Nonquadratic check | 5 | 1 | 100, 10000 | Aligned, rotated |

Each of 14 conditions has 10 instances. Ten instances paired across conditions
prevent one fortunate orientation or optimum location from deciding the result.
Two-dimensional illustrations use 45° rotations; experiments use independently
seeded rotations, not a fixed angle. The κ = 1, β = 0 case is the spherical
negative control; κ = 1, β = 1 is not rotationally symmetric about its optimum.

The unit ball is invariant under rotation. Exactly f_Q(Qx) = f_I(x), so both the
objective and feasible geometry rotate together. Initial points also rotate:
X_Q = X_I Qᵀ for row-vector arrays. Values and pairwise distances match before
search. Initial means derived from these points must rotate too; an origin mean
and isotropic initial scale are also consistent. Matching integer random seeds
alone does not guarantee identical rotated solver trajectories.

Do not replace the ball with a fixed cube, clip coordinates, or add penalties.
That would change the experiment. Future solvers need the same known ball
constraint. Reject/resample infeasible proposals or constrain candidate selection
directly; log proposal failures and their runtime. The boundary can still affect
performance, but it introduces no preferred orientation geometrically.

## Interface and reproducibility

All code is in `code/prepare_benchmark.py`, with functions and dictionaries.
The seven original mathematical check sections and BO checks share `run_checks(specs)`.
`make_problem(dimension=5, conditioning=100, orientation="aligned", beta=0,
instance=0)` returns a plain dictionary containing settings and mathematical data.
Create a separate run dictionary as `state = {"budget": 500, "evaluations": 0}`.
`evaluate(problem, state, x)` validates the budget, performs a budgeted scalar
evaluation and rejects nonfinite, incorrectly shaped, or infeasible input.
Tolerance 1e-12 covers floating-point error at the boundary without clipping.
Every valid evaluation, including a repeated point or initial point, costs one
call. Invalid inputs consume no objective calls; their handling costs runtime.
Exhaustion raises an error before an additional evaluation.

The problem dictionary, exact optimum `problem["rotation"] @ problem["shift"]`,
`loss(problem, x)` and `hessian(problem, x)` are evaluator-only data and analysis functions. Analysis
loss bypasses the budget. A future solver receives an evaluation callback that
calls `evaluate`, plus dimension, feasibility and its initial observations;
it must not receive the problem dictionary or call analysis functions directly.
This separation is an experimental convention, not a security boundary.
The manifest is also evaluator configuration, not solver input.

Master seed: 20261102. Separate NumPy SeedSequence streams generate shifts,
rotations, and initial designs. Instance IDs 0–9; run IDs 0–4. Within a dimension,
the same shift and rotation are reused across κ and β. Rotations use Gaussian QR,
positive diagonal sign correction, and a determinant +1 correction.
Different dimensions are separate experiments, not embedded copies.

`initial_design(problem, run)` supplies 2d+1 points sampled uniformly by volume,
with radius U^(1/d) and a uniform sphere direction. Search randomness must be
independent, e.g. SeedSequence([20261102, d, instance, run, 3]). Both methods
receive the same evaluated initial information within each instance/run. How
each solver incorporates it must be documented in the subsequent method stage.

## Later evaluation contract

- Five independent search runs per instance: 700 runs per method across the suite.
- 500 objective calls, including initial information; checkpoints 50/100/250/500.
- Primary score: regret f(recommendation) − f* on the known noiseless objective.
  A checkpoint recommendation is fixed before evaluator scoring; evaluator-only
  scoring is excluded from the search budget and never fed back to the solver.
- Report regret curves and paired rotation effects. For statistical summaries,
  use log10(max(regret, 1e-12)); clearly label the numerical plotting floor.
  Compute rotated-minus-aligned differences within matched instance/run pairs.
- Report the mean paired log-regret difference and a 95% percentile bootstrap
  interval: 10000 resamples of instance IDs, retaining all five runs and both
  orientations per sampled instance. Seed 20261103. Show per-instance variation;
  do not treat 50 runs as 50 independent problem instances.
- A larger rotation effect at κ = 10000 than at κ = 1 supports the hypothesis;
  no growth or the opposite effect challenges it. Check whether the direction
  persists at d = 10 and β = 1 before making broader claims.
- Record objective time and end-to-end search time, including model fitting,
  candidate selection and constraint handling. Exclude diagnostic plots and
  evaluator scoring. Evaluation efficiency and runtime need separate conclusions.

## Validation and limits

Automated checks cover exact optimum, feasibility, rotation equivalence, local
conditioning, numerical Hessians, spherical control, uniform sampling moments,
unit expected loss, paired starts, reproducibility and budget enforcement.
The preparation script generates a numeric validation report across 180 settings.
Its analysis evaluations do not constitute optimisation trials.

The nonquadratic extension checks transfer within a related family, not broad
real-world generalisation. There is no noise, multimodality, unknown constraint,
or low-dimensional hidden subspace. A failure to distinguish methods is a valid
result; do not select new instances after seeing outcomes to manufacture a winner.
CMA-ES settings are now fixed below; equal tuning allowances must remain fixed before the full comparison.
BO is implemented; eight pilot runs are documented in [bo_pilot.md](bo_pilot.md).

## Implemented BO settings

`run_bo(score, X_initial, budget=500, seed=0, kernel="ard")` evaluates its own
initial points, counts every objective call, and returns observations, recommendations,
diagnostics and timing records. `score` calls the budgeted evaluator. The solver
receives neither the rotation nor the optimum. Invalid configuration fails before
evaluation; runtime failures return partial results with status `failed`.

The primary model is constant-amplitude Matérn-5/2 with one learned length scale
per coordinate (ARD). The isotropic control shares one learned scale. Initial
scale 0.5, bounds [0.01,10], amplitude initially 1 with bounds [0.001,1000],
normalised responses, fixed alpha=1e-8 for regularisation. The first fit uses two
additional hyperparameter restarts; later fits warm-start the previous kernel
and run one optimisation without additional restarts. Warnings are saved.

Each acquisition step considers 512 uniform-ball points and refines the ten
highest-EI novel starts using SLSQP: 100 iterations, ftol=1e-9, smooth squared-norm
constraint and analytic constraint gradient. EI is divided by its positive pool
maximum to stabilise the stopping scale. The original pool and all finite feasible
refinements compete, including feasible points returned with unsuccessful status.
Every selected point must be more than 1e-8 from existing observations. If all pool
EI is zero, use its first novel point and log a random fallback. If a pool has no
novel point, try ten more pools, then fail explicitly. No clipping or penalties.

The seed derives from [master, dimension, instance, run, 3]; separate child streams
control acquisition sampling and GP fitting. Kernel variants and orientations
share these seed identifiers. Acquisition samples are not rotated using hidden
geometry, so individual isotropic runs need not be rotated copies. A separate
test rotates GP training and query points and verifies matching predictions.

Results give the best observed point at every checkpoint, with checkpoint losses
at 50/100/250/500 if reached and at the final budget. Objective, fitting,
acquisition and total search times are recorded separately. File serialization
and plots are outside timing. Library versions and fixed settings accompany each
result. The design manifest describes landscape preparation; BO outcomes live in
their own results directory.

## Implemented CMA-ES settings

`run_cmaes(score, X_initial, budget=500, seed=0)` uses pycma 4.5.0's ask/tell
interface. It evaluates the same 2d+1 initial points as BO and uses their best point
as the initial mean; BO instead uses all observations in its GP. Initial covariance
is identity (the library's tiny diagonal perturbation is compensated using
CMA_teststds), initial sigma is 0.3/sqrt(d), and population is 4+floor(3 log(d)).
Active full covariance updates are enabled; mirrors and restarts are disabled.
Other algorithm and termination settings retain the pinned library defaults.
These are fixed choices, not settings selected by pilot performance.

A private NumPy generator supplies normal samples. Driver seeds derive from
[20261102, d, instance, run, 4]; no global NumPy seed is changed. Matching seeds
across orientations does not guarantee identical rotated sample trajectories.
Only feasible proposals are scored, using the evaluator's 1e-12 tolerance.
Resampling consumes time, not objective calls, and stops with a recorded failure
after 10,000 rejections within a generation. No clipping or penalty is applied.
An incomplete final population uses only remaining calls and skips the update.
The best evaluated point is recommended without an extra evaluation of the mean.

Results use BO's observation/history/checkpoint fields plus termination reason,
completed generations, incomplete population size, rejection count and settings.
Objective and total search times are recorded; optimizer time is their difference
and includes bookkeeping and constraint handling. Library logs are disabled.
An early normal stop records actual calls and only reached checkpoints. For later
fixed-budget plots, carry its final best value forward, clearly marking the stop;
do not fabricate evaluation records. Failures retain partial results separately.
Four pilots are documented in [cmaes_pilot.md](cmaes_pilot.md).

## Staged comparison protocol

Before the full 2,100-run comparison, run 72 experiments: d=5, beta=0,
kappa=1/10000, both orientations, three instances (0–2), two runs (0–1),
and three methods (ARD BO, isotropic BO, CMA-ES). The 500-call cap and all
previously specified solver settings remain fixed. No tuning uses these results.
One BLAS thread and serial execution separate algorithm overhead from concurrency.
Job order is shuffled using seed 20261104. Resume checks code and dependency
fingerprints; existing pilot results are not silently incorporated.

Within each checkpoint, pair methods on instance/run/geometry and orientations
on instance/run/method. In addition to rotation effects, compute BO-minus-CMA
log regret and the increase in rotation effect from kappa=1 to kappa=10000.
Positive method effects favour CMA-ES. For each contrast, average the paired
seed differences within each instance, then average instances equally. Bootstrap
these instance means 10,000 times with seed 20261103. Preserve all available
paired seeds in each resampled instance. Report actual instance/pair counts;
exclude failures from contrasts and disclose that resulting comparisons are
conditional on successful pairs. Three-instance intervals are exploratory.

Known-objective validation is evaluator-only and cannot influence search.
Convergence curves use the mean of instance-average log regret at each call,
with the 1e-12 numerical floor. Runtime distributions report median and IQR
alongside all individual runs. Diagnostic figures and data processing are outside
search timing. The full study and dimension/nonquadratic transfer checks remain
pending after this stage.

## Held-out revised-method validation

After the acquisition diagnostic, freeze logei_local for ARD and isotropic BO.
Validate against the unchanged CMA-ES implementation using instances 3–5, runs
0–1, d=5, beta=0, kappa=1/10000 and both orientations. This is a separate 72-run
study with a 100-call cap including 11 initial evaluations, fixed before outcomes
are observed. Use the existing paired contrasts and whole-instance bootstrap;
only three independent instances remain, so inference is exploratory. Do not pool
these runs with the earlier baseline implementation or interpret between-cohort
changes as a paired correction effect.

The predefined timing pilot consists of the three methods at instance 3, run 0,
kappa=10000, rotated, with 500-call caps. Check identical prefixes against the
100-call runs. Report its measured time growth and scores separately; one setting
cannot establish runtime scaling across the full suite. All jobs run serially with
thread limits set before imports. Failures remain visible and do not trigger tuning.
The full 2,100-run sweep remains pending after these studies.
