# Bayesian optimisation versus CMA-ES

Controlled experiments on rotation, conditioning and acquisition search in
five-dimensional noiseless quadratic problems. The completed studies comprise
72 original 500-call runs, 32 acquisition ablations, 72 held-out 100-call runs
and three separate 500-call timing pilots. Larger studies are optional extensions.

- [Report PDF](report/main.pdf), [LaTeX source](report/main.tex), [Overleaf ZIP](report/overleaf.zip).
- [Authors](authors.md)
- [Detailed study record](report/comparison.md) and [problem specification](report/problem_design.md).

The report's complete source, inline bibliography and required figures are in
`report/`. All executable project code and checks use functions and dictionaries
in `code/prepare_benchmark.py`; there are no custom classes.

## 1. Environment and checks

Run commands from the repository root. The recorded environment used Python
**3.14.7**, macOS ARM64 and NumPy's Apple Accelerate BLAS. `requirements.txt` pins
NumPy 2.5.4, Matplotlib 3.11.2, SciPy 1.18.1, scikit-learn 1.9.1, pycma 4.5.0 and
threadpoolctl 3.8.0. The manifests retain the exact environment for each study.

```sh
python3.14 -m venv .venv
.venv/bin/python --version
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python code/prepare_benchmark.py --check
```

The checks cover mathematical invariants, sampling, EI/log-EI, budgets, constraints,
GP rotation invariance, CMA-ES stopping, reproducibility, comparison pairing,
bootstrap grouping, failures and compatible-run reuse. Checks use temporary
outputs rather than modifying the archived experiment datasets.

Version pins do not guarantee identical floating-point trajectories on another
OS/BLAS. Exact diagnostic replay is strict and may fail in a different numerical
environment; use the recorded environment for exact reproduction. Report any
mismatch instead of editing fingerprints, weakening checks or replacing archives.
Wall-clock timings are machine-specific and cannot be reproduced exactly.

## 2. Synthetic data preparation

No external dataset or download is required. The generator creates shifted,
rotated quadratic/quartic functions on the unit ball from master seed 20261102.
Reported optimisation experiments use dimension 5 and quadratic shape; the
quartic and dimension extensions have mathematical validation only.

Use a fresh directory so original figures and manifests remain intact:

```sh
prep_dir="$(mktemp -d "${TMPDIR:-/tmp}/project23-prep.XXXXXX")"
.venv/bin/python code/prepare_benchmark.py --output "$prep_dir"
printf '%s\n' "$prep_dir"
```

This validates landscapes and writes the problem manifest plus three PNG/PDF
figures. It performs no optimisation runs. `results/problem_design/` contains the
archived preparation outputs.

## 3. Verify saved results without running optimisers

**Analyse copies.** Analysis rewrites plots, including PDF metadata. Rewriting
archived figures would invalidate the checksums used by the acquisition diagnostic.
These commands copy the data and regenerate the original, held-out and timing
analyses without new objective-search evaluations:

```sh
audit_dir="$(mktemp -d "${TMPDIR:-/tmp}/project23-audit.XXXXXX")"
cp -R results/comparison "$audit_dir/comparison"
cp -R results/heldout_comparison "$audit_dir/heldout_comparison"
.venv/bin/python code/prepare_benchmark.py --analyse --study staged --output "$audit_dir/comparison"
.venv/bin/python code/prepare_benchmark.py --analyse --study heldout --output "$audit_dir/heldout_comparison"
.venv/bin/python code/prepare_benchmark.py --analyse --study timing --output "$audit_dir/heldout_comparison"
printf '%s\n' "$audit_dir"
```

Expected summaries: **72/72**, **72/72**, **3/3** completed, with no missing or
failed runs. Validation checks saved objective values, feasibility, initial
observations, budget accounting, best recommendations and histories. Timing
analysis also checks exact first-100 input/output agreement with held-out runs.
Each analysis writes `summary.json`, `checkpoints.csv` and figures; paired-effect
CSVs/plots apply to the repeated studies. Held-out checkpoints have 144 rows and
timing checkpoints 12. The pilot has no bootstrap contrasts.

The acquisition diagnostic's archived `summary.json`, `runs.csv`, `frozen.csv`
and raw JSON files are in `results/bo_diagnostics/`. It contains 32 completed runs
and 21 frozen GP states (84 proposals); it has no separate analysis-only CLI.
Verify archive integrity without rewriting any data:

```sh
.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
checks = json.loads(Path('results/heldout_comparison/preserved_checksums.json').read_text())
for name, expected in checks.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
summary = json.loads(Path('results/bo_diagnostics/summary.json').read_text())
assert summary['completed'] == summary['expected'] == 32
assert summary['frozen_states'] == 21
print('Verified 136 protected artifacts; diagnostic summary: 32 runs, 21 states.')
PY
```

## 4. Reproduce experiments from scratch

These commands **run optimisers** and can take substantial time. They are not
needed merely to inspect or validate saved results. Use one new directory per
reproduction attempt, and keep the held-out and timing studies together:

```sh
repro_dir="$(mktemp -d "${TMPDIR:-/tmp}/project23-repro.XXXXXX")"
printf '%s\n' "$repro_dir"
# Original EI/global-pool BO against unchanged CMA-ES: 72 runs, 500 calls each.
.venv/bin/python code/prepare_benchmark.py --compare --study staged --output "$repro_dir/comparison"
# Four acquisition variants: 32 runs plus frozen-model replay.
.venv/bin/python code/prepare_benchmark.py --bo-diagnostic --output "$repro_dir/bo_diagnostics"
# Frozen logei_local BO against unchanged CMA-ES: 72 runs, 100 calls each.
.venv/bin/python code/prepare_benchmark.py --compare --study heldout --output "$repro_dir/heldout_comparison"
# One case per method: 3 runs, 500 calls; requires the held-out results above.
.venv/bin/python code/prepare_benchmark.py --compare --study timing --output "$repro_dir/heldout_comparison"
```

The diagnostic **always reads the preserved original baseline** from
`results/comparison/` and its checksum manifest from `results/bo_diagnostics/`,
regardless of `--output`. Its output option only changes where new diagnostic
records are written. It does not consume the newly reproduced staged directory.
Keep the original baseline records and protected plots intact. Frozen replay and
the eight baseline ablations must reproduce those original trajectories.

Do not rerun bare `--compare` or `--bo-diagnostic` into the archival directories:
they contain historical source fingerprints that differ from the current source.
Fresh runs record the current fingerprint honestly; archived fingerprints are
never rewritten. Compatible runs in a reproduction directory are reused when
the same command is repeated with the same environment. An interrupted individual
run restarts; completed runs remain saved. `--retry-failed` archives failed records
before retrying. Changed code/dependencies/settings require another fresh directory.

The initial 11 evaluations count towards every budget. Fixed study matrices
ignore single-problem flags; conflicting explicit budgets are rejected. Thread
limits are set before numerical imports (including Apple Accelerate), and jobs
are executed serially in a seeded shuffled order. All raw observations, warnings,
settings, timing and recommendations are saved atomically.

For a small implementation smoke run, use another fresh directory:

```sh
smoke_dir="$(mktemp -d "${TMPDIR:-/tmp}/project23-smoke.XXXXXX")"
.venv/bin/python code/prepare_benchmark.py --bo --budget 15 --output "$smoke_dir/bo"
.venv/bin/python code/prepare_benchmark.py --cmaes --budget 15 --output "$smoke_dir/cmaes"
```

Single BO runs default to original `baseline` ARD BO. The revised variant is
selected by the held-out/timing study runners; it is also available through
`run_bo(..., variant="logei_local")`. The optional full matrix contains 2,100
runs; it was not executed and is not required to reproduce this report.

## 5. Map the report to the saved evidence

| Report content | Saved evidence |
| --- | --- |
| Original 500-call comparison | `results/comparison/staged/`, with raw records in `results/comparison/runs/` |
| Acquisition ablation and frozen-model explanation | `results/bo_diagnostics/`, especially `runs.csv` and `frozen.csv` |
| Held-out 100-call regret and rotation effects | `results/heldout_comparison/heldout/` |
| Separate 500-call timing pilot | `results/heldout_comparison/timing/` |
| Known optimum, normalisation and numerical validation | `results/problem_design/` and `--check` |

The recommendation is the best observed feasible point. Regret equals its loss
because the exact minimum is zero. Curves use log10 regret floored at 1e-12;
reported geometric means exponentiate mean log-regret. Paired effects average
search seeds within an instance, then resample whole instances 10,000 times
(seed 20261103). Three instances are not six independent observations because
there are two search seeds. Positive BO-minus-CMA effects favour CMA-ES; rotation
effects are rotated minus aligned, and the interaction is the high-conditioning
rotation effect minus the sphere-control effect.

Timing includes objective calls, fitting/acquisition or CMA updates, constraints
and bookkeeping, excluding imports, problem/design generation, file writing and
analysis. Invalid CMA proposals cost time but no objective evaluations. Normal
early-stop recommendations are carried forward only to that study's budget;
failures are reported separately. No results are pooled across study cohorts.

## 6. Build and submit the report

Complete `authors.md` and the matching group list in `report/main.tex` first.
Remove unused member rows/items (maximum four). Upload `report/overleaf.zip` as
a new Overleaf project, select `main.tex` and use **XeLaTeX**. The bibliography
is inline; there is no BibTeX step. Preserve the relative `figures/` folder.
Alternatively, with Tectonic installed:

```sh
cd report
tectonic main.tex
cd ..
```

Rebuild the ZIP after source or author edits:

```sh
.venv/bin/python - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
report = Path('report')
with ZipFile(report / 'overleaf.zip', 'w', ZIP_DEFLATED) as archive:
    archive.write(report / 'main.tex', 'main.tex')
    for name in ('heldout_convergence.pdf', 'heldout_interaction.pdf'):
        archive.write(report / 'figures' / name, 'figures/' + name)
PY
```

Check the compiled PDF is **at most 10 pages including references and appendix**
after every edit. No code belongs in the report's main text. The repository URL
at the start must match the published repository. Publish the final source,
figures, PDF, authors and numerical results before submitting; do not include
`.venv/` or temporary reproduction directories. This repository was verified as
**private**; the owner must confirm that GitHub user `alexxthiery` has marker
access. No invitation or publishing action is implied by these local files.

## 7. Reading and understanding checklist

Relevant sources already cited in the report include Frazier's BO tutorial,
Hansen's CMA-ES tutorial, the BBOB function definitions, pycma documentation and
Ament et al.'s log-EI paper. The last extends beyond the starter references.
The bibliography contains the links. Merely citing these sources does not
establish that a group member has read or understood them.

Before submission, each member should read the relevant sections, inspect or
rerun experiments, and explain in their own words:

- Why use a ball, rotate the initial design, and normalise expected random-point loss?
- Why does an ARD metric favour aligned anisotropy, and what does the isotropic control test?
- What separates EI underflow from poor candidate coverage in the frozen-model experiment?
- Why do three instances limit the bootstrap evidence, and why is the timing pilot separate?
- Which comparisons have matched budgets, what tuning imbalance remains, and what could falsify the proposed explanation?

Record actual contributions and observations truthfully if required by the course.
This checklist makes no claim that any member has completed these activities.
