"""Generate benchmark landscapes, run checks, or run BO or CMA-ES."""
import os
import sys

THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
# Accelerate caches this setting: set it before importing NumPy/SciPy.
if __name__ == "__main__" and ("--compare" in sys.argv or "--check" in sys.argv or "--bo-diagnostic" in sys.argv):
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"

import argparse
import csv
import hashlib
import threadpoolctl
import json
import platform
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterMathtext
import numpy as np
import cma
import scipy
import sklearn
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import erfcx
import copy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

# Problem creation and evaluation
MASTER_SEED = 20261102
FEASIBILITY_TOL = 1e-12


def _check_int(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def sample_ball(dimension, count, rng):
    """Uniform volume sampling, not uniform radius or sphere-surface sampling."""
    _check_int(dimension, "dimension", 2)
    _check_int(count, "count")
    directions = rng.normal(size=(count, dimension))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    while np.any(norms == 0):
        zero = norms[:, 0] == 0
        directions[zero] = rng.normal(size=(np.count_nonzero(zero), dimension))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
    # U^(1/d) spreads points uniformly through the ball's volume.
    return directions / norms * rng.random((count, 1)) ** (1 / dimension)


def make_problem(dimension=5, conditioning=100.0, orientation="aligned", beta=0.0, instance=0):
    """Build evaluator-only problem data in a plain dictionary."""
    _check_int(dimension, "dimension", 2)
    _check_int(instance, "instance")
    if not np.isfinite(conditioning) or conditioning < 1:
        raise ValueError("conditioning must be finite and >= 1")
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and >= 0")
    if orientation not in ("aligned", "rotated"):
        raise ValueError("orientation must be aligned or rotated")
    d = dimension
    # Separate seeds keep the shift and rotation paired across problem settings.
    shift_rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, d, instance, 0]))
    rotation_rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, d, instance, 1]))
    a = shift_rng.normal(size=d)
    a *= 0.25 / np.linalg.norm(a)
    # QR gives orthogonal directions; sign corrections give a random rotation.
    q, r = np.linalg.qr(rotation_rng.normal(size=(d, d)))
    q *= np.where(np.diag(r) >= 0, 1.0, -1.0)
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    if orientation == "aligned":
        q = np.eye(d)
    weights = conditioning ** (np.arange(d) / (d - 1))
    normalizer = float(np.sum(weights * (
        1 / (d + 2) + a**2 + beta * (
            3 / ((d + 2) * (d + 4)) + 6 * a**2 / (d + 2) + a**4
        )
    )))
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("settings exceed the supported floating-point range")
    for array in (a, q, weights):
        array.setflags(write=False)
    return {
        "dimension": d,
        "conditioning": conditioning,
        "orientation": orientation,
        "beta": beta,
        "instance": instance,
        "rotation": q,
        "shift": a,
        "weights": weights,
        "normalizer": normalizer,
    }


def loss(problem, x):
    """Scalar or batch loss for analysis; does not consume a search budget."""
    x = np.asarray(x, dtype=float)
    if x.ndim < 1 or x.shape[-1] != problem["dimension"] or not np.all(np.isfinite(x)):
        raise ValueError(f"expected finite points with last dimension {problem['dimension']}")
    z = x @ problem["rotation"] - problem["shift"]
    return np.sum(problem["weights"] * (z**2 + problem["beta"] * z**4), axis=-1) / problem["normalizer"]


def hessian(problem, x):
    x = np.asarray(x, dtype=float)
    if x.ndim < 1 or x.shape[-1] != problem["dimension"] or not np.all(np.isfinite(x)):
        raise ValueError(f"expected finite points with last dimension {problem['dimension']}")
    if x.ndim != 1:
        raise ValueError("hessian expects one point")
    z = x @ problem["rotation"] - problem["shift"]
    curvature = problem["weights"] * (2 + 12 * problem["beta"] * z**2) / problem["normalizer"]
    return (problem["rotation"] * curvature) @ problem["rotation"].T


def is_feasible(problem, x):
    try:
        x = np.asarray(x, dtype=float)
        if x.ndim < 1 or x.shape[-1] != problem["dimension"] or not np.all(np.isfinite(x)):
            return False
    except (ValueError, TypeError):
        return False
    return bool(x.ndim == 1 and np.linalg.norm(x) <= 1 + FEASIBILITY_TOL)


def evaluate(problem, state, x):
    """Budgeted scalar evaluation. Invalid inputs do not consume calls."""
    _check_int(state["budget"], "budget", 1)
    if not is_feasible(problem, x):
        raise ValueError("expected one finite point in the unit Euclidean ball")
    if state["evaluations"] >= state["budget"]:
        raise RuntimeError("objective evaluation budget exhausted")
    value = float(loss(problem, np.asarray(x, dtype=float)))
    if not np.isfinite(value):
        raise FloatingPointError("objective evaluation is not finite")
    state["evaluations"] += 1
    return value


def initial_design(problem, run, count=None):
    """Matched uniform-ball starts; evaluating these consumes the search budget."""
    _check_int(run, "run")
    d = problem["dimension"]
    rng = np.random.default_rng(np.random.SeedSequence([MASTER_SEED, d, problem["instance"], run, 2]))
    points = sample_ball(d, 2 * d + 1 if count is None else count, rng)
    return points @ problem["rotation"].T


# Figures and landscape validation


def landscapes(output, beta):
    coordinates = np.linspace(-1, 1, 501)
    xx, yy = np.meshgrid(coordinates, coordinates)
    points = np.stack((xx, yy), axis=-1)
    outside = np.sum(points**2, axis=-1) > 1
    theta = np.pi / 4
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # Illustrations use exactly 45 degrees; experiment instances use random rotations.
    problems, values = [], []
    for row in range(2):
        for kappa in (1, 100, 10000):
            problem = make_problem(2, kappa, "aligned", beta)
            if row:
                problem = dict(problem, rotation=rotation)
            problems.append(problem)
            values.append(np.ma.array(loss(problem, points), mask=outside))
    vmax = 10 ** np.ceil(np.log10(max(float(v.max()) for v in values)))
    levels = np.geomspace(1e-5, vmax, 29)
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.3), layout="constrained")
    for index, (ax, problem, value) in enumerate(zip(axes.flat, problems, values)):
        artist = ax.contourf(xx, yy, value, levels=levels, norm=LogNorm(1e-5, vmax), cmap="viridis", extend="min")
        ax.contour(xx, yy, value, levels=[.001, .01, .1, 1], colors="white", linewidths=.45, alpha=.65)
        ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="#334155", linewidth=1))
        x_star = problem["rotation"] @ problem["shift"]
        ax.plot(*x_star, marker="*", color="#ff795e", markersize=10, markeredgecolor="white", markeredgewidth=.6)
        ax.set(aspect="equal", xlim=(-1.02, 1.02), ylim=(-1.02, 1.02), xlabel="$x_1$", ylabel="$x_2$")
        orientation = "Aligned" if index < 3 else "Rotated 45°"
        ax.set_title(f"{orientation} · κ = {problem['conditioning']:g}", fontsize=11)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        for spine in ax.spines.values():
            spine.set_visible(False)
    shape = "Quadratic" if beta == 0 else "Quadratic + quartic"
    fig.suptitle(f"{shape} valleys: orientation and conditioning\nStar = exact optimum; shared logarithmic colour scale", fontsize=15)
    fig.colorbar(artist, ax=axes, shrink=.85, label="Normalised loss (log scale)",
                 ticks=10.0 ** np.arange(-5, int(np.log10(vmax)) + 1),
                 format=LogFormatterMathtext())
    name = "quadratic_landscapes" if beta == 0 else "nonquadratic_landscapes"
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def curvature_figure(output):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    for kappa, colour in zip((1, 100, 10000), ("#167c80", "#dc7c32", "#6b55a0")):
        problem = make_problem(5, kappa)
        x_star = problem["rotation"] @ problem["shift"]
        eigenvalues = np.linalg.eigvalsh(hessian(problem, x_star))
        axes[0].plot(range(1, 6), eigenvalues / eigenvalues[0], "o-", color=colour, label=f"κ = {kappa:g}")
    axes[0].set(yscale="log", xlabel="Principal direction", ylabel="Curvature / smallest curvature", title="What conditioning changes", xticks=range(1, 6))
    axes[0].legend(frameon=False)
    distance = np.linspace(-.65, .65, 301)
    for beta, style, colour in ((0, "-", "#167c80"), (1, "--", "#dc7c32")):
        problem = make_problem(5, 100, beta=beta)
        x_star = problem["rotation"] @ problem["shift"]
        points = np.tile(x_star, (len(distance), 1))
        points[:, 0] += distance
        # Remove the direction weight and normaliser to expose shape alone.
        curve = loss(problem, points) * problem['normalizer'] / problem['weights'][0]
        axes[1].plot(distance, curve, style, color=colour, label=f"β = {beta}")
    axes[1].set(xlabel="Displacement from optimum", ylabel="Rescaled loss: $C f / \\lambda_1$", title="Nonquadratic shape check")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"geometry_diagnostics.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def validate_landscapes():
    rng = np.random.default_rng(9182)
    report = {"settings_checked": 0, "max_rotation_error": 0.0,
              "max_condition_relative_error": 0.0, "max_optimum_loss": 0.0,
              "max_unit_mean_error": 0.0, "max_mean_error_in_standard_errors": 0.0}
    for d in (2, 5, 10):
        points = sample_ball(d, 20000, rng)
        for kappa in (1, 100, 10000):
            for beta in (0, 1):
                for instance in range(10):
                    spec = dict(dimension=d, conditioning=kappa, beta=beta, instance=instance)
                    aligned = make_problem(**spec)
                    rotated = make_problem(**spec, orientation="rotated")
                    error = float(np.max(np.abs(loss(aligned, points) - loss(rotated, points @ rotated['rotation'].T))))
                    x_star = rotated["rotation"] @ rotated["shift"]
                    eig = np.linalg.eigvalsh(hessian(rotated, x_star))
                    condition_error = abs(float(eig[-1]/eig[0]/kappa)-1)
                    optimum_loss = float(loss(rotated, x_star))
                    losses = loss(rotated, points)
                    mean_error = abs(float(np.mean(losses))-1)
                    standard_errors = mean_error / (float(np.std(losses, ddof=1))/np.sqrt(len(losses)))
                    if error > 1e-11 or condition_error > 1e-8 or optimum_loss > 1e-25 or standard_errors > 6:
                        raise AssertionError(f"Landscape validation failed: {spec}")
                    report["settings_checked"] += 1
                    for key, value in (("max_rotation_error", error), ("max_condition_relative_error", condition_error),
                                       ("max_optimum_loss", optimum_loss), ("max_unit_mean_error", mean_error),
                                       ("max_mean_error_in_standard_errors", standard_errors)):
                        report[key] = max(report[key], value)
    report["status"] = "passed"
    report["monte_carlo_points_per_dimension"] = 20000
    return report


# Seven checks, runnable with --check


def run_checks(specs):
    """Run seven checks in order, without generating files."""
    if not __debug__:
        raise RuntimeError("Run checks without Python's -O flag so assertions remain enabled")

    # 1. Rotation, optimum and conditioning
    rng = np.random.default_rng(90)
    for d in (2, 5, 10):
        points = sample_ball(d, 100, rng)
        for kappa in (1, 100, 10000):
            for beta in (0, 1):
                for instance in range(10):
                    a = make_problem(d, kappa, "aligned", beta, instance)
                    b = make_problem(d, kappa, "rotated", beta, instance)
                    q = b["rotation"]
                    x_star = q @ b["shift"]
                    np.testing.assert_allclose(q.T @ q, np.eye(d), atol=1e-14)
                    assert abs(np.linalg.det(q) - 1) < 1e-7
                    assert abs(np.linalg.norm(x_star) - .25) < 1e-7
                    assert is_feasible(b, x_star)
                    assert loss(b, x_star) < 1e-25
                    assert np.all(loss(b, points) >= 0)
                    np.testing.assert_allclose(loss(a, points), loss(b, points @ q.T), atol=1e-13)
                    eig = np.linalg.eigvalsh(hessian(b, x_star))
                    assert abs(eig[-1] / eig[0] / kappa - 1) < 5e-10
    print("PASS: check_mathematical_invariants")

    # 2. Hessian versus finite differences
    for beta in (0, 1):
        p = make_problem(5, 100, "rotated", beta)
        x = np.array([.1, -.2, .05, .1, -.05])
        steps = np.eye(5) * 1e-4
        numeric = np.zeros((5, 5))
        for i, u in enumerate(steps):
            for j, v in enumerate(steps):
                numeric[i, j] = (loss(p, x + u + v) - loss(p, x + u - v)
                                 - loss(p, x - u + v) + loss(p, x - u - v)) / 4e-8
        np.testing.assert_allclose(hessian(p, x), numeric, rtol=1e-5, atol=1e-6)
    print("PASS: check_hessian")

    # 3. Uniform sampling and normalisation
    rng = np.random.default_rng(42)
    for d in (2, 5, 10):
        x = sample_ball(d, 100000, rng)
        assert np.all(np.linalg.norm(x, axis=1) <= 1)
        np.testing.assert_allclose(np.mean(x*x, axis=0), 1/(d+2), atol=.003)
        for beta in (0, 1):
            for kappa in (1, 100, 10000):
                values = loss(make_problem(d, kappa, "rotated", beta), x)
                assert abs(np.mean(values)-1) < 6*np.std(values)/np.sqrt(len(x))
    print("PASS: check_sampling_and_normalisation")

    # 4. Spherical control
    p = make_problem(5, 1, "rotated")
    x = sample_ball(5, 100, np.random.default_rng(12))
    x_star = p["rotation"] @ p["shift"]
    np.testing.assert_allclose(loss(p, x), np.sum((x-x_star)**2, axis=1)/p["normalizer"])
    print("PASS: check_sphere_control")

    # 5. Budget and invalid inputs
    p = make_problem()
    state = {"budget": 2, "evaluations": 0}
    for invalid in ([0, 0], [0]*6, [[0]*5], [np.nan]*5, [np.inf]*5, [1]*5, "bad"):
        assert not is_feasible(p, invalid)
        try:
            evaluate(p, state, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")
    assert state["evaluations"] == 0
    first = evaluate(p, state, [0]*5)
    assert isinstance(first, float)
    assert first == evaluate(p, state, [0]*5)
    try:
        evaluate(p, state, [0]*5)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError")
    assert state["evaluations"] == 2
    for budget in (0, -1, 1.5, True):
        state = {"budget": budget, "evaluations": 0}
        try:
            evaluate(p, state, [0]*5)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")
        assert state["evaluations"] == 0
    print("PASS: check_budget_and_validation")

    # 6. Reproducibility and paired starts
    a = make_problem()
    b = make_problem(orientation="rotated")
    again = make_problem(orientation="rotated")
    np.testing.assert_array_equal(b["rotation"], again["rotation"])
    np.testing.assert_allclose(initial_design(b, 0), initial_design(a, 0) @ b["rotation"].T)
    np.testing.assert_allclose(loss(a, initial_design(a, 0)), loss(b, initial_design(b, 0)))
    assert not np.array_equal(initial_design(b, 0), initial_design(b, 1))
    assert not np.array_equal(make_problem(instance=1)["shift"], a["shift"])
    print("PASS: check_reproducibility")

    # 7. Experiment settings
    assert len(specs) == 140
    assert len({tuple(spec.items()) for spec in specs}) == 140
    for args in ({"dimension": 1}, {"dimension": 2.5}, {"instance": -1},
                 {"conditioning": 0}, {"conditioning": np.inf}, {"beta": -1},
                 {"beta": np.nan}, {"orientation": "invalid"}):
        try:
            make_problem(**args)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected ValueError")
    print("PASS: check_settings")
    print("All seven checks passed.")

    # BO checks use small real GPs, then controlled failures to exercise fallbacks.
    from types import SimpleNamespace
    from unittest.mock import patch
    np.testing.assert_allclose(expected_improvement(np.array([-1., 0., 1.]),
                                                   np.zeros(3), 0), [1, 0, 0])
    np.testing.assert_allclose(expected_improvement(np.zeros(1), np.ones(1), 0),
                               [1 / np.sqrt(2 * np.pi)])
    aligned = make_problem(2, 100)
    rotated = make_problem(2, 100, "rotated")
    initial = initial_design(aligned, 0)
    values = loss(aligned, initial)
    rotated_initial = initial_design(rotated, 0)
    gp, _ = fit_gp(initial, values, "isotropic")
    gp_rotated, _ = fit_gp(rotated_initial, values, "isotropic")
    queries = sample_ball(2, 20, np.random.default_rng(14))
    mean, std = gp.predict(queries, return_std=True)
    rotated_mean, rotated_std = gp_rotated.predict(queries @ rotated["rotation"].T, return_std=True)
    np.testing.assert_allclose(mean, rotated_mean, atol=1e-6)
    np.testing.assert_allclose(std, rotated_std, atol=1e-6)

    outcomes = []
    for repeat in range(2):
        state = {"budget": 6, "evaluations": 0}
        result = run_bo(lambda x: evaluate(aligned, state, x), initial, 6, 19)
        assert result["status"] == "completed", result["error"]
        assert state["evaluations"] == result["evaluations"] == 6
        assert all(is_feasible(aligned, x) for x in result["X"])
        assert result["checkpoints"][-1]["evaluations"] == 6
        assert result["best_loss"] == min(result["y"])
        outcomes.append(result)
    np.testing.assert_array_equal(outcomes[0]["X"], outcomes[1]["X"])
    np.testing.assert_array_equal(outcomes[0]["y"], outcomes[1]["y"])
    state = {"budget": 4, "evaluations": 0}
    try:
        run_bo(lambda x: evaluate(aligned, state, x), initial, 4)
    except ValueError:
        pass
    else:
        raise AssertionError("Small budgets must be rejected before evaluation")
    assert state["evaluations"] == 0

    zero_ei_gp = SimpleNamespace(predict=lambda x, return_std: (np.ones(len(x)), np.zeros(len(x))))
    x, info = find_next_point(zero_ei_gp, initial, 0, np.random.default_rng(7))
    assert info["random_fallback"] and is_feasible(aligned, x)
    assert np.min(np.linalg.norm(initial-x, axis=1)) > 1e-8
    with patch.dict(find_next_point.__globals__, {"sample_ball": lambda d, n, rng: np.tile(initial[0], (n, 1))}):
        try:
            find_next_point(gp, initial, float(values.min()), np.random.default_rng(7))
        except RuntimeError:
            pass
        else:
            raise AssertionError("Exhausted novel candidates must fail explicitly")
    for rejected in (initial[0], np.ones(2)):
        with patch.dict(find_next_point.__globals__, {
                "minimize": lambda *a, **kw: SimpleNamespace(x=rejected, success=False)}):
            x, info = find_next_point(gp, initial, float(values.min()), np.random.default_rng(8))
        assert is_feasible(aligned, x)
        assert np.min(np.linalg.norm(initial-x, axis=1)) > 1e-8
        assert info["slsqp_failures"] == 10
        assert info["selected_ei"] >= info["best_pool_ei"] - 1e-14
    with patch.dict(run_bo.__globals__, {"fit_gp": SimpleNamespace()}):
        # A non-callable fitting dependency must be recorded as a failed run.
        result = run_bo(lambda x: float(loss(aligned, x)), initial, 6)
    assert result["status"] == "failed" and result["evaluations"] == 5
    assert "TypeError" in result["error"]
    print("PASS: Bayesian optimisation checks")

    # CMA-ES: count initial calls, complete and partial generations, and failures.
    global_rng_before = np.random.get_state()
    outcomes = []
    for repeat in range(2):
        state = {"budget": 20, "evaluations": 0}
        result = run_cmaes(lambda x: evaluate(aligned, state, x), initial, 20, 19)
        assert result["status"] == "completed", result["error"]
        assert state["evaluations"] == result["evaluations"] == 20
        np.testing.assert_array_equal(result["X"][:5], initial)
        assert all(is_feasible(aligned, x) for x in result["X"])
        assert result["generations"] == 2 and result["incomplete_population_evaluations"] == 3
        assert result["best_loss"] == min(result["y"])
        assert len(result["history"]) == 20
        assert result["checkpoints"][-1]["evaluations"] == 20
        outcomes.append(result)
    np.testing.assert_array_equal(outcomes[0]["X"], outcomes[1]["X"])
    np.testing.assert_array_equal(outcomes[0]["y"], outcomes[1]["y"])
    global_rng_after = np.random.get_state()
    assert global_rng_before[0] == global_rng_after[0]
    np.testing.assert_array_equal(global_rng_before[1], global_rng_after[1])
    assert global_rng_before[2:] == global_rng_after[2:]
    state = {"budget": 4, "evaluations": 0}
    for budget in (4, 0, True):
        try:
            run_cmaes(lambda x: evaluate(aligned, state, x), initial, budget)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid budget must fail before scoring")
    assert state["evaluations"] == 0
    with patch.object(cma, "CMAEvolutionStrategy", side_effect=RuntimeError("injected failure")):
        result = run_cmaes(lambda x: float(loss(aligned, x)), initial, 5)
        assert result["status"] == "completed" and result["evaluations"] == 5
        result = run_cmaes(lambda x: float(loss(aligned, x)), initial, 6)
        assert result["status"] == "failed" and result["evaluations"] == 5
    with patch.object(cma.CMAEvolutionStrategy, "stop", return_value={"tolx": 1e-11}):
        result = run_cmaes(lambda x: float(loss(aligned, x)), initial, 20)
        assert result["status"] == "completed" and result["evaluations"] == 5
        assert "tolx" in result["termination_reason"]
    with patch.object(cma.CMAEvolutionStrategy, "ask", return_value=[np.ones(2)] * 6):
        result = run_cmaes(lambda x: float(loss(aligned, x)), initial, 20)
        assert result["status"] == "failed" and result["evaluations"] == 5
        assert result["rejected_proposals"] == 10000
    for d in (5, 10):
        for beta in (0, 1):
            problem = make_problem(d, 10000, "rotated", beta)
            state = {"budget": 50, "evaluations": 0}
            result = run_cmaes(lambda x: evaluate(problem, state, x), initial_design(problem, 0), 50, 21)
            assert result["status"] == "completed", result["error"]
            assert state["evaluations"] == result["evaluations"] == 50
            assert all(is_feasible(problem, x) for x in result["X"])
    print("PASS: CMA-ES checks")

    # A cheap deterministic scorer tests the study plumbing, not optimiser quality.
    import tempfile
    import contextlib
    import io
    def fixed_run(score, points, budget, seed, kernel=None, variant="baseline"):
        xs = list(points) + [points[0]] * (budget-len(points))
        ys = [float(score(x)) for x in xs]
        best = np.minimum.accumulate(ys)
        history = [{"evaluations": i+1, "best_loss": float(v), "elapsed_seconds": .01}
                   for i,v in enumerate(best)]
        return dict(status="completed", error=None, evaluations=budget,
                    X=[x.tolist() for x in xs], y=ys,
                    recommendation=xs[int(np.argmin(ys))].tolist(), best_loss=min(ys),
                    history=history, checkpoints=[], settings={"variant":variant,"budget":budget}, timings={"total_seconds": .01, "objective_seconds": .005})
    with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
        folder = Path(directory)
        with patch.dict(run_comparison.__globals__, {"run_bo": fixed_run, "run_cmaes": fixed_run}):
            summary = run_comparison(folder)
        assert summary["completed"] == summary["expected"] == 72
        assert all(r["instances"] == 3 and r["pairs"] == 6 for r in summary["contrasts"])
        for r in summary["contrasts"]:
            if r["kind"] == "method":
                assert r["mean"] == r["lower"] == r["upper"] == 0
        with patch.dict(run_comparison.__globals__, {"run_bo": None, "run_cmaes": None}):
            assert run_comparison(folder)["completed"] == 72
        manifest = json.loads((folder / "manifest_staged.json").read_text())
        job = next(j for j in manifest["jobs"] if j["method"] == "cmaes")
        path = folder / "runs" / job["file"]
        saved = json.loads(path.read_text())
        early = dict(saved)
        early.update(X=saved["X"][:20], y=saved["y"][:20], history=saved["history"][:20],
                     evaluations=20, evaluator_calls=20, termination_reason={"tolx": "test"})
        early["best_loss"] = min(early["y"])
        early["recommendation"] = early["X"][int(np.argmin(early["y"]))]
        path.write_text(json.dumps(early))
        assert analyse_comparison(folder)["completed"] == 72
        early.update(status="failed", error="injected failure")
        path.write_text(json.dumps(early))
        failed = analyse_comparison(folder)
        assert failed["completed"] == 71 and len(failed["failures"]) == 1
        with patch.dict(run_comparison.__globals__, {"run_bo": fixed_run, "run_cmaes": fixed_run}):
            assert run_comparison(folder, retry_failed=True)["completed"] == 72
        corrupt = json.loads(path.read_text()); corrupt["fingerprint"]["budget"] = 10
        path.write_text(json.dumps(corrupt))
        try:
            run_comparison(folder)
        except ValueError:
            pass
        else:
            raise AssertionError("Incompatible saved data must be rejected")
        path.unlink()
        missing = analyse_comparison(folder)
        assert missing["completed"] == 71 and missing["missing"] == [job["file"]]
        path.write_text(json.dumps(saved))
        manifest["jobs"] = [job]
        (folder / "manifest_staged.json").write_text(json.dumps(manifest))
        assert analyse_comparison(folder)["completed"] == 1

    with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
        folder = Path(directory)
        with patch.dict(run_comparison.__globals__, {"run_bo": fixed_run, "run_cmaes": fixed_run}):
            heldout = run_comparison(folder, "heldout")
            timing = run_comparison(folder, "timing")
        assert heldout["completed"] == 72 and heldout["budget"] == 100
        assert {r["checkpoint"] for r in heldout["contrasts"]} == {50,100}
        assert timing["completed"] == 3 and timing["budget"] == 500 and not timing["contrasts"]
        manifest = json.loads((folder/"manifest_heldout.json").read_text())
        assert {j["problem"]["instance"] for j in manifest["jobs"]} == {3,4,5}
        for job in manifest["jobs"]:
            result=json.loads((folder/"runs"/job["file"]).read_text())
            assert result["evaluations"] == result["evaluator_calls"] == 100
            if job["method"] != "cmaes":
                assert result["settings"]["variant"] == "logei_local"
        with patch.dict(run_comparison.__globals__, {"run_bo": None, "run_cmaes": None}):
            assert run_comparison(folder,"heldout")["completed"] == 72
        job = next(j for j in manifest["jobs"] if j["method"] == "cmaes")
        path = folder/"runs"/job["file"]
        saved = json.loads(path.read_text())
        stopped = dict(saved, evaluations=20, evaluator_calls=20, termination_reason={"tolx":"test"},
                       X=saved["X"][:20], y=saved["y"][:20], history=saved["history"][:20])
        stopped["best_loss"]=min(stopped["y"])
        stopped["recommendation"]=stopped["X"][int(np.argmin(stopped["y"]))]
        path.write_text(json.dumps(stopped))
        assert analyse_comparison(folder,"heldout")["completed"]==72
        with (folder/"heldout"/"checkpoints.csv").open() as f:
            rows=list(csv.DictReader(f))
        assert {int(r["checkpoint"]) for r in rows}=={50,100}
        assert sum(r["carried_forward"]=="True" for r in rows)==2
    print("PASS: held-out budgets, variant dispatch, timing prefixes and resume")
    print("PASS: comparison pairing, bootstrap, resume, failures and early-stop checks")



    from scipy.integrate import quad
    for z in (-1., -10., -40., -1000., -1e6):
        a = -z
        integral = quad(lambda u: u*np.exp(-u-u*u/(2*a*a)),0,np.inf)[0]
        reference = norm.logpdf(z)-2*np.log(a)+np.log(integral)
        np.testing.assert_allclose(log_expected_improvement([-z],[1.],0)[0], reference,rtol=1e-12,atol=1e-10)
    z=np.array([-.5,0.,1.,10.])
    np.testing.assert_allclose(np.exp(log_expected_improvement(-z,np.ones(4),0)),expected_improvement(-z,np.ones(4),0),rtol=1e-12)
    np.testing.assert_array_equal(log_expected_improvement([-2.,0.,2.],[0.,0.,0.],0),[np.log(2),-np.inf,-np.inf])
    for boundary in (-1.,-1e6):
        values=log_expected_improvement(-np.array([boundary*(1-1e-9),boundary*(1+1e-9)]),[1.,1.],0)
        assert abs(values[1]-values[0]) < 1e-7*max(1,abs(values[0]))
    for std in (-1.,np.nan):
        try:
            log_expected_improvement([0.],[std],0)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid uncertainty must fail")
    for variant in ("baseline","logei","local","logei_local"):
        results=[]
        for repeat in range(2):
            state={"budget":8,"evaluations":0}
            result=run_bo(lambda x:evaluate(aligned,state,x),initial,8,19,"ard",variant)
            assert result["status"]=="completed",result["error"]
            assert result["evaluations"]==state["evaluations"]==8
            assert all(is_feasible(aligned,x) for x in result["X"])
            for info in result["acquisition_diagnostics"]:
                if variant in ("logei","logei_local") and not info["random_fallback"]:
                    assert info["selected_log_ei"]>=info["best_pool_log_ei"]-1e-12
                else:
                    assert info["selected_ei"]>=info["best_pool_ei"]-1e-14
            results.append(result)
        np.testing.assert_array_equal(results[0]["X"],results[1]["X"])
    for variant in ("logei", "logei_local"):
        point,info=find_next_point(zero_ei_gp,initial,0,np.random.default_rng(7),variant,initial[0])
        assert info["random_fallback"] and is_feasible(aligned,point)
        assert np.min(np.linalg.norm(initial-point,axis=1))>1e-8
    print("PASS: stable log-EI references, limits and four acquisition variants")


def fit_gp(X, y, kernel="ard", previous=None, seed=0):
    """Learn directional scales for ARD; share one scale for the isotropic control."""
    if previous is None:
        scale = np.full(X.shape[1], 0.5) if kernel == "ard" else 0.5
        covariance = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=scale, length_scale_bounds=(1e-2, 10.0), nu=2.5)
    else:
        covariance = previous.kernel_
    gp = GaussianProcessRegressor(
        kernel=covariance, alpha=1e-8, normalize_y=True,
        n_restarts_optimizer=2 if previous is None else 0, random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gp.fit(X, y)
    return gp, [str(w.message) for w in caught]


def expected_improvement(mean, std, y_best):
    mean, std = np.asarray(mean), np.asarray(std)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std < 0):
        raise ValueError("GP predictions must be finite, with nonnegative uncertainty")
    improvement = y_best - mean
    ei = np.maximum(improvement, 0.0)
    positive = std > 0
    z = improvement[positive] / std[positive]
    ei[positive] = improvement[positive] * norm.cdf(z) + std[positive] * norm.pdf(z)
    return np.maximum(ei, 0.0)


def log_expected_improvement(mean, std, y_best):
    """Stable analytic log-EI for minimisation; true zero improvement is -inf.

    Tail rearrangement follows Ament et al. (2023), arXiv:2310.20708.
    """
    mean, std = np.broadcast_arrays(np.asarray(mean, float), np.asarray(std, float))
    if not np.isfinite(y_best) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std < 0):
        raise ValueError("Expected finite predictions and nonnegative uncertainty")
    delta = y_best - mean
    result = np.full(mean.shape, -np.inf)
    deterministic = (std == 0) & (delta > 0)
    result[deterministic] = np.log(delta[deterministic])
    positive = std > 0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        z = delta[positive] / std[positive]
        h = np.empty_like(z)
        upper = z > -1
        h[upper] = np.log(norm.pdf(z[upper]) + z[upper] * norm.cdf(z[upper]))
        middle = (z <= -1) & (z > -1e6)
        u = z[middle]
        w = np.log(erfcx(-u / np.sqrt(2)) * np.abs(u)) + .5*np.log(np.pi/2)
        h[middle] = norm.logpdf(u) + np.log(-np.expm1(w))
        lower = z <= -1e6
        h[lower] = norm.logpdf(z[lower]) - 2*np.log(-z[lower])
        values = np.log(std[positive]) + h
        values[np.isposinf(z)] = np.log(delta[positive][np.isposinf(z)])
        result[positive] = values
    if np.any(np.isnan(result)) or np.any(np.isposinf(result)):
        raise FloatingPointError("log-EI exceeded its supported numerical range")
    return result


def find_next_point(gp, X, y_best, rng, variant="baseline", incumbent=None):
    """Compare a random pool and local EI refinements under the same filters."""
    if variant not in ("baseline", "logei", "local", "logei_local"):
        raise ValueError("Unknown BO variant")
    use_log = variant in ("logei", "logei_local")
    d = X.shape[1]
    diagnostics = {"slsqp_failures": 0, "random_fallback": False, "extra_pools": 0}
    for attempt in range(11):
        pool = sample_ball(d, 512, rng)
        distances = np.linalg.norm(pool[:, None, :] - X[None, :, :], axis=2)
        pool = pool[np.min(distances, axis=1) > 1e-8]
        if len(pool):
            break
    else:
        raise RuntimeError("No novel feasible candidate after eleven pools")
    diagnostics["extra_pools"] = attempt
    if variant in ("local", "logei_local"):
        if incumbent is None:
            raise ValueError("Local candidates require the observed incumbent")
        pieces = [pool[:384]]
        for radius in (.01, .1):
            accepted = []
            for _ in range(100):
                proposed = incumbent + radius * sample_ball(d, 64, rng)
                feasible = np.linalg.norm(proposed, axis=1) <= 1 + FEASIBILITY_TOL
                novel = np.min(np.linalg.norm(proposed[:,None,:]-X[None,:,:], axis=2), axis=1) > 1e-8
                accepted.extend(proposed[feasible & novel])
                if len(accepted) >= 64:
                    break
            if len(accepted) < 64:
                raise RuntimeError("Could not draw enough feasible novel local candidates")
            pieces.append(np.asarray(accepted[:64]))
        pool = np.vstack(pieces)
    mean, std = gp.predict(pool, return_std=True)
    pool_ei = expected_improvement(mean, std, y_best)
    log_ei = log_expected_improvement(mean, std, y_best)
    scores = log_ei if use_log else pool_ei
    finite_log = log_ei[np.isfinite(log_ei)]
    z = (y_best-mean[std > 0])/std[std > 0]
    diagnostics.update(variant=variant, positive_std=int(np.sum(std > 0)),
                       positive_ei=int(np.sum(pool_ei > 0)), finite_log_ei=int(len(finite_log)),
                       std_min=float(std.min()), std_max=float(std.max()),
                       z_min=float(z.min()) if len(z) else None, z_max=float(z.max()) if len(z) else None,
                       best_pool_log_ei=float(finite_log.max()) if len(finite_log) else None)
    scale = float(np.max(pool_ei))
    diagnostics["best_pool_ei"] = scale
    if (not use_log and scale == 0) or (use_log and not len(finite_log)):
        diagnostics["random_fallback"] = True
        diagnostics["selected_ei"] = 0.0
        return pool[0], diagnostics

    # Scaling leaves the EI maximum unchanged and avoids tiny stopping scales.
    def negative_ei(x):
        mean, std = gp.predict(np.asarray(x).reshape(1, -1), return_std=True)
        if use_log:
            value = float(log_expected_improvement(mean, std, y_best)[0])
            return -value if np.isfinite(value) else 1e100
        return -float(expected_improvement(mean, std, y_best)[0]) / scale

    candidates = list(pool)
    starts = np.argsort(-scores, kind="stable")[:10]
    constraint = {"type": "ineq", "fun": lambda x: 1.0 - x @ x,
                  "jac": lambda x: -2.0 * x}
    for index in starts:
        result = minimize(negative_ei, pool[index], method="SLSQP",
                          bounds=[(-1.0, 1.0)] * d, constraints=[constraint],
                          options={"ftol": 1e-9, "maxiter": 100})
        diagnostics["slsqp_failures"] += int(not result.success)
        x = result.x
        if (np.all(np.isfinite(x)) and np.linalg.norm(x) <= 1 + FEASIBILITY_TOL
                and np.min(np.linalg.norm(X - x, axis=1)) > 1e-8):
            candidates.append(x)
    candidates = np.asarray(candidates)
    mean, std = gp.predict(candidates, return_std=True)
    ei = expected_improvement(mean, std, y_best)
    selected_scores = log_expected_improvement(mean, std, y_best) if use_log else ei
    index = int(np.argmax(selected_scores))
    selected_log = log_expected_improvement(mean[index:index+1], std[index:index+1], y_best)[0]
    diagnostics["selected_log_ei"] = float(selected_log) if np.isfinite(selected_log) else None
    diagnostics["selected_ei"] = float(ei[index])
    return candidates[index], diagnostics


def run_bo(score, X_initial, budget=500, seed=0, kernel="ard", variant="baseline"):
    """The solver sees a scoring callback and starting points, never hidden geometry."""
    _check_int(budget, "budget", 1)
    _check_int(seed, "seed")
    X_initial = np.asarray(X_initial, dtype=float)
    if (X_initial.ndim != 2 or X_initial.shape[1] < 2
            or len(X_initial) != 2 * X_initial.shape[1] + 1
            or not np.all(np.isfinite(X_initial))
            or np.any(np.linalg.norm(X_initial, axis=1) > 1 + FEASIBILITY_TOL)):
        raise ValueError("Initial design must contain 2d+1 finite points in the ball")
    if budget < len(X_initial):
        raise ValueError("Budget must be at least 2d+1")
    if kernel not in ("ard", "isotropic"):
        raise ValueError("kernel must be ard or isotropic")
    if variant not in ("baseline", "logei", "local", "logei_local"):
        raise ValueError("Unknown BO variant")
    search_seed, fitting_seed = np.random.SeedSequence(seed).spawn(2)
    rng = np.random.default_rng(search_seed)
    gp_seed = int(fitting_seed.generate_state(1)[0])
    X = np.empty((0, X_initial.shape[1]))
    y = np.empty(0)
    gp = None
    timings = {"fit_seconds": 0.0, "acquisition_seconds": 0.0, "objective_seconds": 0.0}
    history, checkpoints, diagnostics, fit_warnings = [], [], [], []
    status, error = "completed", None
    started = time.perf_counter()
    try:
        for call in range(budget):
            if call < len(X_initial):
                x = X_initial[call]
            else:
                tick = time.perf_counter()
                try:
                    gp, messages = fit_gp(X, y, kernel, gp, gp_seed)
                finally:
                    timings["fit_seconds"] += time.perf_counter() - tick
                if messages:
                    fit_warnings.append({"observations": call, "messages": messages})
                tick = time.perf_counter()
                try:
                    x, info = find_next_point(gp, X, float(y.min()), rng, variant, X[int(np.argmin(y))])
                finally:
                    timings["acquisition_seconds"] += time.perf_counter() - tick
                diagnostics.append({"evaluation": call + 1, **info})
            tick = time.perf_counter()
            try:
                value = float(score(x))
            finally:
                timings["objective_seconds"] += time.perf_counter() - tick
            if not np.isfinite(value):
                raise ValueError("Objective returned a nonfinite value")
            X = np.vstack([X, x])
            y = np.append(y, value)
            best = int(np.argmin(y))
            record = {"evaluations": len(y), "best_loss": float(y[best]),
                      "elapsed_seconds": time.perf_counter() - started}
            history.append(record)
            if len(y) in (50, 100, 250, 500, budget):
                checkpoints.append({**record, "recommendation": X[best].tolist()})
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    timings["total_seconds"] = time.perf_counter() - started
    best = int(np.argmin(y)) if len(y) else None
    return {
        "status": status, "error": error, "evaluations": len(y),
        "X": X.tolist(), "y": y.tolist(),
        "recommendation": X[best].tolist() if best is not None else None,
        "best_loss": float(y[best]) if best is not None else None,
        "history": history, "checkpoints": checkpoints, "timings": timings,
        "acquisition_diagnostics": diagnostics, "fit_warnings": fit_warnings,
        "settings": {"kernel": kernel, "budget": budget, "seed": seed, "variant": variant,
                     "alpha": 1e-8, "nu": 2.5, "initial_length_scale": 0.5,
                     "length_scale_bounds": [1e-2, 10.0], "constant_bounds": [1e-3, 1e3],
                     "initial_restarts": 2, "later_restarts": 0, "pool_size": 512,
                     "local_starts": 10, "ftol": 1e-9, "maxiter": 100,
                     "novelty_tolerance": 1e-8, "feasibility_tolerance": FEASIBILITY_TOL},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
                     "matplotlib": matplotlib.__version__},
    }


def run_cmaes(score, X_initial, budget=500, seed=0):
    """CMA-ES sees only evaluated points; rejected proposals cost time, not calls."""
    _check_int(budget, "budget", 1)
    _check_int(seed, "seed")
    X_initial = np.asarray(X_initial, dtype=float)
    if (X_initial.ndim != 2 or X_initial.shape[1] < 2
            or len(X_initial) != 2 * X_initial.shape[1] + 1
            or not np.all(np.isfinite(X_initial))
            or np.any(np.linalg.norm(X_initial, axis=1) > 1 + FEASIBILITY_TOL)):
        raise ValueError("Initial design must contain 2d+1 finite points in the ball")
    if budget < len(X_initial):
        raise ValueError("Budget must be at least 2d+1")
    d = X_initial.shape[1]
    population = 4 + int(3 * np.log(d))
    sigma = 0.3 / np.sqrt(d)
    rng = np.random.default_rng(seed)
    X, y, history, checkpoints = [], [], [], []
    proposals, fitness = [], []
    rejected = generations = 0
    es = None
    status, error, termination = "completed", None, "budget"
    timings = {"objective_seconds": 0.0, "optimizer_seconds": 0.0}
    initial_mean = None
    started = time.perf_counter()
    try:
        for call in range(budget):
            if call < len(X_initial):
                x = X_initial[call]
            else:
                if es is None:
                    initial_mean = X[int(np.argmin(y))].copy()
                    es = cma.CMAEvolutionStrategy(initial_mean, sigma, {
                        "popsize": population, "CMA_diagonal": False,
                        "CMA_active": True, "CMA_mirrors": 0,
                        "randn": lambda *shape: rng.standard_normal(shape),
                        "seed": np.nan, "verbose": -9, "verb_log": 0,
                        "signals_filename": "", "eval_final_mean": False,
                        # Cancel pycma 4.5.0's tiny diagonal perturbation so C0=I.
                        "CMA_teststds": np.exp(-5e-5 * np.arange(d) / d),
                    })
                    np.testing.assert_allclose(es.sm.C, np.eye(d), rtol=0, atol=1e-14)
                if not proposals:
                    stopped = es.stop()
                    if stopped:
                        termination = {str(k): str(v) for k, v in stopped.items()}
                        break
                    proposals = es.ask()
                    fitness = []
                    generation_rejections = 0
                x = np.asarray(proposals[len(fitness)], dtype=float)
                while np.linalg.norm(x) > 1 + FEASIBILITY_TOL:
                    rejected += 1
                    generation_rejections += 1
                    if generation_rejections >= 10000:
                        raise RuntimeError("10,000 rejected proposals in one generation")
                    x = np.asarray(es.ask(1)[0], dtype=float)
                if x.shape != (d,) or not np.all(np.isfinite(x)):
                    raise ValueError("CMA-ES proposed a nonfinite or malformed point")
                proposals[len(fitness)] = x
            tick = time.perf_counter()
            try:
                value = float(score(x))
            finally:
                timings["objective_seconds"] += time.perf_counter() - tick
            if not np.isfinite(value):
                raise ValueError("Objective returned a nonfinite value")
            X.append(np.array(x, copy=True))
            y.append(value)
            if call >= len(X_initial):
                fitness.append(value)
                if len(fitness) == population:
                    es.tell(proposals, fitness)
                    generations += 1
                    proposals, fitness = [], []
            best = int(np.argmin(y))
            record = {"evaluations": len(y), "best_loss": y[best],
                      "elapsed_seconds": time.perf_counter() - started}
            history.append(record)
            if len(y) in (50, 100, 250, 500, budget):
                checkpoints.append({**record, "recommendation": X[best].tolist()})
    except Exception as exc:
        status, error, termination = "failed", f"{type(exc).__name__}: {exc}", "error"
    timings["total_seconds"] = time.perf_counter() - started
    timings["optimizer_seconds"] = timings["total_seconds"] - timings["objective_seconds"]
    best = int(np.argmin(y)) if y else None
    return {
        "status": status, "error": error, "termination_reason": termination,
        "evaluations": len(y), "X": [x.tolist() for x in X], "y": y,
        "recommendation": X[best].tolist() if best is not None else None,
        "best_loss": y[best] if best is not None else None,
        "history": history, "checkpoints": checkpoints, "timings": timings,
        "generations": generations, "rejected_proposals": rejected,
        "incomplete_population_evaluations": len(fitness),
        "settings": {"budget": budget, "seed": seed, "population_size": population,
                     "initial_sigma": float(sigma), "initial_mean": initial_mean.tolist() if initial_mean is not None else None,
                     "initial_covariance": "identity (diagonal perturbation compensated)",
                     "active_covariance": True, "mirrors": 0, "restarts": 0,
                     "max_rejections_per_generation": 10000,
                     "feasibility_tolerance": FEASIBILITY_TOL},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "cma": cma.__version__},
    }


def run_comparison(output, study="staged", retry_failed=False):
    """Save one result at a time; compatible completed runs are never repeated."""
    if any(os.environ.get(name) != "1" for name in THREAD_VARIABLES):
        raise ValueError("Run --compare, or set all THREAD_VARIABLES=1 before importing this module")
    if study not in ("staged", "full", "heldout", "timing"):
        raise ValueError("Unknown study")
    budget = 100 if study == "heldout" else 500
    revised = study in ("heldout", "timing")
    instances = range(3,6) if study == "heldout" else [3] if study == "timing" else range(3 if study == "staged" else 10)
    run_ids = [0] if study == "timing" else range(5 if study == "full" else 2)
    orientations = ["rotated"] if study == "timing" else ("aligned", "rotated")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "runs").mkdir(exist_ok=True)
    configurations = [(5, 0, (1, 10000))] if study in ("staged", "heldout") else [(5, 0, (10000,))] if study == "timing" else [
        (5, 0, (1, 100, 10000)), (10, 0, (100, 10000)), (5, 1, (100, 10000))]
    fingerprint = {"code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                                "cma": cma.__version__, "threadpoolctl": threadpoolctl.__version__},
                   "budget": budget, "bo_variant": "logei_local" if revised else "baseline", "master_seed": MASTER_SEED, "threads": 1,
                   "thread_environment": {name: os.environ[name] for name in THREAD_VARIABLES},
                   "numpy_blas": np.__config__.CONFIG["Build Dependencies"]["blas"]}
    jobs = []
    for d, beta, kappas in configurations:
        for k in kappas:
            for orientation in orientations:
                for instance in instances:
                    for run in run_ids:
                        for method in ("ard", "isotropic", "cmaes"):
                            spec = dict(dimension=d, conditioning=k, orientation=orientation,
                                        beta=beta, instance=instance)
                            variant = ("logei_local" if revised else "baseline") if method != "cmaes" else None
                            prefix = f"{method}_{variant}" if revised and variant else method
                            name = f"{prefix}_d{d}_k{k}_{orientation}_beta{beta}_i{instance}_r{run}_b{budget}.json"
                            jobs.append(dict(problem=spec, run=run, method=method, variant=variant, file=name))
    # Shuffle once, reproducibly, to avoid always timing one method first.
    np.random.default_rng(20261104).shuffle(jobs)
    manifest = dict(study=study, fingerprint=fingerprint, jobs=jobs)
    manifest_path = output / f"manifest_{study}.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Incompatible comparison manifest: use a new output directory")
    for job in jobs:
        path = output / "runs" / job["file"]
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get("fingerprint") != fingerprint or saved.get("job") != job:
                raise ValueError(f"Incompatible saved run: {path}; use a new output directory")
    temp = manifest_path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, indent=2) + "\n")
    temp.replace(manifest_path)
    durations = []
    with threadpoolctl.threadpool_limits(limits=1):
        for number, job in enumerate(jobs, 1):
            path = output / "runs" / job["file"]
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["status"] == "completed" or not retry_failed:
                    print(f"[{number}/{len(jobs)}] Reuse {job['file']}", flush=True)
                    continue
                archive = path.with_suffix(f".failed-{time.time_ns()}.json")
                path.rename(archive)
            p = make_problem(**job["problem"])
            initial = initial_design(p, job["run"])
            state = {"budget": budget, "evaluations": 0}
            seed = int(np.random.SeedSequence([MASTER_SEED, p["dimension"], p["instance"],
                       job["run"], 4 if job["method"] == "cmaes" else 3]).generate_state(1)[0])
            print(f"[{number}/{len(jobs)}] Starting {job['file']}", flush=True)
            if job["method"] == "cmaes":
                result = run_cmaes(lambda x: evaluate(p, state, x), initial, budget, seed)
            else:
                result = run_bo(lambda x: evaluate(p, state, x), initial, budget, seed, job["method"], variant=job["variant"])
            result.update(job=job, fingerprint=fingerprint, evaluator_calls=state["evaluations"],
                          threadpools=threadpoolctl.threadpool_info())
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
            temp.replace(path)
            durations.append(result["timings"]["total_seconds"])
            estimate = np.mean(durations) * (len(jobs) - number) / 60
            print(f"  {result['status']}: {result['evaluations']} calls, loss={result['best_loss']}; "
                  f"rough remaining time {estimate:.1f} min", flush=True)
    return analyse_comparison(output, study)


def analyse_comparison(output, study="staged"):
    """Validate raw runs, pair observations, and bootstrap whole problem instances."""
    output = Path(output)
    manifest = json.loads((output / f"manifest_{study}.json").read_text())
    budget = manifest["fingerprint"]["budget"]
    checkpoints = [v for v in (50,100,250,500) if v <= budget]
    if budget not in checkpoints:
        checkpoints.append(budget)
    revised = manifest["fingerprint"].get("bo_variant") == "logei_local"
    labels = {"ard": "Revised ARD BO" if revised else "ARD BO",
              "isotropic": "Revised isotropic BO" if revised else "Isotropic BO", "cmaes": "CMA-ES"}
    destination = output / study
    destination.mkdir(exist_ok=True)
    records, curves, missing, failures = [], {}, [], []
    for job in manifest["jobs"]:
        path = output / "runs" / job["file"]
        if not path.exists():
            missing.append(job["file"])
            continue
        r = json.loads(path.read_text())
        if r.get("fingerprint") != manifest["fingerprint"] or r.get("job") != job:
            raise ValueError(f"Incompatible record: {path}")
        if r["settings"].get("budget") != budget:
            raise ValueError(f"Mismatched solver budget: {path}")
        if job["method"] != "cmaes" and r["settings"].get("variant", "baseline") != job.get("variant", "baseline"):
            raise ValueError(f"Mismatched BO variant: {path}")
        p = make_problem(**job["problem"])
        x, y = np.asarray(r["X"]), np.asarray(r["y"])
        n = len(y)
        if n != r["evaluations"] or n != r["evaluator_calls"] or n > budget:
            raise ValueError(f"Invalid evaluation accounting: {path}")
        if n:
            if not all(is_feasible(p, v) for v in x):
                raise ValueError(f"Infeasible observations: {path}")
            np.testing.assert_allclose(loss(p, x), y, rtol=1e-10, atol=1e-14)
            initial = initial_design(p, job["run"])
            np.testing.assert_array_equal(x[:min(n, len(initial))], initial[:min(n, len(initial))])
            np.testing.assert_allclose(loss(p, r["recommendation"]), y.min(), rtol=1e-10, atol=1e-14)
            np.testing.assert_array_equal(r["recommendation"], x[np.argmin(y)])
            assert r["best_loss"] == float(y.min())
        if r["status"] != "completed":
            failures.append(dict(file=job["file"], error=r["error"], evaluations=n))
            continue
        if not n or (n < budget and not isinstance(r.get("termination_reason"), dict)):
            raise ValueError(f"Unexplained early stop: {path}")
        best = np.minimum.accumulate(y)
        np.testing.assert_array_equal([h["best_loss"] for h in r["history"]], best)
        assert [h["evaluations"] for h in r["history"]] == list(range(1, n+1))
        for checkpoint in r["checkpoints"]:
            t = checkpoint["evaluations"]
            assert checkpoint["best_loss"] == best[t-1]
            np.testing.assert_array_equal(checkpoint["recommendation"], x[np.argmin(y[:t])])
        key = (p["dimension"], p["beta"], p["conditioning"], p["orientation"],
               p["instance"], job["run"], job["method"])
        curves[key] = np.log10(np.maximum(np.pad(best, (0, budget-n), mode="edge"), 1e-12))
        for checkpoint in checkpoints:
            regret = float(best[min(checkpoint, n)-1])
            records.append(dict(**job["problem"], run=job["run"], method=job["method"],
                                variant=job.get("variant", "baseline" if job["method"] != "cmaes" else None),
                                checkpoint=checkpoint, regret=regret,
                                log_regret=float(np.log10(max(regret, 1e-12))),
                                actual_calls=n, carried_forward=checkpoint > n,
                                total_seconds=r["timings"]["total_seconds"],
                                objective_seconds=r["timings"]["objective_seconds"]))
        if study == "timing":
            heldout_manifest = json.loads((output / "manifest_heldout.json").read_text())
            paired_job = next(j for j in heldout_manifest["jobs"] if j["problem"] == job["problem"] and j["run"] == job["run"] and j["method"] == job["method"])
            shorter = json.loads((output / "runs" / paired_job["file"]).read_text())
            if shorter["status"] != "completed" or shorter["job"].get("variant") != job.get("variant"):
                raise ValueError("Timing pilot requires a successful matching validation run")
            assert shorter["fingerprint"] == heldout_manifest["fingerprint"]
            assert {k:v for k,v in shorter["fingerprint"].items() if k != "budget"} == {k:v for k,v in r["fingerprint"].items() if k != "budget"}
            count = min(len(shorter["y"]), n)
            np.testing.assert_array_equal(x[:count], shorter["X"][:count])
            np.testing.assert_array_equal(y[:count], shorter["y"][:count])
    # Contrasts are formed before any averaging or resampling.
    effects = {}
    for key, curve in curves.items():
        d, beta, k, orientation, instance, run, method = key
        for checkpoint in checkpoints:
            candidates = []
            if method != "cmaes":
                other = (d, beta, k, orientation, instance, run, "cmaes")
                if other in curves:
                    candidates.append((("method", d, beta, k, orientation, method, checkpoint),
                                       curve[checkpoint-1] - curves[other][checkpoint-1]))
            if orientation == "rotated":
                other = (d, beta, k, "aligned", instance, run, method)
                if other in curves:
                    penalty = curve[checkpoint-1] - curves[other][checkpoint-1]
                    candidates.append((("rotation", d, beta, k, "paired", method, checkpoint), penalty))
                    low_r = (d, beta, 1, "rotated", instance, run, method)
                    low_a = (d, beta, 1, "aligned", instance, run, method)
                    if k == 10000 and low_r in curves and low_a in curves:
                        candidates.append((("interaction", d, beta, k, "paired", method, checkpoint),
                                           penalty - (curves[low_r][checkpoint-1] - curves[low_a][checkpoint-1])))
            for effect, value in candidates:
                effects.setdefault(effect, {}).setdefault(instance, []).append(float(value))
    contrasts = []
    rng = np.random.default_rng(20261103)
    for effect, instances in (sorted(effects.items()) if study != "timing" else []):
        means = np.array([np.mean(v) for _, v in sorted(instances.items())])
        draws = rng.integers(0, len(means), size=(10000, len(means)))
        interval = np.percentile(means[draws].mean(axis=1), [2.5, 97.5])
        kind, d, beta, k, orientation, method, checkpoint = effect
        contrasts.append(dict(kind=kind, dimension=d, beta=beta, conditioning=k,
                              orientation=orientation, method=method, checkpoint=checkpoint,
                              mean=float(means.mean()), lower=float(interval[0]), upper=float(interval[1]),
                              instances=len(means), pairs=sum(map(len, instances.values())),
                              instance_means={str(i): float(np.mean(v)) for i,v in instances.items()}))
    summary = dict(study=study, budget=budget, bo_variant=manifest["fingerprint"].get("bo_variant", "baseline"), expected=len(manifest["jobs"]), completed=len(curves),
                   missing=missing, failures=failures, bootstrap_seed=None if study == "timing" else 20261103,
                   bootstrap_resamples=0 if study == "timing" else 10000, floor=1e-12, contrasts=contrasts)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, rows in (("checkpoints", records), ("paired_effects", contrasts)):
        with (destination / f"{name}.csv").open("w", newline="") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    if curves:
        methods = ("ard", "isotropic", "cmaes")
        for d, beta in sorted({(key[0],key[1]) for key in curves}):
            conditions = sorted({(key[2], key[3]) for key in curves if key[:2] == (d,beta)})
            fig, axes = plt.subplots((len(conditions)+1)//2, 2, figsize=(11, 3.5*((len(conditions)+1)//2)), squeeze=False, layout="constrained")
            for ax, (k, orientation) in zip(axes.flat, conditions):
                for method in methods:
                    by_instance = {}
                    for key, curve in curves.items():
                        if key[:4] == (d,beta,k,orientation) and key[-1] == method:
                            by_instance.setdefault(key[4], []).append(curve)
                    if by_instance:
                        instance_curves = [np.mean(v,axis=0) for v in by_instance.values()]
                        ax.plot(range(1,budget+1), np.mean(instance_curves,axis=0), label=labels[method])
                ax.set(title=f"κ={k}, {orientation}", xlabel="Objective evaluations",
                       ylabel="Mean log10 regret (floor 1e-12)", xlim=(2*d+1,budget))
                ax.legend(); ax.grid(alpha=.2)
            for ax in axes.flat[len(conditions):]:
                ax.set_visible(False)
            for suffix in ("png", "pdf"):
                fig.savefig(destination / f"convergence_d{d}_beta{beta}.{suffix}", dpi=180)
            plt.close(fig)
        for kind in ("method", "rotation", "interaction"):
            rows = [r for r in contrasts if r["kind"] == kind and r["checkpoint"] == budget]
            if not rows:
                continue
            fig, ax = plt.subplots(figsize=(10, max(3,len(rows)*.4)), layout="constrained")
            for index, r in enumerate(rows):
                ax.plot([r["lower"],r["upper"]],[index,index], color="C0")
                ax.plot(r["mean"],index,"o",color="C0")
                ax.scatter(list(r["instance_means"].values()), [index]*r["instances"], color="C1",s=15)
            ax.set_yticks(range(len(rows)), [f"d={r['dimension']} β={r['beta']} κ={r['conditioning']} {r['orientation']} {r['method']} (n={r['instances']})" for r in rows])
            ax.axvline(0,color="gray",lw=1)
            ax.set(xlabel="Paired log10-regret difference", title=f"{kind.capitalize()} effects at {budget} calls; exploratory 95% instance bootstrap")
            for suffix in ("png","pdf"):
                fig.savefig(destination / f"{kind}_effects.{suffix}",dpi=180)
            plt.close(fig)
        fig, ax = plt.subplots(figsize=(7,4),layout="constrained")
        for index, method in enumerate(methods):
            values = [r["total_seconds"] for r in records if r["checkpoint"]==budget and r["method"]==method]
            if values:
                ax.scatter(np.full(len(values),index),values,alpha=.5)
                q = np.percentile(values,[25,50,75])
                ax.plot([index,index],[q[0],q[2]],color="black",lw=3)
                ax.plot(index,q[1],"_",color="black",ms=15)
        ax.set(xticks=range(3),xticklabels=[labels[m] for m in methods],yscale="log",ylabel="Total search seconds",title="Timing pilot: one run per method" if study == "timing" else "Runtime: individual runs, median and IQR")
        for suffix in ("png","pdf"):
            fig.savefig(destination / f"runtime.{suffix}",dpi=180)
        plt.close(fig)
    print(f"Analysis: {len(curves)}/{len(manifest['jobs'])} completed, {len(failures)} failed, {len(missing)} missing",flush=True)
    return summary


def diagnose_bo(output):
    """Frozen-model audit plus a fixed 2x2 acquisition experiment; no GP retuning."""
    output = Path(output)
    baseline = Path(__file__).resolve().parents[1] / "results" / "comparison"
    checks_path = Path(__file__).resolve().parents[1] / "results" / "bo_diagnostics" / "baseline_checksums.json"
    checks = json.loads(checks_path.read_text())
    repository = Path(__file__).resolve().parents[1]
    for name, digest in checks.items():
        if hashlib.sha256((repository / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Baseline changed: {name}")
    if any(os.environ.get(name) != "1" for name in THREAD_VARIABLES):
        raise ValueError("Use --bo-diagnostic to set thread limits before imports")
    output.mkdir(parents=True, exist_ok=True)
    (output / "runs").mkdir(exist_ok=True)
    variants = ("baseline", "logei", "local", "logei_local")
    fingerprint = dict(code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       baseline_checks=hashlib.sha256(checks_path.read_bytes()).hexdigest(),
                       numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                       python=platform.python_version(), budget=100,
                       thread_environment={name: os.environ[name] for name in THREAD_VARIABLES})
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != fingerprint:
        raise ValueError("Diagnostic configuration changed: choose a new output directory")
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(fingerprint, indent=2)+"\n"); temp.replace(path)
    summaries, frozen, outcomes = [], [], []
    with threadpoolctl.threadpool_limits(limits=1):
        for kernel in ("ard", "isotropic"):
            for kappa in (1, 10000):
                for orientation in ("aligned", "rotated"):
                    stem = f"{kernel}_d5_k{kappa}_{orientation}_beta0_i0_r0"
                    original = json.loads((baseline / "runs" / f"{stem}_b500.json").read_text())
                    X, y = np.asarray(original["X"]), np.asarray(original["y"])
                    seed = original["settings"]["seed"]
                    problem = make_problem(5, kappa, orientation)
                    audit_path = output / f"frozen_{stem}.json"
                    if audit_path.exists():
                        audit = json.loads(audit_path.read_text())
                        if audit["fingerprint"] != fingerprint:
                            raise ValueError("Incompatible frozen-model audit")
                        frozen.extend(audit["records"])
                    else:
                        print(f"Replay {stem}", flush=True)
                        zeros = [v["evaluation"] for v in original["acquisition_diagnostics"] if v["random_fallback"]]
                        targets = {50, 100} | ({zeros[0]} if zeros else set())
                        search_seed, fitting_seed = np.random.SeedSequence(seed).spawn(2)
                        rng = np.random.default_rng(search_seed)
                        gp_seed = int(fitting_seed.generate_state(1)[0])
                        gp, records = None, []
                        for n in range(11, max(targets)):
                            gp, messages = fit_gp(X[:n], y[:n], kernel, gp, gp_seed)
                            if n+1 in targets:
                                incumbent = X[int(np.argmin(y[:n]))]
                                for variant in variants:
                                    point, info = find_next_point(gp, X[:n], float(y[:n].min()), copy.deepcopy(rng), variant, incumbent)
                                    if variant == "baseline":
                                        np.testing.assert_allclose(point, X[n], rtol=0, atol=1e-10)
                                    mean, std = gp.predict(point[None,:],return_std=True)
                                    records.append(dict(kernel=kernel, conditioning=kappa, orientation=orientation,
                                                        evaluation=n+1, variant=variant, kernel_theta=gp.kernel_.theta.tolist(),
                                                        candidate=point.tolist(), mean=float(mean[0]), std=float(std[0]),
                                                        observed_best=float(y[:n].min()), offline_loss=float(loss(problem, point)),
                                                        fit_warnings=messages, **{key:value for key,value in info.items() if key != "variant"}))
                            old = original["acquisition_diagnostics"][n-11]
                            for _ in range(old["extra_pools"]+1):
                                sample_ball(5,512,rng)
                        audit = dict(fingerprint=fingerprint, records=records)
                        temp = audit_path.with_suffix(".tmp")
                        temp.write_text(json.dumps(audit,indent=2,allow_nan=False)+"\n"); temp.replace(audit_path)
                        frozen.extend(records)
                    for variant in variants:
                        target = output / "runs" / f"{stem}_{variant}_b100.json"
                        if target.exists():
                            result = json.loads(target.read_text())
                            if result.get("fingerprint") != fingerprint:
                                raise ValueError(f"Incompatible result: {target}")
                            print(f"Reuse {target.name}", flush=True)
                        else:
                            print(f"Run {target.name}", flush=True)
                            state = {"budget": 100, "evaluations": 0}
                            result = run_bo(lambda x: evaluate(problem,state,x), initial_design(problem,0),100,seed,kernel,variant)
                            result.update(fingerprint=fingerprint, evaluator_calls=state["evaluations"],
                                          problem=dict(dimension=5,conditioning=kappa,orientation=orientation,beta=0,instance=0),run=0)
                            temp = target.with_suffix(".tmp")
                            temp.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n"); temp.replace(target)
                        if result["status"] == "completed":
                            assert result["evaluations"] == result["evaluator_calls"] == 100
                            assert all(is_feasible(problem, x) for x in result["X"])
                            np.testing.assert_allclose(loss(problem,result["X"]), result["y"],rtol=1e-10,atol=1e-14)
                            np.testing.assert_array_equal(result["recommendation"],np.asarray(result["X"])[np.argmin(result["y"])])
                            if variant == "baseline":
                                np.testing.assert_array_equal(result["X"],X[:100])
                                np.testing.assert_array_equal(result["y"],y[:100])
                        summaries.append(dict(kernel=kernel,conditioning=kappa,orientation=orientation,variant=variant,
                                              status=result["status"],error=result["error"],evaluations=result["evaluations"],
                                              loss50=min(result["y"][:50]) if result["y"] else None,
                                              loss100=result["best_loss"] if result["evaluations"]==100 else None,
                                              fallbacks=sum(v["random_fallback"] for v in result["acquisition_diagnostics"]),
                                              slsqp_failures=sum(v["slsqp_failures"] for v in result["acquisition_diagnostics"]),
                                              seconds=result["timings"]["total_seconds"]))
                        outcomes.append(result)
                        print(f"  {result['status']}; best={result['best_loss']}",flush=True)
    for name, rows in (("runs",summaries),("frozen",frozen)):
        with (output / f"{name}.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader(); writer.writerows(rows)
    summary=dict(expected=32,completed=sum(r["status"]=="completed" for r in summaries),
                 runs=summaries, frozen_states=len(frozen)//4, fingerprint=fingerprint)
    (output / "summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    fig,axes=plt.subplots(4,2,figsize=(11,13),layout="constrained")
    for index,ax in enumerate(axes.flat):
        for row,result in zip(summaries[4*index:4*index+4],outcomes[4*index:4*index+4]):
            if result["y"]:
                ax.plot(range(1,len(result["y"])+1),np.log10(np.maximum(np.minimum.accumulate(result["y"]),1e-12)),label=row["variant"])
        ax.set(title=f"{row['kernel']} κ={row['conditioning']} {row['orientation']}",xlabel="Objective calls",ylabel="log10 regret (floor 1e-12)",xlim=(11,100))
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    for suffix in ("png","pdf"):
        fig.savefig(output / f"convergence.{suffix}",dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout="constrained")
    ax.bar(variants,[sum(r["fallbacks"] for r in summaries if r["variant"]==v) for v in variants])
    ax.set(ylabel="Random fallbacks across eight runs",title="Acquisition fallback counts (100-call budget)")
    for suffix in ("png","pdf"):
        fig.savefig(output / f"fallbacks.{suffix}",dpi=180)
    plt.close(fig)
    for name,digest in checks.items():
        assert hashlib.sha256((repository/name).read_bytes()).hexdigest()==digest, name
    print(f"Diagnostics: {summary['completed']}/32 completed; baseline checksums unchanged",flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="run mathematical, BO and CMA-ES checks")
    modes.add_argument("--bo", action="store_true", help="run one BO experiment")
    modes.add_argument("--cmaes", action="store_true", help="run one CMA-ES experiment")
    parser.add_argument("--dimension", type=int, default=5)
    parser.add_argument("--conditioning", type=float, default=100.0)
    parser.add_argument("--orientation", choices=("aligned", "rotated"), default="aligned")
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--kernel", choices=("ard", "isotropic"), default="ard")
    modes.add_argument("--compare", action="store_true", help="run or resume the comparison")
    modes.add_argument("--analyse", action="store_true", help="analyse saved comparison runs")
    parser.add_argument("--study", choices=("staged", "full", "heldout", "timing"), default="staged")
    parser.add_argument("--retry-failed", action="store_true")
    modes.add_argument("--bo-diagnostic", action="store_true", help="replay and test four BO acquisition variants")
    args = parser.parse_args()
    if args.bo_diagnostic:
        result = diagnose_bo(args.output or Path(__file__).resolve().parents[1] / "results" / "bo_diagnostics")
        if result["completed"] != result["expected"]:
            raise RuntimeError("Some diagnostic runs failed; inspect summary.json")
        return
    if args.compare or args.analyse:
        expected_budget = 100 if args.study == "heldout" else 500
        if args.budget is not None and args.budget != expected_budget:
            parser.error(f"Study {args.study} fixes the budget at {expected_budget}")
        folder = "heldout_comparison" if args.study in ("heldout", "timing") else "comparison"
        output = args.output or Path(__file__).resolve().parents[1] / "results" / folder
        if args.compare:
            result = run_comparison(output, args.study, args.retry_failed)
            if result["failures"] or result["missing"]:
                raise RuntimeError("Comparison has failed or missing runs; inspect its summary")
        else:
            analyse_comparison(output, args.study)
        return
    args.budget = 500 if args.budget is None else args.budget
    # Experiment settings are shared by the manifest and the checks.
    specs = []
    for d, beta, kappas in ((5, 0, (1, 100, 10000)), (10, 0, (100, 10000)), (5, 1, (100, 10000))):
        for kappa in kappas:
            for orientation in ("aligned", "rotated"):
                for instance in range(10):
                    specs.append({
                        "dimension": d, "conditioning": kappa, "orientation": orientation,
                        "beta": beta, "instance": instance,
                    })

    if args.check:
        run_checks(specs)
        return
    if args.output is None:
        args.output = Path(__file__).resolve().parents[1] / "results" / ("bo" if args.bo else "cmaes" if args.cmaes else "problem_design")
    if args.bo or args.cmaes:
        spec = dict(dimension=args.dimension, conditioning=args.conditioning,
                    orientation=args.orientation, beta=args.beta, instance=args.instance)
        problem = make_problem(**spec)
        points = initial_design(problem, args.run)
        state = {"budget": args.budget, "evaluations": 0}
        seed = int(np.random.SeedSequence([MASTER_SEED, args.dimension, args.instance,
                                          args.run, 3 if args.bo else 4]).generate_state(1)[0])
        if args.bo:
            result = run_bo(lambda x: evaluate(problem, state, x), points,
                            args.budget, seed, args.kernel)
        else:
            result = run_cmaes(lambda x: evaluate(problem, state, x), points, args.budget, seed)
        result["problem"] = spec
        result["run"] = args.run
        result["master_seed"] = MASTER_SEED
        # The evaluator knows f*=0; the optimiser only receives score observations.
        result["simple_regret"] = result["best_loss"]
        args.output.mkdir(parents=True, exist_ok=True)
        name = (f"{args.kernel if args.bo else 'cmaes'}_d{args.dimension}_k{args.conditioning:g}_{args.orientation}"
                f"_beta{args.beta:g}_i{args.instance}_r{args.run}_b{args.budget}.json")
        target = args.output / name
        target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"{'BO' if args.bo else 'CMA-ES'} {result['status']}: {result['evaluations']} calls, best observed loss = {result['best_loss']}")
        print(f"Saved {target}")
        if result["status"] != "completed":
            raise RuntimeError(result["error"])
        return
    validation = validate_landscapes()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1, "master_seed": MASTER_SEED,
        "description": "Evaluator configuration only; no optimisation runs have been performed.",
        "domain": "unit Euclidean ball", "noise": "none", "budget": 500,
        "checkpoints": [50, 100, 250, 500], "runs_per_instance": 5,
        "initial_design_size": "2 * dimension + 1", "run_ids": list(range(5)),
        "conditions": 14, "instances_per_condition": 10,
        "problems": specs,
    }
    for name, data in (("experiment_manifest.json", manifest), ("validation.json", validation)):
        (args.output / name).write_text(json.dumps(data, indent=2) + "\n")
    landscapes(args.output, 0)
    landscapes(args.output, 1)
    curvature_figure(args.output)
    print(json.dumps(validation, indent=2))
    print(f"Wrote manifest and three PNG/PDF figures to {args.output}")


if __name__ == "__main__":
    main()
