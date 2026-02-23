"""benchmark_fro.py

Ground-truth benchmark for NMF with Frobenius (MSE) loss.

Data generation follows the paper:
    M = X̂Ŷ  (X̂ ∈ R^{m×r}, Ŷ ∈ R^{r×n}, entries i.i.d. from U[0,1])
    SVD:  M = U diag(σ_1,...,σ_r, 0,...,0) Vᵀ
    Tail: σ_{r+1},...,σ_{min(m,n)} ~ U[0, 0.1 σ_r]
    Z   = U diag(σ_1,...,σ_r, σ_{r+1},...,σ_{min(m,n)}) Vᵀ

Optimal function value (achievable by the non-negative pair X̂, Ŷ):
    f* = (1/2mn) ||Z - X̂Ŷ||_F^2 = (1/2mn) Σ_{i=r+1}^{min(m,n)} σ_i^2

Convergence plots show the optimality gap  f(w_k) - f*.
"""

from nmf import NMFProblemFroInstance
import numpy as np
import scipy
from scipy.optimize import minimize, Bounds
import time
import os
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Optimizer base class
# ---------------------------------------------------------------------------

class BaseOptimizer(ABC):
    """Abstract base class for all optimizers."""
    def __init__(self, name, options=None):
        self.name = name
        self.options = options if options is not None else {}
        self.history = []
        self.solve_time = 0.0
        self.max_iters = self.options.get('maxiter', 200)

    @abstractmethod
    def solve(self, problem, w0):
        """Solves the given problem starting from w0.
        Must populate self.history with the loss at each iteration."""
        pass

    def __str__(self):
        return f"{self.name} (Time: {self.solve_time:.2f}s)"


# ---------------------------------------------------------------------------
# Scipy wrapper
# ---------------------------------------------------------------------------

class ScipyOptimizer(BaseOptimizer):
    """Wrapper for scipy.optimize.minimize methods."""
    def __init__(self, name, method, options=None):
        super().__init__(name, options)
        self.method = method
        self.result = None
        self._problem = None

    def _callback(self, xk, *args):
        self.history.append(self._problem.objective(xk))

    def solve(self, problem, w0):
        self._problem = problem
        self.history = [problem.objective(w0)]

        num_vars = w0.size
        bounds = Bounds(lb=np.zeros(num_vars), ub=np.inf)

        kwargs = {
            'fun': problem.objective,
            'x0': w0,
            'method': self.method,
            'jac': problem.gradient,
            'bounds': bounds,
            'callback': self._callback,
            'options': self.options,
        }

        if self.method == 'trust-constr':
            kwargs['hess'] = problem.full_hessian

        print(f"\n--- Running {self.name} ---")
        start_time = time.time()
        self.result = minimize(**kwargs)
        self.solve_time = time.time() - start_time

        print(self.result.message)
        print(f"--- {self.name} finished in {self.solve_time:.2f} seconds ---")


# ---------------------------------------------------------------------------
# IPOPT wrapper (optional – requires cyipopt)
# ---------------------------------------------------------------------------

class IpoptOptimizer(BaseOptimizer):
    """Wrapper for IPOPT via cyipopt with simple bound constraints (w >= 0)."""

    def __init__(self, name="IPOPT (cyipopt)", options=None):
        super().__init__(name, options)
        self.info = None
        self.solution = None

    def solve(self, problem, w0):
        print(f"\n--- Running {self.name} ---")
        start_time = time.time()

        try:
            import cyipopt
        except ImportError as e:
            raise ImportError(
                "cyipopt is required for the IPOPT optimizer. "
                "Install with `pip install cyipopt` or "
                "`conda install -c conda-forge cyipopt ipopt`."
            ) from e

        n = int(w0.size)
        m = 0
        ipopt_inf = 2.0e19
        lb = np.zeros(n, dtype=float)
        ub = np.full(n, ipopt_inf, dtype=float)
        cl = np.zeros(m, dtype=float)
        cu = np.zeros(m, dtype=float)

        max_iter = int(self.options.get('maxiter', self.max_iters))
        print_level = int(self.options.get('print_level', 0))
        tol = self.options.get('tol', None)
        use_hessian = bool(self.options.get('use_hessian', False))
        hess_approx = self.options.get(
            'hessian_approximation',
            'exact' if use_hessian else 'limited-memory'
        )
        bound_relax_factor = float(self.options.get('bound_relax_factor', 0.0))

        self.history = [float(problem.objective(w0))]
        outer = self

        class _Callbacks:
            def objective(self, x):
                return float(problem.objective(np.maximum(np.asarray(x, dtype=float), 0.0)))

            def gradient(self, x):
                return np.asarray(
                    problem.gradient(np.maximum(np.asarray(x, dtype=float), 0.0)),
                    dtype=float
                )

            def constraints(self, x):
                return np.zeros(0, dtype=float)

            def jacobian(self, x):
                return np.zeros(0, dtype=float)

            def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du,
                             mu, d_norm, regularization_size, alpha_du, alpha_pr,
                             ls_trials):
                val = float(obj_value)
                if len(outer.history) == 0 or abs(outer.history[-1] - val) > 1e-15:
                    outer.history.append(val)
                return True

            if use_hessian:
                def hessianstructure(self):
                    return np.tril_indices(n)

                def hessian(self, x, lagrange, obj_factor):
                    x_eval = np.maximum(np.asarray(x, dtype=float), 0.0)
                    H = obj_factor * problem.full_hessian(x_eval)
                    row, col = np.tril_indices(n)
                    return np.asarray(H[row, col], dtype=float)

        nlp = cyipopt.Problem(
            n=n, m=m,
            problem_obj=_Callbacks(),
            lb=lb, ub=ub, cl=cl, cu=cu,
        )
        nlp.add_option('max_iter', max_iter)
        nlp.add_option('print_level', print_level)
        nlp.add_option('hessian_approximation', hess_approx)
        nlp.add_option('bound_relax_factor', bound_relax_factor)
        if tol is not None:
            nlp.add_option('tol', float(tol))

        reserved = {'maxiter', 'print_level', 'tol', 'use_hessian',
                    'hessian_approximation', 'bound_relax_factor'}
        for k, v in self.options.items():
            if k not in reserved:
                try:
                    nlp.add_option(k, v)
                except Exception:
                    pass

        x_opt, info = nlp.solve(w0)
        self.solution = x_opt
        self.info = info
        self.solve_time = time.time() - start_time
        print(info.get('status_msg', 'IPOPT finished.'))
        print(f"--- {self.name} finished in {self.solve_time:.2f} seconds ---")


# ---------------------------------------------------------------------------
# ARA optimizer
# ---------------------------------------------------------------------------

class CustomARAOptimizer(BaseOptimizer):
    """Adaptive Regularization Algorithm (ARA) with regularized Newton steps."""
    def __init__(self, name, options=None):
        super().__init__(name, options)
        self.kappa = self.options.get('kappa', 1.0)
        self.ell = self.options.get('ell', 1.0)
        self.gammas = self.options.get('gammas', (0.5, 2.0, 2.0))
        self.etas = self.options.get('etas', (0.01, 0.9))

    def solve(self, problem, w0):
        print(f"\n--- Running {self.name} ---")
        start_time = time.time()

        reg_class = problem.get_regularizer_class()
        reg_problem = reg_class(problem.Z, problem.r)

        w = w0.copy()
        self.history = []

        ell = self.ell
        kappa = self.kappa
        gamma1, gamma2, gamma3 = self.gammas
        eta1, eta2 = self.etas

        for k in range(self.max_iters):
            obj_old = problem.objective(w)
            self.history.append(obj_old)
            g = problem.gradient(w)

            reg_hess = reg_problem.hessian(w, ell)
            if reg_hess.ndim == 1:
                H_sc = problem.full_hessian(w) + np.diag(reg_hess)
            else:
                H_sc = problem.full_hessian(w) + reg_hess

            flag = False
            rho = -1
            obj_new = np.inf
            w_new = w

            try:
                d = scipy.linalg.solve(H_sc, -g, assume_a='sym')
                lam = np.dot(d, -g)
                if lam >= 0:
                    w_new = w + d / (1 + kappa * np.sqrt(lam))
                    if np.any(w_new < 0):
                        flag = True
                        obj_new_val = np.nan
                    else:
                        obj_new_val = problem.objective(w_new)
                    if not (np.isinf(obj_new_val) or np.isnan(obj_new_val)):
                        obj_new = obj_new_val
                        model_decrease = (kappa * lam - np.log(1 + kappa * lam)) / kappa ** 2
                        rho = (obj_old - obj_new) / (model_decrease + 1e-12)
                    else:
                        flag = True
            except (np.linalg.LinAlgError, ValueError):
                flag = True

            if (k + 1) % 20 == 0 or k == 0:
                print(f"Iter {k+1:4} | Loss: {obj_old:.8f} | Rho: {rho:.4f} "
                      f"| ell: {ell:.4f} | kappa: {kappa:.4f}")

            if not flag and rho >= eta1:
                w = w_new
            elif abs(obj_old - obj_new) < 1e-8:
                print(f"\nConverged at iteration {k+1} due to small change in objective.")
                break

            if not flag:
                if rho > eta2:
                    ell = gamma1 * ell
                elif rho < eta1:
                    ell = gamma2 * ell
            else:
                ell = gamma3 * ell

        self.solve_time = time.time() - start_time
        print(f"--- {self.name} finished in {self.solve_time:.2f} seconds ---")


# ---------------------------------------------------------------------------
# COCARC optimizer
# ---------------------------------------------------------------------------

class CustomCOCARCOptimizer(BaseOptimizer):
    """Adaptive Regularization with Cubics for Convex Constraints (COCARC).

    Lightweight implementation of Algorithm 2.1 from:
    Cartis, Gould & Toint, *An adaptive cubic regularization algorithm for
    nonconvex optimization with convex constraints*.
    Bound constraints w >= 0 are handled via projection.
    """

    def __init__(self, name, options=None):
        super().__init__(name, options)
        self.sigma = float(self.options.get('sigma', 1.0))
        self.etas = self.options.get('etas', (0.01, 0.9))
        self.gammas = self.options.get('gammas', (0.5, 2.0))
        self.kappa_ubs = float(self.options.get('kappa_ubs', 0.1))
        self.kappa_lbs = float(self.options.get('kappa_lbs', 0.9))
        self.kappa_epp = float(self.options.get('kappa_epp', 0.25))
        self.t0 = float(self.options.get('t0', 1.0))
        self.tol = float(self.options.get('tol', 1e-6))
        self.max_line_iters = int(self.options.get('max_line_iters', 50))
        self.eps = float(self.options.get('eps', 1e-12))

    @staticmethod
    def _proj_F(w):
        return np.maximum(w, 0.0)

    @staticmethod
    def _proj_tangent_cone_nonneg(x, v, zero_tol=1e-15):
        out = v.copy()
        mask = x <= zero_tol
        out[mask] = np.maximum(out[mask], 0.0)
        return out

    def _model_value(self, f0, g, w0, s, sigma, problem):
        Hv = problem.hvp(w0, s)
        return (f0 + float(np.dot(g, s))
                + 0.5 * float(np.dot(s, Hv))
                + (sigma / 3.0) * float(np.linalg.norm(s)) ** 3)

    def _generalized_cauchy_point(self, w, f0, g, sigma, problem):
        t = self.t0
        tmin, tmax = 0.0, np.inf
        minus_g = -g
        w_best = w.copy()
        s_best = np.zeros_like(w)
        mk_best = f0

        for _ in range(self.max_line_iters):
            w_trial = self._proj_F(w - t * g)
            s = w_trial - w
            s_norm = np.linalg.norm(s)

            if s_norm <= self.eps:
                w_best, s_best = w_trial, s
                mk_best = f0
                break

            mk_val = self._model_value(f0, g, w, s, sigma, problem)
            gs = float(np.dot(g, s))

            cond24 = mk_val <= f0 + self.kappa_ubs * gs
            cond25 = mk_val >= f0 + self.kappa_lbs * gs
            proj_t = self._proj_tangent_cone_nonneg(w_trial, minus_g)
            cond26 = np.linalg.norm(proj_t) <= self.kappa_epp * abs(gs)

            w_best, s_best, mk_best = w_trial, s, mk_val

            if not cond24:
                tmax = t
            elif (not cond25) and (not cond26):
                tmin = t
            else:
                break

            t = 2.0 * t if np.isinf(tmax) else 0.5 * (tmin + tmax)

        return w_best, s_best, mk_best

    def solve(self, problem, w0):
        print(f"\n--- Running {self.name} ---")
        start_time = time.time()

        w = w0.copy()
        self.history = []
        sigma = self.sigma
        eta1, eta2 = self.etas
        gamma_dec, gamma_inc = self.gammas

        for k in range(self.max_iters):
            f0 = float(problem.objective(w))
            self.history.append(f0)
            g = problem.gradient(w)

            pg = self._proj_tangent_cone_nonneg(w, -g)
            if np.linalg.norm(pg) <= self.tol:
                print(f"Converged at iter {k} (projected-grad norm <= {self.tol}).")
                break

            w_plus, s_plus, mk_plus = self._generalized_cauchy_point(
                w, f0, g, sigma, problem)

            pred_red = float(f0 - mk_plus)
            f_plus = float(problem.objective(w_plus))
            act_red = float(f0 - f_plus)

            if pred_red <= self.eps:
                rho = -np.inf
            else:
                rho = act_red / (pred_red + self.eps)

            if (k + 1) % 20 == 0 or k == 0:
                print(f"Iter {k+1:4} | Loss: {f0:.8f} | rho: {rho:.4f} "
                      f"| sigma: {sigma:.4f} | ||s||: {np.linalg.norm(s_plus):.3e}")

            if rho >= eta1:
                w = w_plus

            if rho >= eta2:
                sigma = max(gamma_dec * sigma, self.eps)
            elif rho < eta1:
                sigma = gamma_inc * sigma

        self.solve_time = time.time() - start_time
        print(f"--- {self.name} finished in {self.solve_time:.2f} seconds ---")


# ---------------------------------------------------------------------------
# Experiment runner (plots optimality gap)
# ---------------------------------------------------------------------------

class NMFExperimentRunner:
    def __init__(self, problem, optimal_loss=0.0):
        self.problem = problem
        self.optimal_loss = optimal_loss
        self.optimizers = []
        self.title = f"Convergence Comparison for NMF (MSE)"

    def add_optimizer(self, optimizer):
        self.optimizers.append(optimizer)

    def run_all(self):
        print(f"\n{'='*50}")
        print(f"=== Starting Experiment: {self.problem.get_problem_name()} ===")
        print(f"=== Z: ({self.problem.m}×{self.problem.n}), rank r={self.problem.r} ===")
        print(f"=== Initial loss : {self.problem.objective(self.problem.w0):.8f} ===")
        print(f"=== Optimal loss : {self.optimal_loss:.8f} ===")
        print(f"{'='*50}")

        for opt in self.optimizers:
            opt.solve(self.problem, self.problem.w0.copy())

    def plot_results(self, output_dir="figures"):
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 8))

        markers = ['o', 'x', 's', 'D', '^', 'v']
        linestyles = ['-', '--', ':', '-.', '-', '--']

        for i, opt in enumerate(self.optimizers):
            # Optimality gap; clamp to small positive to avoid log(0)
            gap = [max(v - self.optimal_loss, 1e-16) for v in opt.history]
            plt.plot(
                gap,
                label=opt.name,
                marker=markers[i % len(markers)],
                linestyle=linestyles[i % len(linestyles)],
                markersize=4,
            )

        plt.title(self.title, fontsize=20)
        plt.xlabel('Iteration', fontsize=16)
        plt.ylabel('Frobenius Error Optimality Gap', fontsize=16)
        plt.yscale('log')
        plt.legend(fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()

        fname = self.problem.get_problem_name().replace(" ", "_") + "_gap.png"
        filepath = os.path.join(output_dir, fname)
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"Figure saved to {filepath}")


# ---------------------------------------------------------------------------
# Data generation (Frobenius / MSE)
# ---------------------------------------------------------------------------

m, n, r = 100, 20, 10

print(f"Generating synthetic data: Z=({m}×{n}), rank r={r}")

np.random.seed(0)  # reproducible data

X_hat = np.random.rand(m, r)      # m×r, entries from U[0,1]
Y_hat = np.random.rand(r, n)      # r×n, entries from U[0,1]
M = X_hat @ Y_hat                  # rank-r nonnegative matrix

# Full SVD of M; shape: U (m,m), s (min(m,n),), Vt (n,n)
U_full, s_full, Vt_full = np.linalg.svd(M, full_matrices=True)

# M has rank r, so s_full[r:] ≈ 0.  Use the theoretical r-th singular value.
sigma_r = s_full[r - 1]           # smallest "true" singular value of M

# Sample tail singular values σ_{r+1},...,σ_{min(m,n)} ~ U[0, 0.1 σ_r]
k_tail = min(m, n) - r
sigma_tail = np.random.uniform(0.0, 0.1 * sigma_r, size=k_tail)

# Build the full singular-value vector for Z
sigma_Z = np.concatenate([s_full[:r], sigma_tail])

# Construct Z using the same singular vectors as M
# U_full[:, :min(m,n)] is (m, min(m,n)), Vt_full is (n,n)
Z_data = U_full[:, :min(m, n)] @ np.diag(sigma_Z) @ Vt_full

# Ground-truth optimal function value:  f* = (1/2mn) Σ_{i>r} σ_i^2
optimal_loss = 0.5 / (m * n) * float(np.sum(sigma_tail ** 2))

print(f"Optimal (ground-truth) loss f* = {optimal_loss:.10f}")
print(f"  (σ_r = {sigma_r:.6f}, tail range [0, {0.1*sigma_r:.6f}])\n")

# ---------------------------------------------------------------------------
# Optimizers
# ---------------------------------------------------------------------------

max_iters = 200
common_options = {'maxiter': max_iters}

seed = 42
problem_fro = NMFProblemFroInstance(Z_data, r, seed=seed)

optimizers_to_run = [
    ScipyOptimizer(
        name="L-BFGS-B (scipy)",
        method="L-BFGS-B",
        options=common_options,
    ),
    ScipyOptimizer(
        name="trust-constr (scipy)",
        method="trust-constr",
        options=common_options,
    ),
    CustomARAOptimizer(
        name="ARA",
        options={**common_options, 'kappa': 1.0, 'ell': 1.0},
    ),
    CustomCOCARCOptimizer(
        name="COCARC",
        options={**common_options, 'sigma': 1.0, 'tol': 1e-6},
    ),
]

# Optional: IPOPT (requires cyipopt)
try:
    import cyipopt  # noqa: F401
    _HAS_CYIPOPT = True
except ImportError:
    _HAS_CYIPOPT = False
    print("Note: cyipopt not installed → skipping IPOPT optimizer.")

if _HAS_CYIPOPT:
    optimizers_to_run.insert(
        2,
        IpoptOptimizer(
            name="IPOPT (cyipopt)",
            options={**common_options, 'print_level': 0, 'bound_relax_factor': 0.0},
        ),
    )

# ---------------------------------------------------------------------------
# Run experiment
# ---------------------------------------------------------------------------

experiment_fro = NMFExperimentRunner(problem_fro, optimal_loss=optimal_loss)
for opt in optimizers_to_run:
    experiment_fro.add_optimizer(opt)

experiment_fro.run_all()
experiment_fro.plot_results()
