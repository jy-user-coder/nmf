from nmf import NMFProblemKLInstance
import numpy as np
import scipy
from scipy.optimize import minimize, Bounds
import time
import os
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

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
        """
        Solves the given problem starting from w0.
        Must populate self.history with the loss at each iteration.
        """
        pass

    def __str__(self):
        return f"{self.name} (Time: {self.solve_time:.2f}s)"

class ScipyOptimizer(BaseOptimizer):
    """Wrapper for scipy.optimize.minimize methods."""
    def __init__(self, name, method, options=None):
        super().__init__(name, options)
        self.method = method
        self.result = None
        self._problem = None # To store problem for callback

    def _callback(self, xk, *args):
        """Internal callback to store history."""
        loss = self._problem.objective(xk)
        self.history.append(loss)

    def solve(self, problem, w0):
        self._problem = problem # Store problem for callback
        self.history = [problem.objective(w0)] # Initial loss
        
        num_vars = w0.size
        bounds = Bounds(lb=np.zeros(num_vars), ub=np.inf)
        
        # Build arguments for minimize
        kwargs = {
            'fun': problem.objective,
            'x0': w0,
            'method': self.method,
            'jac': problem.gradient,
            'bounds': bounds,
            'callback': self._callback,
            'options': self.options
        }
        
        # Add hessian if method supports it (like trust-constr)
        if self.method == 'trust-constr':
            kwargs['hess'] = problem.full_hessian
        
        print(f"\n--- Running {self.name} ---")
        start_time = time.time()
        self.result = minimize(**kwargs)
        self.solve_time = time.time() - start_time
        
        print(self.result.message)
        print(f"--- {self.name} finished in {self.solve_time:.2f} seconds ---")

class IpoptOptimizer(BaseOptimizer):
    """Wrapper for IPOPT (via cyipopt) with simple bound constraints.

    Notes
    -----
    - Requires `cyipopt` (a Python wrapper around Ipopt).
    - This benchmark uses only bound constraints w >= 0 (NMF nonnegativity).
    - By default, we use Ipopt's limited-memory Hessian approximation to avoid
      forming the full Hessian for large problems.

    Options (passed via `options` dict)
    ----------------------------------
    maxiter : int
        Maximum number of IPOPT iterations (mapped to Ipopt option 'max_iter').
    tol : float, optional
        Ipopt convergence tolerance (Ipopt option 'tol').
    print_level : int, optional
        Ipopt verbosity (Ipopt option 'print_level'). Default 0.
    hessian_approximation : str, optional
        Ipopt option 'hessian_approximation' (e.g., 'limited-memory').
        If not provided, defaults to 'limited-memory' unless use_hessian=True.
    bound_relax_factor : float, optional
        Ipopt option 'bound_relax_factor'. We default to 0.0 to prevent
        relaxed bounds from introducing negative values that may break the KL
        objective (log).
    use_hessian : bool, optional
        If True, provide a dense exact Hessian via `problem.full_hessian`.
        Warning: can be expensive for large n.
    """

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
                "Install with `pip install cyipopt` or `conda install -c conda-forge cyipopt ipopt`."
            ) from e

        n = int(w0.size)
        m = 0  # no general constraints, only bounds

        ipopt_inf = 2.0e19  # Ipopt's typical 'infinity' constant
        lb = np.zeros(n, dtype=float)
        ub = np.full(n, ipopt_inf, dtype=float)
        cl = np.zeros(m, dtype=float)
        cu = np.zeros(m, dtype=float)

        # --- IPOPT option handling ---
        max_iter = int(self.options.get('maxiter', self.max_iters))
        print_level = int(self.options.get('print_level', 0))
        tol = self.options.get('tol', None)

        use_hessian = bool(self.options.get('use_hessian', False))
        hess_approx = self.options.get(
            'hessian_approximation',
            'exact' if use_hessian else 'limited-memory'
        )

        # Ipopt may relax bounds slightly; this can cause issues for objectives with logs.
        bound_relax_factor = float(self.options.get('bound_relax_factor', 0.0))

        # History (start with the initial objective)
        self.history = [float(problem.objective(w0))]

        outer = self  # capture for callback

        class _NMFIpoptCallbacks:
            # --- Required callbacks ---
            def objective(self, x):
                # Safety: if Ipopt ever evaluates slightly outside bounds, project back.
                x_eval = np.maximum(np.asarray(x, dtype=float), 0.0)
                return float(problem.objective(x_eval))

            def gradient(self, x):
                x_eval = np.maximum(np.asarray(x, dtype=float), 0.0)
                return np.asarray(problem.gradient(x_eval), dtype=float)

            def constraints(self, x):
                return np.zeros(0, dtype=float)

            def jacobian(self, x):
                return np.zeros(0, dtype=float)

            # --- Optional callbacks ---
            def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                             d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
                # Record the objective per IPOPT iteration.
                val = float(obj_value)
                if len(outer.history) == 0 or abs(outer.history[-1] - val) > 1e-15:
                    outer.history.append(val)
                return True

            if use_hessian:
                def hessianstructure(self):
                    return np.tril_indices(n)

                def hessian(self, x, lagrange, obj_factor):
                    # Only objective term since m=0 (no constraints).
                    x_eval = np.maximum(np.asarray(x, dtype=float), 0.0)
                    H = obj_factor * problem.full_hessian(x_eval)
                    row, col = np.tril_indices(n)
                    return np.asarray(H[row, col], dtype=float)

        nlp = cyipopt.Problem(
            n=n,
            m=m,
            problem_obj=_NMFIpoptCallbacks(),
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu,
        )

        # Core options
        nlp.add_option('max_iter', max_iter)
        nlp.add_option('print_level', print_level)
        nlp.add_option('hessian_approximation', hess_approx)
        nlp.add_option('bound_relax_factor', bound_relax_factor)

        if tol is not None:
            nlp.add_option('tol', float(tol))

        # Pass through any other IPOPT options the user might have provided
        reserved = {'maxiter', 'print_level', 'tol', 'use_hessian', 'hessian_approximation', 'bound_relax_factor'}
        for k, v in self.options.items():
            if k in reserved:
                continue
            try:
                nlp.add_option(k, v)
            except Exception:
                # If an option isn't recognized or has wrong type, skip silently.
                pass

        # Solve
        x_opt, info = nlp.solve(w0)

        self.solution = x_opt
        self.info = info

        self.solve_time = time.time() - start_time
        msg = info.get('status_msg', 'IPOPT finished.')
        print(msg)
        print(f"--- {self.name} finished in {self.solve_time:.2f} seconds ---")


class CustomARAOptimizer(BaseOptimizer):
    """Implements the custom 'ARA' optimization algorithm."""
    def __init__(self, name, options=None):
        super().__init__(name, options)
        self.kappa = self.options.get('kappa', 1.0)
        self.ell = self.options.get('ell', 1.0)
        self.gammas = self.options.get('gammas', (0.5, 2.0, 2.0))
        self.etas = self.options.get('etas', (0.01, 0.9))

    def solve(self, problem, w0):
        print(f"\n--- Running {self.name} ---")
        start_time = time.time()
        
        # Dynamically get the correct regularizer from the problem
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
            
            # Use the problem's regularizer to get the Hessian
            reg_hess_diag = reg_problem.hessian(w, ell) # Assumes this returns 1D array
            if len(np.shape(reg_hess_diag)) == 1:
                H_sc = problem.full_hessian(w) + np.diag(reg_hess_diag)
            else:
                H_sc = problem.full_hessian(w) + reg_hess_diag

            flag = False
            rho = -1
            obj_new = np.inf
            w_new = w

            try:
                d = scipy.linalg.solve(H_sc, -g, assume_a='sym')
                lam = np.dot(d, -g)
                if lam >= 0:
                    w_new = w + d / (1 + kappa * np.sqrt(lam))
                    if np.sum(w_new < 0) > 0: # Enforce non-negativity
                        flag = True
                        obj_new_val = np.nan
                    else:
                        obj_new_val = problem.objective(w_new)
                    if not (np.isinf(obj_new_val) or np.isnan(obj_new_val)):
                        obj_new = obj_new_val
                        model_decrease = (kappa * lam - np.log(1 + kappa * lam)) / kappa**2
                        rho = (obj_old - obj_new) / (model_decrease + 1e-12)
                    else:
                        flag = True
            except (np.linalg.LinAlgError, ValueError):
                flag = True

            if (k + 1) % 20 == 0 or k == 0:
                print(f"Iter {k+1:4} | Loss: {obj_old:.8f} | Rho: {rho:.4f} | ell: {ell:.4f} | kappa: {kappa:.4f}")

            if not flag and rho >= eta1:
                w = w_new
            elif abs(obj_old - obj_new) < 1e-8:
                print(f"\nConverged at iteration {k+1} due to small change in objective.")
                break

            if not flag:
                if rho > eta2: ell = gamma1 * ell
                elif rho < eta1: ell = gamma2 * ell
            else:
                ell = gamma3 * ell
        
        self.solve_time = time.time() - start_time
        print(f"--- {self.name} finished in {self.solve_time:.2f} seconds ---")

class CustomARCOptimizerBound(BaseOptimizer):
    """Implements the custom 'ARC' (cubic regularization) algorithm."""
    def __init__(self, name, options=None):
        super().__init__(name, options)
        self.sigma = self.options.get('sigma', 1.0)
        self.gammas = self.options.get('gammas', (0.5, 2.0, 2.0))
        self.etas = self.options.get('etas', (0.01, 0.9))
        self.mu = self.options.get('mu', 1e-6)
        self.eps = self.options.get('eps', 1e-9)
        # Internal state for subproblem
        self._g = None
        self._H = None
        self._sigma_current = None

    # --- Helper methods for the cubic subproblem ---
    def _subproblem_m(self, dx):
        norm_dx = np.linalg.norm(dx)
        return np.dot(self._g, dx) + 0.5 * np.dot(dx, self._H @ dx) + (self._sigma_current / 3.0) * norm_dx**3

    def _subproblem_grad(self, dx):
        r = np.linalg.norm(dx)
        if r < 1e-9: return self._g + self._H @ dx
        return self._g + self._H @ dx + self._sigma_current * r * dx

    def _subproblem_hess(self, dx):
        r = np.linalg.norm(dx)
        n = len(dx)
        if r < 1e-9: return self._H
        dx_outer = np.outer(dx, dx)
        return self._H + self._sigma_current * (r * np.eye(n) + dx_outer / r)
    # --- End of helpers ---

    def solve(self, problem, w0):
        print(f"\n--- Running {self.name} ---")
        start_time = time.time()
        
        w = w0.copy()
        self.history = []
        
        self._sigma_current = self.sigma
        gamma1, gamma2, gamma3 = self.gammas
        eta1, eta2 = self.etas
        
        for k in range(self.max_iters):
            obj_old = problem.objective(w) - self.mu * np.sum(np.log(w + self.eps))
            true_obj = problem.objective(w)
            self.history.append(true_obj)
            
            self._g = problem.gradient(w) - self.mu / (w + self.eps)
            self._H = problem.full_hessian(w) + self.mu * np.diag(1/((w + self.eps)**2))
            
            flag = False
            rho = -1
            obj_new = np.inf
            w_new = w

            try:
                d_init = scipy.linalg.solve(self._H, -self._g, assume_a='sym')
                res = minimize(
                    self._subproblem_m, 
                    d_init, 
                    method="Newton-CG", 
                    jac=self._subproblem_grad, 
                    hess=self._subproblem_hess,
                    options={'maxiter': 50} # Inner loop limit
                )
                
                d = res.x
                w_new = w + d
                
                if np.sum(w_new < 0) > 0: # Enforce non-negativity
                    flag = True
                    obj_new_val = np.nan
                else:
                    obj_new_val = problem.objective(w_new) - self.mu * np.sum(np.log(w_new + self.eps))
                if not (np.isinf(obj_new_val) or np.isnan(obj_new_val)):
                    obj_new = obj_new_val
                    model_decrease = -res.fun
                    rho = (obj_old - obj_new) / (model_decrease + 1e-12)
                else:
                    flag = True
            except (np.linalg.LinAlgError, ValueError):
                flag = True

            if (k + 1) % 20 == 0 or k == 0:
                print(f"Iter {k+1:4} | Loss: {true_obj:.8f} | Rho: {rho:.4f} | sigma: {self._sigma_current:.4f}")

            if not flag and rho >= eta1:
                w = w_new
            elif abs(obj_old - obj_new) < 1e-8:
                print(f"\nConverged at iteration {k+1} due to small change in objective.")
                break

            if not flag:
                if rho > eta2: self._sigma_current = gamma1 * self._sigma_current
                elif rho < eta1: self._sigma_current = gamma2 * self._sigma_current
            else:
                self._sigma_current = gamma3 * self._sigma_current
        
        self.solve_time = time.time() - start_time
        print(f"--- {self.name} finished in {self.solve_time:.2f} seconds ---")

class CustomCOCARCOptimizer(BaseOptimizer):
    """Adaptive Regularization with Cubics for Convex Constraints (COCARC).

    This is a lightweight implementation of Algorithm 2.1 (COCARC) from:
    Cartis, Gould & Toint, *An adaptive cubic regularization algorithm for nonconvex optimization
    with convex constraints and its function-evaluation complexity*.

    Notes
    -----
    - Designed here for simple bound constraints w >= 0 (NMF nonnegativity), where the projection
      P_F is just elementwise clipping.
    - Uses the *generalized Cauchy point* as the trial step (Step 2 picks x_k^+ = x_k^{GC}),
      i.e. it satisfies the model decrease requirement (2.9) by construction.
    """

    def __init__(self, name, options=None):
        super().__init__(name, options)
        # ARC / regularization parameters
        self.sigma = float(self.options.get('sigma', 1.0))
        self.etas = self.options.get('etas', (0.01, 0.9))   # (eta1, eta2)
        self.gammas = self.options.get('gammas', (0.5, 2.0)) # (gamma_dec, gamma_inc)

        # Generalized Goldstein line-search parameters for the Cauchy arc (2.4)-(2.6)
        self.kappa_ubs = float(self.options.get('kappa_ubs', 0.1))
        self.kappa_lbs = float(self.options.get('kappa_lbs', 0.9))
        self.kappa_epp = float(self.options.get('kappa_epp', 0.25))
        self.t0 = float(self.options.get('t0', 1.0))

        # Termination / safeguards
        self.tol = float(self.options.get('tol', 1e-6))
        self.max_line_iters = int(self.options.get('max_line_iters', 50))
        self.eps = float(self.options.get('eps', 1e-12))

    # ---- Projection helpers for F = {w >= 0} ----
    @staticmethod
    def _proj_F(w):
        return np.maximum(w, 0.0)

    @staticmethod
    def _proj_tangent_cone_nonneg(x, v, zero_tol=1e-15):
        """Project v onto the tangent cone of the nonnegative orthant at x.

        Tangent cone for x >= 0:
          - if x_i > 0 : d_i free  -> projection is v_i
          - if x_i = 0 : d_i >= 0 -> projection is max(v_i, 0)
        """
        out = v.copy()
        mask = x <= zero_tol
        out[mask] = np.maximum(out[mask], 0.0)
        return out

    # ---- Cubic model evaluation mk(xk + s) ----
    def _model_value(self, f0, g, w0, s, sigma, problem):
        # Use Hessian-vector products to avoid forming full Hessians.
        Hv = problem.hvp(w0, s)
        quad = 0.5 * float(np.dot(s, Hv))
        lin = float(np.dot(g, s))
        r = float(np.linalg.norm(s))
        return f0 + lin + quad + (sigma / 3.0) * (r ** 3)

    def _generalized_cauchy_point(self, w, f0, g, sigma, problem):
        """Compute x_k^{GC} along the projected-gradient path via Algorithm 2.1 Step 1."""
        t = self.t0
        tmin, tmax = 0.0, np.inf

        # Precompute -g for (2.6)
        minus_g = -g

        w_best = w.copy()
        s_best = np.zeros_like(w)
        mk_best = f0

        for _ in range(self.max_line_iters):
            w_trial = self._proj_F(w - t * g)
            s = w_trial - w
            s_norm = np.linalg.norm(s)

            # If projection killed the step, we are at (approx) first-order stationarity
            if s_norm <= self.eps:
                w_best, s_best = w_trial, s
                mk_best = f0
                break

            mk_val = self._model_value(f0, g, w, s, sigma, problem)
            gs = float(np.dot(g, s))  # should be <= 0

            # Goldstein-like conditions (2.4)-(2.6)
            cond24 = mk_val <= f0 + self.kappa_ubs * gs
            cond25 = mk_val >= f0 + self.kappa_lbs * gs

            proj_t = self._proj_tangent_cone_nonneg(w_trial, minus_g)
            cond26 = np.linalg.norm(proj_t) <= self.kappa_epp * abs(gs)

            w_best, s_best, mk_best = w_trial, s, mk_val

            if not cond24:
                # Not enough decrease -> reduce t (set an upper bound)
                tmax = t
            elif (not cond25) and (not cond26):
                # Step too small (and arc not "ending") -> increase t (set a lower bound)
                tmin = t
            else:
                # Accept as generalized Cauchy point
                break

            # Update t as in Algorithm 2.1 Step 1.3
            if np.isinf(tmax):
                t *= 2.0
            else:
                t = 0.5 * (tmin + tmax)

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

            # Termination based on projected gradient norm (proxy for first-order criticality)
            pg = self._proj_tangent_cone_nonneg(w, -g)
            if np.linalg.norm(pg) <= self.tol:
                print(f"Converged at iter {k} (projected-grad norm <= {self.tol}).")
                break

            # Step 1: generalized Cauchy point along projected gradient arc
            w_plus, s_plus, mk_plus = self._generalized_cauchy_point(w, f0, g, sigma, problem)

            # Predicted / actual decrease and acceptance ratio (2.8)
            pred_red = float(f0 - mk_plus)
            f_plus = float(problem.objective(w_plus))
            act_red = float(f0 - f_plus)

            if pred_red <= self.eps:
                rho = -np.inf
            else:
                rho = act_red / (pred_red + self.eps)

            if (k + 1) % 20 == 0 or k == 0:
                print(f"Iter {k+1:4} | Loss: {f0:.8f} | rho: {rho:.4f} | sigma: {sigma:.4f} | ||s||: {np.linalg.norm(s_plus):.3e}")

            # Step 3: accept / reject
            if rho >= eta1:
                w = w_plus

            # Step 4: regularization parameter update (paper's logic, with fixed choices)
            if rho >= eta2:
                sigma = max(gamma_dec * sigma, self.eps)
            elif rho < eta1:
                sigma = gamma_inc * sigma
            # else: keep sigma unchanged

        self.solve_time = time.time() - start_time
        print(f"--- {self.name} finished in {self.solve_time:.2f} seconds ---")


class NMFExperimentRunner:
    def __init__(self, problem):
        self.problem = problem
        self.optimizers = []
        self.title = f"Convergence Comparison for NMF ({problem.get_problem_name()})"
        self.ylabel = problem.get_loss_name()

    def add_optimizer(self, optimizer):
        """Adds an optimizer instance to the experiment."""
        self.optimizers.append(optimizer)
        
    def run_all(self):
        """Runs all registered optimizers on the problem."""
        print(f"\n{'='*40}")
        print(f"=== Starting Experiment: {self.problem.get_problem_name()} ===")
        print(f"=== Problem: Z=({self.problem.m}x{self.problem.n}), Rank r={self.problem.r} ===")
        print(f"=== Initial Loss: {self.problem.objective(self.problem.w0):.8f} ===")
        print(f"{'='*40}")
        
        for opt in self.optimizers:
            # Pass a copy of w0 so each optimizer starts fresh
            opt.solve(self.problem, self.problem.w0.copy())
            
    def plot_results(self, output_dir="figures"):
        """Plots the convergence history of all optimizers and saves to output_dir."""
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 8))

        markers = ['o', 'x', 's', 'D']
        linestyles = ['-', '--', ':', '-.']

        for i, opt in enumerate(self.optimizers):
            plt.plot(
                opt.history,
                label=f"{opt.name}",
                marker=markers[i % len(markers)],
                linestyle=linestyles[i % len(linestyles)],
                markersize=4
            )

        plt.title(self.title, fontsize=20)
        plt.xlabel('Iteration', fontsize=16)
        plt.ylabel(self.ylabel, fontsize=16)
        plt.yscale('log')
        plt.legend(fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()

        problem_name = self.problem.get_problem_name().replace(" ", "_")
        filepath = os.path.join(output_dir, f"{problem_name}.png")
        plt.savefig(filepath, dpi=150)
        plt.close()
        print(f"Figure saved to {filepath}")

max_iters = 200
common_options = {'maxiter': max_iters}

optimizers_to_run = [
    ScipyOptimizer(
        name="L-BFGS-B (scipy)", 
        method="L-BFGS-B", 
        options=common_options
    ),
    ScipyOptimizer(
        name="trust-constr (scipy)",
        method="trust-constr",
        options=common_options
    ),
    CustomARAOptimizer(
        name="ARA",
        options={**common_options, 'kappa': 1.0, 'ell': 1.0}
    ),
    CustomCOCARCOptimizer(
        name="COCARC",
        options={**common_options, 'sigma': 1.0, 'tol': 1e-6}
    ),
    # CustomARCOptimizerBound(
    #     name=r"ARC $\mu=0$",
    #     options={**common_options, 'sigma': 1.0, 'mu': 0.0}
    # ),
    # CustomARCOptimizerBound(
    #     name=r"ARC $\mu=10^{-5}$",
    #     options={**common_options, 'sigma': 1.0, 'mu': 1e-5}
    # ),
    # CustomARCOptimizerBound(
    #     name=r"ARC $\mu=10^{-6}$",
    #     options={**common_options, 'sigma': 1.0, 'mu': 1e-6}
    # ),
    # CustomARCOptimizerBound(
    #     name=r"ARC $\mu=10^{-7}$",
    #     options={**common_options, 'sigma': 1.0, 'mu': 1e-7}
    # )
]

# --- Optional: IPOPT (requires cyipopt) ---
try:
    import cyipopt  # noqa: F401
    _HAS_CYIPOPT = True
except ImportError:
    _HAS_CYIPOPT = False
    print("Note: cyipopt not installed -> skipping IPOPT optimizer. "
          "Install with `pip install cyipopt` or `conda install -c conda-forge cyipopt ipopt`.")

if _HAS_CYIPOPT:
    # Insert after the two SciPy baselines
    optimizers_to_run.insert(
        2,
        IpoptOptimizer(
            name="IPOPT (cyipopt)",
            options={**common_options, 'print_level': 0, 'bound_relax_factor': 0.0}
        )
    )

# --- 3. SETUP SYNTHETIC DATA (KL divergence) ---
# Z = X̂Ŷ + 0.01Ẑ, where X̂ ∈ R^{m×r}, Ŷ ∈ R^{r×n}, Ẑ ∈ R^{m×n},
# entries i.i.d. from U[0,1].
m, n, r = 100, 20, 10
print(f"Creating a synthetic problem of size Z=({m}x{n}) with rank r={r}\n")

np.random.seed(0)
X_hat = np.random.rand(m, r)
Y_hat = np.random.rand(r, n)
Z_hat = np.random.rand(m, n)
Z_data = X_hat @ Y_hat + 0.01 * Z_hat

seed = 42
problem_kl = NMFProblemKLInstance(Z_data, r, seed=seed)
experiment_kl = NMFExperimentRunner(problem_kl)

for opt in optimizers_to_run:
    experiment_kl.add_optimizer(opt)

experiment_kl.run_all()
experiment_kl.plot_results()
