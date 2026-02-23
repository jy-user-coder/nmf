import unittest
import numpy as np
import torch
from nmf import NMFProblemKL, KL_base, NMFProblemFro, Fro_base
from torch.autograd.functional import hessian

class TestNMFDerivatives(unittest.TestCase):
    """
    A test suite for verifying the analytic gradient and Hessian computations
    in the NMFProblemKL class using numerical methods.
    """

    def setUp(self):
        """Set up a common test environment for all test cases."""
        # Define problem dimensions
        self.m, self.n, self.r = 10, 8, 3
        
        # Create a random non-negative target matrix Z
        self.Z = np.random.rand(self.m, self.n) + 0.1
        
        # Instantiate the NMF problem
        self.nmf_problem = NMFProblemKL(self.Z, self.r)
        
        # Create random initial factor matrices X and Y
        X = np.random.rand(self.m, self.r) + 0.1
        Y = np.random.rand(self.r, self.n) + 0.1
        
        # Store the flattened state vector w
        self.w = self.nmf_problem._flatten(X, Y)
        
        # Set a tolerance for comparing floating point numbers
        self.rtol = 1e-12
        self.atol = 1e-12

    def _torch_i_divergence(self, X_torch, Y_torch, Z_torch, epsilon=1e-9):
        """
        Computes the I-divergence loss using PyTorch for autograd.
        This serves as the ground truth for our derivative calculations.
        """
        XY = torch.maximum(X_torch @ Y_torch, torch.tensor(epsilon))
        Z_temp = torch.maximum(Z_torch, torch.tensor(epsilon))
        loss = torch.mean(Z_temp * torch.log(Z_temp / XY) - Z_temp + XY)
        return loss

    def test_gradients(self):
        """
        Tests the full flattened analytic gradient against PyTorch's autograd.
        """
        X, Y = self.nmf_problem._unflatten(self.w)
        
        # --- PyTorch (Numerical) Gradient Calculation ---
        X_torch = torch.tensor(X, requires_grad=True, dtype=torch.float64)
        Y_torch = torch.tensor(Y, requires_grad=True, dtype=torch.float64)
        Z_torch = torch.tensor(self.Z, dtype=torch.float64)
        
        loss = self._torch_i_divergence(X_torch, Y_torch, Z_torch, self.nmf_problem.epsilon)
        loss.backward()
        
        grad_X_torch = X_torch.grad.numpy()
        grad_Y_torch = Y_torch.grad.numpy()
        grad_torch_flat = self.nmf_problem._flatten(grad_X_torch, grad_Y_torch)

        # --- Analytic Gradient Calculation ---
        grad_analytic_flat = self.nmf_problem.gradient(self.w)
        
        # --- Comparison ---
        np.testing.assert_allclose(
            grad_analytic_flat, grad_torch_flat, rtol=self.rtol, atol=self.atol,
            err_msg="Flattened gradient is incorrect."
        )
        print("\nGradient tests passed.")

    def test_hessian_taylor_expansion(self):
        """
        Tests the full analytic Hessian using the second-order Taylor expansion.
        Checks that: f(w+d) - f(w) - g'd is close to 0.5 * d'Hd
        """
        # Create a small random direction vector
        d_flat = np.random.randn(self.w.size) * 1e-5
        
        # --- Compute Taylor expansion terms ---
        loss1 = self.nmf_problem.i_divergence(self.w)
        loss2 = self.nmf_problem.i_divergence(self.w + d_flat)
        g_flat = self.nmf_problem.gradient(self.w)
        H_full = self.nmf_problem.full_hessian(self.w)
        
        # --- Verify Second-Order Term ---
        actual_diff = loss2 - loss1 - np.dot(g_flat, d_flat)
        expected_diff = 0.5 * d_flat.T @ H_full @ d_flat
        
        np.testing.assert_allclose(
            actual_diff, expected_diff, rtol=1e-4, # Tolerance is looser due to approximation
            err_msg="Full Hessian matrix does not match Taylor expansion."
        )
        print("Full Hessian test passed.")


    def test_hvp(self):
        """
        Tests the analytic Hessian-vector product (HVP) against PyTorch's autograd.
        """
        X, Y = self.nmf_problem._unflatten(self.w)
        
        # Create a random direction vector p
        p_flat = np.random.randn(self.w.size)
        dX, dY = self.nmf_problem._unflatten(p_flat)

        # --- PyTorch (Numerical) HVP Calculation ---
        X_torch = torch.tensor(X, requires_grad=True, dtype=torch.float64)
        Y_torch = torch.tensor(Y, requires_grad=True, dtype=torch.float64)
        Z_torch = torch.tensor(self.Z, dtype=torch.float64)
        dX_torch = torch.tensor(dX, dtype=torch.float64)
        dY_torch = torch.tensor(dY, dtype=torch.float64)

        loss = self._torch_i_divergence(X_torch, Y_torch, Z_torch, self.nmf_problem.epsilon)

        grad_X_torch, grad_Y_torch = torch.autograd.grad(loss, (X_torch, Y_torch), create_graph=True)
        grad_dot_d = (grad_X_torch * dX_torch).sum() + (grad_Y_torch * dY_torch).sum()
        hvp_X_torch, hvp_Y_torch = torch.autograd.grad(grad_dot_d, (X_torch, Y_torch))
        
        hvp_torch_flat = self.nmf_problem._flatten(hvp_X_torch.numpy(), hvp_Y_torch.numpy())

        # --- Analytic HVP Calculation ---
        hvp_analytic_flat = self.nmf_problem.hvp(self.w, p_flat)

        # --- Comparison ---
        np.testing.assert_allclose(
            hvp_analytic_flat, hvp_torch_flat, rtol=self.rtol, atol=self.atol,
            err_msg="Hessian-vector product is incorrect."
        )
        print("Hessian-vector product test passed.")


class TestKLBaseDerivatives(unittest.TestCase):
    """
    A test suite for verifying the analytic gradient and Hessian computations
    in the KL_base class using numerical methods.
    """

    def setUp(self):
        """Set up a common test environment for all test cases."""
        self.m, self.n, self.r = 12, 10, 4
        self.Z = np.random.rand(self.m, self.n) + 0.1
        self.kl_problem = KL_base(self.Z, self.r)
        
        X = np.random.rand(self.m, self.r)
        Y = np.random.rand(self.r, self.n)
        self.w = self.kl_problem._flatten(X, Y)
        
        self.rtol = 1e-8
        self.atol = 1e-8

    def _torch_F(self, X_torch, Y_torch, Z_torch, ell=1.0, epsilon=1e-9):
        """
        Computes the F(X, Y) objective function using PyTorch for autograd.
        """
        logX = torch.log(X_torch + epsilon)
        logY = torch.log(Y_torch + epsilon)
        
        ones_for_Y = torch.ones((self.r, self.n), dtype=X_torch.dtype, device=X_torch.device)
        ones_for_X = torch.ones((self.m, self.r), dtype=X_torch.dtype, device=X_torch.device)
        
        inner_sum = torch.matmul(logX, ones_for_Y) + torch.matmul(ones_for_X, logY)
        
        return -ell * torch.sum(Z_torch * inner_sum) / (self.m * self.n)

    def test_gradient_with_torch(self):
        """
        Tests the analytic flattened gradient of F against PyTorch's autograd.
        """
        X, Y = self.kl_problem._unflatten(self.w)
        
        # --- PyTorch (Numerical) Gradient Calculation ---
        X_torch = torch.tensor(X, requires_grad=True, dtype=torch.float64)
        Y_torch = torch.tensor(Y, requires_grad=True, dtype=torch.float64)
        Z_torch = torch.tensor(self.Z, dtype=torch.float64)
        
        loss = self._torch_F(X_torch, Y_torch, Z_torch, epsilon=self.kl_problem.epsilon)
        loss.backward()
        
        grad_X_torch = X_torch.grad.numpy()
        grad_Y_torch = Y_torch.grad.numpy()
        grad_torch_flat = self.kl_problem._flatten(grad_X_torch, grad_Y_torch)

        # --- Analytic Gradient Calculation ---
        grad_analytic_flat = self.kl_problem.gradient(self.w)
        
        # --- Comparison ---
        np.testing.assert_allclose(
            grad_analytic_flat, grad_torch_flat, rtol=self.rtol, atol=self.atol,
            err_msg="Gradient of F is incorrect."
        )
        print("\nGradient tests for F passed (compared with PyTorch).")

    def test_hessian_with_taylor_expansion(self):
        """
        Tests the analytic diagonal Hessian of F using second-order Taylor expansion.
        """
        # Create a small random direction vector
        d_flat = np.random.randn(self.w.size) * 1e-6
        
        # --- Compute Taylor expansion terms ---
        loss1 = self.kl_problem.F(self.w)
        loss2 = self.kl_problem.F(self.w + d_flat)
        g_flat = self.kl_problem.gradient(self.w)
        diag_H_full = self.kl_problem.hessian(self.w)
        
        # --- Verify Second-Order Term ---
        actual_diff = loss2 - loss1 - np.dot(g_flat, d_flat)
        expected_diff = 0.5 * np.sum(diag_H_full * (d_flat**2))
        
        np.testing.assert_allclose(
            actual_diff, expected_diff, rtol=1e-3,
            err_msg="Diagonal Hessian of F does not match Taylor expansion."
        )
        print("Hessian tests for F passed (compared with Taylor expansion).")

class TestNMFProblemFroDerivatives(unittest.TestCase):
    """
    Test suite for verifying the analytic derivatives of the NMFProblemFro class
    against PyTorch's automatic differentiation.
    """

    def setUp(self):
        """Set up the test fixture before every test method."""
        # Use small dimensions for the full Hessian check to keep it fast
        self.m, self.n, self.r = 5, 4, 3
        self.dtype = np.float64  # Use float64 for better numerical precision

        # Set seeds for reproducibility of random data
        np.random.seed(42)
        torch.manual_seed(42)

        # Create random problem data
        self.Z_np = np.random.rand(self.m, self.n).astype(self.dtype)
        X_np = np.random.rand(self.m, self.r).astype(self.dtype)
        Y_np = np.random.rand(self.r, self.n).astype(self.dtype)

        # Create the flattened vector 'w' which represents the parameters
        self.w_np = NMFProblemFro._flatten(X_np, Y_np)

        # Create a random direction vector 'p' for Hessian-vector product test
        self.p_np = np.random.rand(self.w_np.shape[0]).astype(self.dtype)

        # Instantiate the class to be tested
        self.nmf_problem = NMFProblemFro(self.Z_np, self.r)
        
        # Tolerances for floating point comparisons
        self.rtol = 1e-6
        self.atol = 1e-8

    def _torch_loss_func(self, w_vec):
        """A PyTorch-compatible version of the loss function for autograd."""
        Z_torch = torch.tensor(self.Z_np, dtype=torch.float64)

        # Unflatten the parameter vector 'w' into matrices X and Y
        X_size = self.m * self.r
        X_flat = w_vec[:X_size]
        Y_flat = w_vec[X_size:]
        X = X_flat.reshape(self.m, self.r)
        Y = Y_flat.reshape(self.r, self.n)

        # Calculate the reconstruction error and the loss
        E = X @ Y - Z_torch
        # Note: The scaling factor 1 / (m * n) must match the original class
        return 0.5 * torch.norm(E, 'fro')**2 / (self.m * self.n)

    def test_gradient(self):
        """Verify the analytic gradient against PyTorch's autograd."""
        # 1. Calculate analytic gradient using the method from NMFProblemFro
        grad_analytic = self.nmf_problem.gradient(self.w_np)

        # 2. Calculate gradient with PyTorch's automatic differentiation
        w_torch = torch.tensor(self.w_np, dtype=torch.float64, requires_grad=True)
        loss = self._torch_loss_func(w_torch)
        loss.backward()
        grad_torch = w_torch.grad.numpy()

        # 3. Assert that the two gradients are close enough
        self.assertTrue(
            np.allclose(grad_analytic, grad_torch, rtol=self.rtol, atol=self.atol),
            msg="Gradient calculation does not match PyTorch autograd."
        )

    def test_hvp(self):
        """Verify the analytic Hessian-vector product against PyTorch."""
        # 1. Calculate analytic HVP using the method from NMFProblemFro
        hvp_analytic = self.nmf_problem.hvp(self.w_np, self.p_np)

        # 2. Calculate HVP with PyTorch (using the "grad-of-grad" trick)
        w_torch = torch.tensor(self.w_np, dtype=torch.float64, requires_grad=True)
        p_torch = torch.tensor(self.p_np, dtype=torch.float64)

        loss = self._torch_loss_func(w_torch)
        (grad_w,) = torch.autograd.grad(loss, w_torch, create_graph=True)
        (hvp_torch_vec,) = torch.autograd.grad(grad_w, w_torch, grad_outputs=p_torch)
        hvp_torch = hvp_torch_vec.numpy()

        # 3. Assert that the two HVPs are close enough
        self.assertTrue(
            np.allclose(hvp_analytic, hvp_torch, rtol=self.rtol, atol=self.atol),
            msg="Hessian-vector product does not match PyTorch autograd."
        )

    def test_full_hessian(self):
        """Verify the full analytic Hessian against PyTorch."""
        # 1. Calculate analytic Hessian using the method from NMFProblemFro
        hess_analytic = self.nmf_problem.full_hessian(self.w_np)
        
        # 2. Calculate Hessian with PyTorch's functional hessian utility
        w_torch = torch.tensor(self.w_np, dtype=torch.float64, requires_grad=True)
        hess_torch = hessian(self._torch_loss_func, w_torch).numpy()
        
        # 3. Assert that the two Hessian matrices are close enough
        self.assertTrue(
            np.allclose(hess_analytic, hess_torch, rtol=self.rtol, atol=self.atol),
            msg="Full Hessian calculation does not match PyTorch autograd."
        )

class TestFroBaseDerivatives(unittest.TestCase):
    """
    Test suite for verifying the analytic derivatives of the Fro_base class
    against PyTorch's automatic differentiation.
    """

    def setUp(self):
        """Set up the test fixture before every test method."""
        # Use small dimensions for the full Hessian check to keep it fast
        self.m, self.n, self.r = 5, 4, 3
        self.dtype = np.float64  # Use float64 for better numerical precision

        # Set seeds for reproducibility
        np.random.seed(43) # Use a different seed than other tests
        torch.manual_seed(43)

        # Z is needed for parent __init__, but not for F()
        self.Z_np = np.random.rand(self.m, self.n).astype(self.dtype)
        
        # X and Y must be > 0 for log()
        X_np = np.random.rand(self.m, self.r).astype(self.dtype) + 0.1 
        Y_np = np.random.rand(self.r, self.n).astype(self.dtype) + 0.1

        # Instantiate the class to be tested
        self.fro_base_problem = Fro_base(self.Z_np, self.r)
        
        # Create the flattened vector 'w'
        self.w_np = self.fro_base_problem._flatten(X_np, Y_np)
        
        # Test with a non-unity 'ell'
        self.ell = 1.5 

        # Tolerances for floating point comparisons
        self.rtol = 1e-6
        self.atol = 1e-8

    def _torch_F(self, w_vec):
        """
        A PyTorch-compatible version of the F(w) objective function
        for autograd.
        """
        
        # Get problem dimensions
        m, n, r = self.m, self.n, self.r
        epsilon = self.fro_base_problem.epsilon

        # Unflatten the parameter vector 'w' into matrices X and Y
        X_size = m * r
        X_flat = w_vec[:X_size]
        Y_flat = w_vec[X_size:]
        X_torch = X_flat.reshape(m, r)
        Y_torch = Y_flat.reshape(r, n)

        # Add epsilon for log stability
        X_eps = X_torch + epsilon
        Y_eps = Y_torch + epsilon
        
        # Term 1: (||X||_F^2 + ||Y||_F^2 + 1)^2
        norm_X_sq = torch.sum(X_torch**2)
        norm_Y_sq = torch.sum(Y_torch**2)
        S = norm_X_sq + norm_Y_sq + 1.0
        term1 = S**2
        
        # Term 2: sum(log(X)) + sum(log(Y))
        term2 = torch.sum(torch.log(X_eps)) + torch.sum(torch.log(Y_eps))
        
        # Return the final scaled objective value
        return self.ell * (term1 - term2)

    def test_gradient(self):
        """Verify the analytic gradient against PyTorch's autograd."""
        
        # 1. Calculate analytic gradient using the method from Fro_base
        grad_analytic = self.fro_base_problem.gradient(self.w_np, self.ell)

        # 2. Calculate gradient with PyTorch's automatic differentiation
        w_torch = torch.tensor(self.w_np, dtype=torch.float64, requires_grad=True)
        loss = self._torch_F(w_torch)
        loss.backward()
        grad_torch = w_torch.grad.numpy()

        # 3. Assert that the two gradients are close enough
        np.testing.assert_allclose(
            grad_analytic, grad_torch, rtol=self.rtol, atol=self.atol,
            err_msg="Fro_base gradient calculation does not match PyTorch."
        )
        print("\nFro_base gradient test passed.")

    def test_full_hessian(self):
        """Verify the full analytic Hessian against PyTorch."""
        
        # 1. Calculate analytic Hessian using the method from Fro_base
        hess_analytic = self.fro_base_problem.hessian(self.w_np, self.ell)
        
        # 2. Calculate Hessian with PyTorch's functional hessian utility
        w_torch = torch.tensor(self.w_np, dtype=torch.float64)
        # Note: hessian function needs a function that takes a tensor and returns a scalar
        hess_torch = hessian(self._torch_F, w_torch).numpy()
        
        # 3. Assert that the two Hessian matrices are close enough
        np.testing.assert_allclose(
            hess_analytic, hess_torch, rtol=self.rtol, atol=self.atol,
            err_msg="Fro_base full Hessian calculation does not match PyTorch."
        )
        print("Fro_base full Hessian test passed.")