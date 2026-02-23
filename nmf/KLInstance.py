import numpy as np

class NMFProblemKL:
    """
    A class that defines the NMF problem and provides methods to compute
    the I-divergence loss, its gradient, and Hessian-vector products.

    The problem is defined by a target matrix Z and a rank r. The main
    computation methods accept the factor matrices X and Y as a single
    flattened vector w.
    """
    def __init__(self, Z, r, epsilon=1e-9):
        """
        Initializes the NMF problem objective.

        Args:
            Z (np.ndarray): The target matrix of shape (m, n).
            r (int): The rank of the factorization.
            epsilon (float): A small value to prevent division by zero or log(0).
        """
        self.Z = Z
        self.r = r
        self.m, self.n = Z.shape
        self.epsilon = epsilon

    def _get_reconstruction(self, X, Y):
        """Computes the reconstructed matrix XY with a floor value."""
        return X @ Y + self.epsilon

    def i_divergence(self, w):
        """
        Computes the I-divergence loss for a given flattened state w.
        
        Args:
            w (np.ndarray): Flattened vector representing matrices X and Y.
        """
        X, Y = self._unflatten(w)
        XY = self._get_reconstruction(X, Y)
        Z_temp = self.Z
        return np.mean(Z_temp * np.log(Z_temp / XY) - Z_temp + XY)

    def gradient(self, w):
        """
        Computes the analytic gradient with respect to the flattened state w.
        
        Args:
            w (np.ndarray): Flattened vector representing matrices X and Y.
        
        Returns:
            np.ndarray: The flattened gradient vector.
        """
        X, Y = self._unflatten(w)
        XY = self._get_reconstruction(X, Y)
        R = self.Z / XY
        
        grad_X = (np.ones((self.m, self.n)) - R) @ Y.T
        grad_X /= (self.m * self.n)
        
        grad_Y = X.T @ (np.ones((self.m, self.n)) - R)
        grad_Y /= (self.m * self.n)
        
        return self._flatten(grad_X, grad_Y)

    def hvp(self, w, p):
        """
        Computes the analytic Hessian-vector product.

        Args:
            w (np.ndarray): The flattened vector [X, Y] at which to evaluate the Hessian.
            p (np.ndarray): The flattened vector [dX, dY] to multiply the Hessian by.

        Returns:
            np.ndarray: The resulting flattened Hessian-vector product.
        """
        X, Y = self._unflatten(w)
        dX, dY = self._unflatten(p)
        
        S = self._get_reconstruction(X, Y)
        d1 = 1.0 - self.Z / S
        d2 = self.Z / (S**2)
        dS = dX @ Y + X @ dY
        dd1 = d2 * dS
        
        hvp_X = (dd1 @ Y.T + d1 @ dY.T) / (self.m * self.n)
        hvp_Y = (X.T @ dd1 + dX.T @ d1) / (self.m * self.n)
        
        return self._flatten(hvp_X, hvp_Y)

    def full_hessian(self, w):
        """
        Computes and returns the full analytic Hessian matrix with respect to
        the flattened parameters w = [X, Y].

        The Hessian is structured as a block matrix:
        [[H_XX, H_XY],
         [H_YX, H_YY]]
        
        Args:
            w (np.ndarray): The flattened vector [X, Y] at which to evaluate.

        Returns:
            np.ndarray: The full Hessian matrix of shape 
                        ((m*r + r*n), (m*r + r*n)).
        """
        X, Y = self._unflatten(w)
        
        # Compute the individual blocks of the Hessian
        H_XX = self._hessian_X(X, Y)
        H_YY = self._hessian_Y(X, Y)
        H_XY = self._hessian_XY(X, Y)
        
        # The Hessian is symmetric, so H_YX is the transpose of H_XY
        H_YX = H_XY.T
        
        # Assemble the full Hessian using np.block
        H_full = np.block([[H_XX, H_XY], [H_YX, H_YY]])
        
        return H_full

    # Internal helper methods for Hessian components
    def _hessian_X(self, X, Y):
        """Computes the analytic Hessian with respect to X (flattened)."""
        XY = self._get_reconstruction(X, Y)
        Hessian_X = np.zeros((self.m * self.r, self.m * self.r))
        R_squared = self.Z / (XY**2)
        for i in range(self.m):
            for k in range(self.r):
                for p in range(self.r):
                    Hessian_X[i*self.r + k, i*self.r + p] = np.sum(R_squared[i, :] * Y[k, :] * Y[p, :])
        return Hessian_X / (self.m * self.n)

    def _hessian_Y(self, X, Y):
        """Computes the analytic Hessian with respect to Y (flattened)."""
        XY = self._get_reconstruction(X, Y)
        Hessian_Y = np.zeros((self.r * self.n, self.r * self.n))
        R_squared = self.Z / (XY**2)
        for j in range(self.n):
            for k in range(self.r):
                for p in range(self.r):
                    Hessian_Y[k*self.n + j, p*self.n + j] = np.sum(R_squared[:, j] * X[:, k] * X[:, p])
        return Hessian_Y / (self.m * self.n)

    def _hessian_XY(self, X, Y):
        """Computes the analytic cross-term Hessian d^2f/(dX dY)."""
        XY = self._get_reconstruction(X, Y)
        Hessian_XY = np.zeros((self.m * self.r, self.r * self.n))
        R = self.Z / XY
        R_squared = self.Z / (XY**2)

        for l in range(self.m):
            for p in range(self.r):
                for q in range(self.r):
                    for j_prime in range(self.n):
                        row_idx = l * self.r + p
                        col_idx = q * self.n + j_prime
                        term1 = (1 - R[l, j_prime]) * (1 if p == q else 0)
                        term2 = R_squared[l, j_prime] * Y[p, j_prime] * X[l, q]
                        Hessian_XY[row_idx, col_idx] = term1 + term2
        return Hessian_XY / (self.m * self.n)

    @staticmethod
    def _flatten(mat_X, mat_Y):
        """Flattens X and Y into a single vector."""
        return np.concatenate([mat_X.ravel(), mat_Y.ravel()])

    def _unflatten(self, vec):
        """Un-flattens a vector into X and Y matrices."""
        X_size = self.m * self.r
        X_flat = vec[:X_size]
        Y_flat = vec[X_size:]
        return X_flat.reshape(self.m, self.r), Y_flat.reshape(self.r, self.n)

class KL_base(NMFProblemKL):
    def __init__(self, Z: np.ndarray, r: int, epsilon: float = 1e-9):
        """
        Initializes the NMF problem objective.

        Args:
            Z (np.ndarray): The target matrix of shape (m, n).
            r (int): The rank of the factorization.
            epsilon (float): A small value to prevent division by zero or log(0).
        """
        super().__init__(Z, r, epsilon)
    
    def F(self, w: np.ndarray, ell: float = 1.0) -> float:
        """
        Computes the objective function F(w).

        Args:
            w (np.ndarray): The flattened vector representing matrices X and Y.
            ell (float): A scaling factor.

        Returns:
            float: The scalar value of the objective function.
        """
        X, Y = self._unflatten(w)
        
        # Add epsilon for numerical stability, preventing log(0)
        logX = np.log(X + self.epsilon)
        logY = np.log(Y + self.epsilon)
        
        ones_for_Y = np.ones((self.r, self.n))
        ones_for_X = np.ones((self.m, self.r))
        
        inner_sum = np.matmul(logX, ones_for_Y) + np.matmul(ones_for_X, logY)
        
        return -ell * np.sum(self.Z * inner_sum) / (self.m * self.n)
        
    def gradient(self, w: np.ndarray, ell: float = 1.0) -> np.ndarray:
        """
        Computes the gradient of F(w) with respect to w.

        Args:
            w (np.ndarray): The flattened vector representing matrices X and Y.
            ell (float): A scaling factor.

        Returns:
            np.ndarray: The flattened gradient vector.
        """
        X, Y = self._unflatten(w)
        
        # Add epsilon to avoid division by zero
        X_eps = X + self.epsilon
        Y_eps = Y + self.epsilon

        # Gradient with respect to X
        row_sums_Z = np.sum(self.Z, axis=1, keepdims=True)
        grad_X = -(ell / (self.m * self.n)) * (1 / X_eps) * row_sums_Z

        # Gradient with respect to Y
        col_sums_Z = np.sum(self.Z, axis=0, keepdims=True)
        grad_Y = -(ell / (self.m * self.n)) * (1 / Y_eps) * col_sums_Z
        
        return self._flatten(grad_X, grad_Y)

    def hessian(self, w: np.ndarray, ell: float = 1.0) -> np.ndarray:
        """
        Computes the diagonal elements of the Hessian of F(w).
        
        For this objective function, the Hessian is a diagonal matrix. This function
        efficiently computes and returns only the diagonal elements as a single
        flattened 1D array.

        Args:
            w (np.ndarray): The flattened vector representing matrices X and Y.
            ell (float): A scaling factor.

        Returns:
            np.ndarray: A 1D array containing the diagonal elements of the Hessian.
        """
        X, Y = self._unflatten(w)
        
        # Add epsilon to avoid division by zero
        X_eps = X + self.epsilon
        Y_eps = Y + self.epsilon
        
        # Hessian with respect to X is diagonal.
        # Its diagonal entries are: (ell / (m*n)) * (1 / X[a, b]^2) * sum_{j=1 to n} Z[a, j]
        row_sums_Z = np.sum(self.Z, axis=1, keepdims=True)
        diag_H_XX_matrix = (ell / (self.m * self.n)) * (1 / (X_eps**2)) * row_sums_Z

        # Hessian with respect to Y is diagonal.
        # Its diagonal entries are: (ell / (m*n)) * (1 / Y[c, d]^2) * sum_{i=1 to m} Z[i, d]
        col_sums_Z = np.sum(self.Z, axis=0, keepdims=True)
        diag_H_YY_matrix = (ell / (self.m * self.n)) * (1 / (Y_eps**2)) * col_sums_Z

        # Flatten the two matrices of diagonal elements into a single vector
        return self._flatten(diag_H_XX_matrix, diag_H_YY_matrix)

class NMFProblemKLInstance(NMFProblemKL):
    """Adapter class for the KL Problem"""
    def __init__(self, Z, r, seed=42):
        super().__init__(Z, r) # Call the original's __init__
        self.Z = Z
        self.r = r
        self.m, self.n = Z.shape
        self.seed = seed
        self.w0 = self._create_initial_guess()

    def _create_initial_guess(self):
        np.random.seed(self.seed)
        X0 = np.random.rand(self.m, self.r)
        Y0 = np.random.rand(self.r, self.n)
        return self._flatten(X0, Y0)

    # Standard interface methods
    def objective(self, w):
        return self.i_divergence(w) # Call original method
    
    def gradient(self, w):
        return super().gradient(w) # Call original method
    
    def full_hessian(self, w):
        return super().full_hessian(w) # Call original method

    # Methods for the experiment runner
    def get_problem_name(self):
        return "KL Divergence"
    
    def get_loss_name(self):
        return "KL-Divergence (Loss)"

    def get_regularizer_class(self):
        return KL_base