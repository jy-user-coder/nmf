import numpy as np

class NMFProblemFro:
    """
    Defines the NMF problem with the Frobenius norm loss and provides methods
    to compute the loss, its gradient, and Hessian-vector products.

    The loss function is L(X, Y) = 0.5 * ||Z - XY||_F^2.
    """
    def __init__(self, Z, r):
        """
        Initializes the NMF problem objective.

        Args:
            Z (np.ndarray): The target matrix of shape (m, n).
            r (int): The rank of the factorization.
        """
        self.Z = Z
        self.r = r
        self.m, self.n = Z.shape

    def _get_reconstruction(self, X, Y):
        """Computes the reconstructed matrix XY."""
        return X @ Y

    def frobenius_loss(self, w):
        """
        Computes the Frobenius norm loss for a given flattened state w.
        
        Args:
            w (np.ndarray): Flattened vector representing matrices X and Y.
        """
        X, Y = self._unflatten(w)
        E = self._get_reconstruction(X, Y) - self.Z
        # Using 0.5 * norm^2 is standard for least-squares problems
        return 0.5 * np.linalg.norm(E, 'fro')**2 / self.m / self.n

    def gradient(self, w):
        """
        Computes the analytic gradient with respect to the flattened state w.
        
        Args:
            w (np.ndarray): Flattened vector representing matrices X and Y.
        
        Returns:
            np.ndarray: The flattened gradient vector.
        """
        X, Y = self._unflatten(w)
        E = self._get_reconstruction(X, Y) - self.Z  # Error matrix E = XY - Z
        
        # Gradient w.r.t. X is E @ Y.T
        grad_X = E @ Y.T
        
        # Gradient w.r.t. Y is X.T @ E
        grad_Y = X.T @ E
        
        return self._flatten(grad_X, grad_Y) / self.m / self.n

    def hvp(self, w, p):
        """
        Computes the analytic Hessian-vector product.

        Args:
            w (np.ndarray): The point [X, Y] at which to evaluate the Hessian.
            p (np.ndarray): The vector [dX, dY] to multiply the Hessian by.

        Returns:
            np.ndarray: The resulting flattened Hessian-vector product.
        """
        X, Y = self._unflatten(w)
        dX, dY = self._unflatten(p) # The direction vector
        
        E = self._get_reconstruction(X, Y) - self.Z # Error matrix
        
        # HVP on X: (dX @ Y + X @ dY) @ Y.T + E @ dY.T
        hvp_X = dX @ Y @ Y.T + X @ dY @ Y.T + E @ dY.T
        
        # HVP on Y: X.T @ (X @ dY + dX @ Y) + dX.T @ E
        hvp_Y = X.T @ X @ dY + X.T @ dX @ Y + dX.T @ E
        
        return self._flatten(hvp_X, hvp_Y) / self.m / self.n

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
            np.ndarray: The full Hessian matrix.
        """
        X, Y = self._unflatten(w)
        
        # Compute the individual blocks of the Hessian
        H_XX = self._hessian_X(X, Y)
        H_YY = self._hessian_Y(X, Y)
        H_XY = self._hessian_XY(X, Y)
        
        # The Hessian is symmetric, so H_YX is the transpose of H_XY
        H_YX = H_XY.T
        
        # Assemble the full Hessian
        return np.block([[H_XX, H_XY], [H_YX, H_YY]])

    # Internal helper methods for Hessian components
    def _hessian_X(self, X, Y):
        """Computes the Hessian block w.r.t. X using Kronecker product."""
        # H_XX is block diagonal, with Y @ Y.T on each diagonal block.
        # This is equivalent to I_m kron (Y @ Y.T).
        return np.kron(np.eye(self.m), Y @ Y.T) / self.m / self.n

    def _hessian_Y(self, X, Y):
        """Computes the Hessian block w.r.t. Y using Kronecker product."""
        # H_YY is block diagonal, with X.T @ X on each diagonal block.
        # This is equivalent to (X.T @ X) kron I_n.
        return np.kron(X.T @ X, np.eye(self.n)) / self.m / self.n

    def _hessian_XY(self, X, Y):
        """
        Computes the cross-term Hessian d^2L/(dX dY) using einsum.
        The formula for the (i,k), (l,p) entry is X[i,l]*Y[k,p] + E[i,p]*delta_kl.
        """
        E = self._get_reconstruction(X, Y) - self.Z
        
        # Term 1: X[i,l]*Y[k,p]
        term1 = np.einsum('il,kp->iklp', X, Y)
        
        # Term 2: E[i,p]*delta_kl, where delta_kl is the Kronecker delta
        term2 = np.einsum('ip,kl->iklp', E, np.eye(self.r))
        
        # Reshape the 4D tensor into the final (m*r, r*n) matrix
        return (term1 + term2).reshape(self.m * self.r, self.r * self.n) / self.m / self.n

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

class Fro_base(NMFProblemFro):
    def __init__(self, Z: np.ndarray, r: int, epsilon: float = 1e-9):
        """
        Initializes the NMF problem objective.

        Args:
            Z (np.ndarray): The target matrix of shape (m, n).
            r (int): The rank of the factorization.
            epsilon (float): A small value to prevent log(0).
        """
        super().__init__(Z, r)
        self.epsilon = epsilon
    
    def F(self, w: np.ndarray, ell: float = 1.0) -> float:
        """
        Computes the objective function F(w) from the image.
        F(w) = ell * [ (||X||_F^2 + ||Y||_F^2 + 1)^2 + sum(log(X)) + sum(log(Y)) ]

        Args:
            w (np.ndarray): The flattened vector representing matrices X and Y.
            ell (float): A scaling factor.

        Returns:
            float: The scalar value of the objective function.
        """
        X, Y = self._unflatten(w)
        
        # Add epsilon for numerical stability, preventing log(0)
        X_eps = X + self.epsilon
        Y_eps = Y + self.epsilon
        
        # ||X||_F^2 + ||Y||_F^2 + 1
        norm_X_sq = np.sum(X**2)
        norm_Y_sq = np.sum(Y**2)
        S = norm_X_sq + norm_Y_sq + 1.0
        
        # Term 1: (||X||_F^2 + ||Y||_F^2 + 1)^2
        term1 = S**2
        
        # Term 2: sum(log(X)) + sum(log(Y))
        term2 = np.sum(np.log(X_eps)) + np.sum(np.log(Y_eps))
        
        return ell * (term1 - term2)
        
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

        # S = ||X||_F^2 + ||Y||_F^2 + 1
        norm_X_sq = np.sum(X**2)
        norm_Y_sq = np.sum(Y**2)
        S = norm_X_sq + norm_Y_sq + 1.0
        
        # --- Gradient of Term 1: S^2 ---
        # d(S^2)/dX = 2*S * (dS/dX) = 2*S * (2*X) = 4*S*X
        grad_t1_X = 4.0 * S * X
        grad_t1_Y = 4.0 * S * Y
        
        # --- Gradient of Term 2: sum(log(X)) + sum(log(Y)) ---
        # d(sum(log(X)))/dX = 1/X (element-wise)
        grad_t2_X = 1.0 / X_eps
        grad_t2_Y = 1.0 / Y_eps
        
        # Total gradient
        grad_X = ell * (grad_t1_X - grad_t2_X)
        grad_Y = ell * (grad_t1_Y - grad_t2_Y)
        
        return self._flatten(grad_X, grad_Y)

    def hessian(self, w: np.ndarray, ell: float = 1.0) -> np.ndarray:
        """
        Computes the full analytic Hessian of F(w) with respect to w.
        
        The Hessian is structured as a block matrix:
        [[H_XX, H_XY],
         [H_YX, H_YY]]

        Args:
            w (np.ndarray): The flattened vector representing matrices X and Y.
            ell (float): A scaling factor.

        Returns:
            np.ndarray: The full Hessian matrix.
        """
        X, Y = self._unflatten(w)
        
        # Add epsilon to avoid division by zero
        X_eps = X + self.epsilon
        Y_eps = Y + self.epsilon
        
        # Get dimensions
        mr = self.m * self.r
        rn = self.r * self.n
        
        # S = ||X||_F^2 + ||Y||_F^2 + 1
        norm_X_sq = np.sum(X**2)
        norm_Y_sq = np.sum(Y**2)
        S = norm_X_sq + norm_Y_sq + 1.0
        
        # Flatten X and Y for outer products
        x_flat = X.ravel()
        y_flat = Y.ravel()

        # --- Hessian of Term 2: sum(log(X)) + sum(log(Y)) ---
        # H_XX(T2) = diag(-1 / X_ik^2)
        diag_H_XX_T2 = -1.0 / (X_eps**2)
        H_XX_T2 = np.diag(diag_H_XX_T2.ravel())
        
        # H_YY(T2) = diag(-1 / Y_kj^2)
        diag_H_YY_T2 = -1.0 / (Y_eps**2)
        H_YY_T2 = np.diag(diag_H_YY_T2.ravel())
        
        # Cross-terms are zero
        H_XY_T2 = np.zeros((mr, rn))
        H_YX_T2 = np.zeros((rn, mr))

        # --- Hessian of Term 1: S^2 ---
        # H_XX(T1) = d(4*S*X)/dX = 4 * ( (dS/dX) * X.T + S * I )
        # dS/dX is 2*X. So d(4*S*X)/dX_vec = 4 * ( 2*vec(X) * vec(X).T + S * I )
        # H_XX(T1) = 8 * vec(X)vec(X)^T + 4*S*I
        H_XX_T1 = 8.0 * np.outer(x_flat, x_flat) + 4.0 * S * np.eye(mr)
        
        # H_YY(T1) = 8 * vec(Y)vec(Y)^T + 4*S*I
        H_YY_T1 = 8.0 * np.outer(y_flat, y_flat) + 4.0 * S * np.eye(rn)
        
        # H_XY(T1) = d(4*S*X)/dY = 4 * (dS/dY) * X.T
        # dS/dY is 2*Y. So d(4*S*X)/dY_vec = 4 * (2*vec(Y)) * vec(X).T
        # H_XY(T1) = 8 * vec(X)vec(Y)^T
        H_XY_T1 = 8.0 * np.outer(x_flat, y_flat)
        
        # H_YX(T1) = H_XY(T1).T
        H_YX_T1 = H_XY_T1.T
        
        # --- Total Hessian = ell * (H(T1) + H(T2)) ---
        H_XX = ell * (H_XX_T1 - H_XX_T2)
        H_YY = ell * (H_YY_T1 - H_YY_T2)
        H_XY = ell * (H_XY_T1 - H_XY_T2) # H_XY_T2 is zero
        H_YX = ell * (H_YX_T1 - H_YX_T2) # H_YX_T2 is zero
        
        # Assemble the full Hessian
        return np.block([[H_XX, H_XY], [H_YX, H_YY]])
    
class NMFProblemFroInstance(NMFProblemFro):
    """Adapter class for the Frobenius Problem"""
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
        return self.frobenius_loss(w) # Call original method
    
    def gradient(self, w):
        return super().gradient(w) # Call original method
    
    def full_hessian(self, w):
        return super().full_hessian(w) # Call original method

    # Methods for the experiment runner
    def get_problem_name(self):
        return "Frobenius Norm (MSE)"
    
    def get_loss_name(self):
        return "MSE Loss"
    
    def get_regularizer_class(self):
        return Fro_base