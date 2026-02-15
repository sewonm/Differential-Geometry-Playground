"""Laplace-Beltrami operator implementation for triangulated surfaces."""

import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.linalg import eigsh
from typing import Tuple, Optional
import polyscope as ps

# Handle both relative and absolute imports
try:
    from .utils import MeshUtils
except ImportError:
    from utils import MeshUtils


class LaplaceBeltrami:
    """
    Laplace-Beltrami operator on triangulated surfaces.

    The discrete Laplace-Beltrami operator uses the cotangent formula:
        L_{ij} = (cot α_{ij} + cot β_{ij}) for i ≠ j
        L_{ii} = -Σ_j L_{ij}

    The generalized eigenvalue problem is:
        L φ = λ M φ

    where M is the mass matrix (diagonal matrix of vertex areas).
    """

    def __init__(self, mesh: trimesh.Trimesh):
        """
        Initialize the Laplace-Beltrami operator.

        Args:
            mesh: Input triangular mesh
        """
        self.mesh = mesh
        self.n_vertices = len(mesh.vertices)

        # Compute geometric quantities
        self.vertex_areas = MeshUtils.compute_vertex_areas(mesh)
        self.L = None  # Laplacian matrix
        self.M = None  # Mass matrix
        self._build_operators()

    def _build_operators(self):
        """Build the Laplace-Beltrami operator and mass matrix."""
        # Compute cotangent weights
        rows, cols, weights = MeshUtils.compute_cotangent_weights(self.mesh)

        # Build sparse matrix
        L = sparse.coo_matrix(
            (weights, (rows, cols)),
            shape=(self.n_vertices, self.n_vertices)
        ).tocsr()

        # Make symmetric by averaging (handle numerical errors)
        L = 0.5 * (L + L.T)

        # Set diagonal entries: L_{ii} = -Σ_j L_{ij}
        L.setdiag(0)  # Clear diagonal first
        diagonal = -L.sum(axis=1).A1  # Sum of off-diagonal entries
        L.setdiag(diagonal)

        # Normalize by vertex areas (discrete metric)
        # This gives the actual Laplace-Beltrami operator
        area_inv = sparse.diags(1.0 / (self.vertex_areas + 1e-10))
        self.L = area_inv @ L

        # Build mass matrix (diagonal matrix of vertex areas)
        self.M = sparse.diags(self.vertex_areas)

    def compute_eigenpairs(self, k: int = 10, which: str = 'SM') -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the generalized eigenvalue problem: L φ = λ M φ

        Args:
            k: Number of eigenpairs to compute
            which: Which eigenvalues to compute ('SM' for smallest magnitude)

        Returns:
            Tuple of (eigenvalues, eigenvectors)
            - eigenvalues: Array of shape (k,)
            - eigenvectors: Array of shape (n_vertices, k)
        """
        # Solve generalized eigenvalue problem
        # Note: We use -L because eigsh finds largest eigenvalues by default
        # and we want smallest eigenvalues of L
        k = min(k, self.n_vertices - 2)  # Ensure k is valid

        eigenvalues, eigenvectors = eigsh(
            -self.L,
            k=k,
            M=self.M,
            which='SA',  # Smallest algebraic (most negative)
            sigma=None
        )

        # Negate eigenvalues back
        eigenvalues = -eigenvalues

        # Sort by eigenvalue magnitude
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        return eigenvalues, eigenvectors

    def visualize_eigenfunction(
        self,
        eigenfunction: np.ndarray,
        title: str = "Eigenfunction",
        cmap: str = "coolwarm"
    ):
        """
        Visualize an eigenfunction on the mesh using Polyscope.

        Args:
            eigenfunction: Function values at vertices (n_vertices,)
            title: Title for the visualization
            cmap: Colormap name
        """
        ps.init()

        # Register the mesh
        ps_mesh = ps.register_surface_mesh(
            "mesh",
            self.mesh.vertices,
            self.mesh.faces
        )

        # Add the eigenfunction as a scalar quantity
        ps_mesh.add_scalar_quantity(
            title,
            eigenfunction,
            defined_on='vertices',
            cmap=cmap,
            enabled=True
        )

        ps.show()

    def visualize_eigenfunctions_grid(
        self,
        eigenvectors: np.ndarray,
        n_show: Optional[int] = None,
        eigenvalues: Optional[np.ndarray] = None
    ):
        """
        Visualize multiple eigenfunctions in a grid.

        Args:
            eigenvectors: Array of eigenvectors (n_vertices, k)
            n_show: Number of eigenfunctions to show (default: all)
            eigenvalues: Optional eigenvalues for titles
        """
        ps.init()

        if n_show is None:
            n_show = eigenvectors.shape[1]
        n_show = min(n_show, eigenvectors.shape[1])

        # Register the mesh
        ps_mesh = ps.register_surface_mesh(
            "mesh",
            self.mesh.vertices,
            self.mesh.faces
        )

        # Add each eigenfunction
        for i in range(n_show):
            if eigenvalues is not None:
                name = f"φ_{i} (λ={eigenvalues[i]:.4f})"
            else:
                name = f"φ_{i}"

            ps_mesh.add_scalar_quantity(
                name,
                eigenvectors[:, i],
                defined_on='vertices',
                cmap='coolwarm',
                enabled=(i == 0)  # Only enable the first one by default
            )

        ps.show()

    def apply_operator(self, function: np.ndarray) -> np.ndarray:
        """
        Apply the Laplace-Beltrami operator to a function.

        Args:
            function: Function values at vertices (n_vertices,)

        Returns:
            Result of L @ function
        """
        return self.L @ function

    def compute_heat_kernel_signature(
        self,
        time: float,
        k_eigenfunctions: int = 100
    ) -> np.ndarray:
        """
        Compute the Heat Kernel Signature (HKS) at a given time.

        HKS(x, t) = Σ_i e^(-λ_i * t) * φ_i(x)^2

        Args:
            time: Time parameter
            k_eigenfunctions: Number of eigenfunctions to use

        Returns:
            Heat kernel signature values at each vertex
        """
        eigenvalues, eigenvectors = self.compute_eigenpairs(k=k_eigenfunctions)

        # Compute HKS
        exp_lambda = np.exp(-eigenvalues * time)
        hks = np.sum(exp_lambda * eigenvectors ** 2, axis=1)

        return hks

    def get_laplacian(self) -> sparse.csr_matrix:
        """Get the Laplace-Beltrami operator matrix."""
        return self.L

    def get_mass_matrix(self) -> sparse.csr_matrix:
        """Get the mass matrix."""
        return self.M

    def compute_laplacian_spectrum(self, k: int = 50) -> np.ndarray:
        """
        Compute the Laplacian spectrum (eigenvalues only).

        Args:
            k: Number of eigenvalues to compute

        Returns:
            Array of eigenvalues
        """
        eigenvalues, _ = self.compute_eigenpairs(k=k)
        return eigenvalues

    def smooth_function(
        self,
        function: np.ndarray,
        iterations: int = 1,
        timestep: float = 0.01
    ) -> np.ndarray:
        """
        Smooth a function using implicit Laplacian smoothing.

        Solves: (M - τL) u_{t+1} = M u_t

        Args:
            function: Input function values
            iterations: Number of smoothing iterations
            timestep: Time step for each iteration

        Returns:
            Smoothed function
        """
        from scipy.sparse.linalg import spsolve

        result = function.copy()

        for _ in range(iterations):
            # Implicit smoothing: (M - τL) u_{t+1} = M u_t
            A = self.M - timestep * self.L
            b = self.M @ result
            result = spsolve(A, b)

        return result
