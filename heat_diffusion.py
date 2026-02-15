"""Heat diffusion and heat method implementation for triangulated surfaces."""

import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import Optional, Union, List
import polyscope as ps

# Handle both relative and absolute imports
try:
    from .laplace_beltrami import LaplaceBeltrami
    from .utils import MeshUtils
except ImportError:
    from laplace_beltrami import LaplaceBeltrami
    from utils import MeshUtils


class HeatDiffusion:
    """
    Heat diffusion on triangulated surfaces.

    Solves the heat equation:
        ∂u/∂t = Δu

    Using implicit Euler integration:
        (M - tL) u_{n+1} = M u_n

    where L is the Laplace-Beltrami operator and M is the mass matrix.
    """

    def __init__(self, mesh: trimesh.Trimesh):
        """
        Initialize the heat diffusion solver.

        Args:
            mesh: Input triangular mesh
        """
        self.mesh = mesh
        self.lb = LaplaceBeltrami(mesh)
        self.n_vertices = len(mesh.vertices)

        # Get operators
        self.L = self.lb.get_laplacian()
        self.M = self.lb.get_mass_matrix()

    def diffuse_from_point(
        self,
        vertex_idx: int,
        time: float,
        timestep: Optional[float] = None,
        return_history: bool = False
    ) -> Union[np.ndarray, tuple]:
        """
        Simulate heat diffusion from a single point source.

        Args:
            vertex_idx: Index of the source vertex
            time: Total diffusion time
            timestep: Time step for integration (default: auto-computed)
            return_history: If True, return history of all time steps

        Returns:
            Heat distribution at final time, or (final, history) if return_history=True
        """
        # Auto-compute timestep if not provided
        if timestep is None:
            mean_edge_length = MeshUtils.compute_mean_edge_length(self.mesh)
            timestep = mean_edge_length ** 2  # Heuristic from heat method paper

        # Initialize with delta function at source
        u = np.zeros(self.n_vertices)
        u[vertex_idx] = 1.0 / self.lb.vertex_areas[vertex_idx]  # Point source

        # Number of time steps
        n_steps = int(np.ceil(time / timestep))
        actual_timestep = time / n_steps

        # Pre-factor the system matrix
        A = self.M - actual_timestep * self.L

        # Store history if requested
        history = [u.copy()] if return_history else None

        # Time integration
        for _ in range(n_steps):
            b = self.M @ u
            u = spsolve(A, b)

            if return_history:
                history.append(u.copy())

        if return_history:
            return u, np.array(history)
        return u

    def diffuse_from_multiple_points(
        self,
        vertex_indices: List[int],
        time: float,
        timestep: Optional[float] = None,
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Simulate heat diffusion from multiple point sources.

        Args:
            vertex_indices: List of source vertex indices
            time: Total diffusion time
            timestep: Time step for integration
            weights: Optional weights for each source (default: equal weights)

        Returns:
            Heat distribution at final time
        """
        if weights is None:
            weights = np.ones(len(vertex_indices))
        weights = weights / np.sum(weights)  # Normalize

        # Initialize with multiple sources
        u = np.zeros(self.n_vertices)
        for idx, weight in zip(vertex_indices, weights):
            u[idx] += weight / self.lb.vertex_areas[idx]

        # Auto-compute timestep if not provided
        if timestep is None:
            mean_edge_length = MeshUtils.compute_mean_edge_length(self.mesh)
            timestep = mean_edge_length ** 2

        # Number of time steps
        n_steps = int(np.ceil(time / timestep))
        actual_timestep = time / n_steps

        # Pre-factor the system matrix
        A = self.M - actual_timestep * self.L

        # Time integration
        for _ in range(n_steps):
            b = self.M @ u
            u = spsolve(A, b)

        return u

    def compute_heat_kernel(
        self,
        vertex_idx: int,
        time: float
    ) -> np.ndarray:
        """
        Compute the heat kernel from a point at a specific time.

        This is equivalent to diffuse_from_point but emphasizes the
        interpretation as a heat kernel.

        Args:
            vertex_idx: Source vertex index
            time: Time parameter

        Returns:
            Heat kernel values at all vertices
        """
        return self.diffuse_from_point(vertex_idx, time)

    def compute_geodesic_distances(
        self,
        vertex_idx: int,
        time: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute geodesic distances using the heat method.

        Algorithm:
        1. Solve heat equation for short time
        2. Compute gradient of heat distribution
        3. Normalize gradient to get unit vector field
        4. Solve Poisson equation to recover distances

        Args:
            vertex_idx: Source vertex index
            time: Heat diffusion time (default: auto-computed)

        Returns:
            Geodesic distances from source to all vertices
        """
        # Step 1: Diffuse heat for short time
        if time is None:
            mean_edge_length = MeshUtils.compute_mean_edge_length(self.mesh)
            time = mean_edge_length ** 2  # Short time

        u = self.diffuse_from_point(vertex_idx, time)

        # Step 2 & 3: Compute and normalize gradient (on faces)
        grad = self._compute_gradient(u)
        grad_normalized = grad / (np.linalg.norm(grad, axis=1, keepdims=True) + 1e-10)

        # Flip direction (heat flows away from source, we want distance to increase toward source)
        grad_normalized = -grad_normalized

        # Step 4: Solve Poisson equation: Δφ = ∇·X
        div = self._compute_divergence(grad_normalized)
        distances = spsolve(self.L, -div)

        # Shift so that source has distance 0
        distances = distances - distances[vertex_idx]

        return np.abs(distances)  # Ensure non-negative

    def _compute_gradient(self, vertex_function: np.ndarray) -> np.ndarray:
        """
        Compute gradient of a vertex function on each face.

        Args:
            vertex_function: Function values at vertices

        Returns:
            Gradient vectors on each face (n_faces, 3)
        """
        n_faces = len(self.mesh.faces)
        gradients = np.zeros((n_faces, 3))

        vertices = self.mesh.vertices
        faces = self.mesh.faces

        for i, face in enumerate(faces):
            i0, i1, i2 = face

            # Get vertex positions and function values
            v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]
            f0, f1, f2 = vertex_function[i0], vertex_function[i1], vertex_function[i2]

            # Compute edge vectors
            e1 = v1 - v0
            e2 = v2 - v0

            # Compute face normal
            n = np.cross(e1, e2)
            area = 0.5 * np.linalg.norm(n)
            n = n / (np.linalg.norm(n) + 1e-10)

            # Compute gradient using the formula:
            # ∇f = (f1-f0)(n × e2) + (f2-f0)(e1 × n) / (2*area)
            grad = ((f1 - f0) * np.cross(n, e2) + (f2 - f0) * np.cross(e1, n)) / (2 * area + 1e-10)
            gradients[i] = grad

        return gradients

    def _compute_divergence(self, face_vector_field: np.ndarray) -> np.ndarray:
        """
        Compute divergence of a vector field defined on faces.

        Args:
            face_vector_field: Vector field on faces (n_faces, 3)

        Returns:
            Divergence at each vertex (n_vertices,)
        """
        n_verts = self.n_vertices
        divergence = np.zeros(n_verts)

        vertices = self.mesh.vertices
        faces = self.mesh.faces

        for i, face in enumerate(faces):
            i0, i1, i2 = face
            X = face_vector_field[i]

            # Get vertex positions
            v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]

            # Compute edge vectors
            e0 = v2 - v1  # Opposite to vertex 0
            e1 = v0 - v2  # Opposite to vertex 1
            e2 = v1 - v0  # Opposite to vertex 2

            # Compute face normal and area
            n = np.cross(v1 - v0, v2 - v0)
            area = 0.5 * np.linalg.norm(n)

            # Compute cotangents
            cot0 = MeshUtils._compute_cotangent(v1 - v0, v2 - v0)
            cot1 = MeshUtils._compute_cotangent(v2 - v1, v0 - v1)
            cot2 = MeshUtils._compute_cotangent(v0 - v2, v1 - v2)

            # Accumulate divergence contributions
            divergence[i0] += 0.5 * (cot1 * np.dot(X, e1) + cot2 * np.dot(X, e2))
            divergence[i1] += 0.5 * (cot2 * np.dot(X, e2) + cot0 * np.dot(X, e0))
            divergence[i2] += 0.5 * (cot0 * np.dot(X, e0) + cot1 * np.dot(X, e1))

        # Normalize by vertex areas
        divergence /= (self.lb.vertex_areas + 1e-10)

        return divergence

    def visualize(
        self,
        heat_distribution: np.ndarray,
        title: str = "Heat Distribution",
        cmap: str = "plasma"
    ):
        """
        Visualize heat distribution on the mesh.

        Args:
            heat_distribution: Heat values at vertices
            title: Title for visualization
            cmap: Colormap name
        """
        ps.init()

        ps_mesh = ps.register_surface_mesh(
            "mesh",
            self.mesh.vertices,
            self.mesh.faces
        )

        ps_mesh.add_scalar_quantity(
            title,
            heat_distribution,
            defined_on='vertices',
            cmap=cmap,
            enabled=True
        )

        ps.show()

    def visualize_animation(
        self,
        history: np.ndarray,
        title: str = "Heat Diffusion",
        cmap: str = "plasma",
        fps: int = 10
    ):
        """
        Visualize heat diffusion over time as an animation.

        Note: This creates an interactive view where you can scrub through time.

        Args:
            history: Array of shape (n_timesteps, n_vertices)
            title: Title for visualization
            cmap: Colormap name
            fps: Frames per second (for reference)
        """
        ps.init()

        ps_mesh = ps.register_surface_mesh(
            "mesh",
            self.mesh.vertices,
            self.mesh.faces
        )

        # Add all timesteps as separate scalar quantities
        for i, timestep_data in enumerate(history):
            ps_mesh.add_scalar_quantity(
                f"{title}_t{i}",
                timestep_data,
                defined_on='vertices',
                cmap=cmap,
                enabled=(i == 0)
            )

        print(f"Animation with {len(history)} frames loaded.")
        print("Use the Polyscope GUI to switch between timesteps.")

        ps.show()
