"""Spectral geometry tools including embeddings, clustering, and analysis."""

import numpy as np
import trimesh
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from typing import Tuple, Optional, List, Union
import polyscope as ps

# Handle both relative and absolute imports
try:
    from .laplace_beltrami import LaplaceBeltrami
    from .utils import MeshUtils
except ImportError:
    from laplace_beltrami import LaplaceBeltrami
    from utils import MeshUtils


class SpectralGeometry:
    """
    Spectral geometry operations on triangulated surfaces.

    Provides tools for:
    - Spectral embeddings
    - Spectral clustering
    - Mesh smoothing
    - Heat kernel signatures
    - Shape analysis
    """

    def __init__(self, mesh: trimesh.Trimesh):
        """
        Initialize spectral geometry tools.

        Args:
            mesh: Input triangular mesh
        """
        self.mesh = mesh
        self.lb = LaplaceBeltrami(mesh)
        self.n_vertices = len(mesh.vertices)

        # Cache for eigenpairs
        self._eigenvalues = None
        self._eigenvectors = None

    def compute_spectral_embedding(
        self,
        n_components: int = 3,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Compute spectral embedding of the mesh.

        Uses the first k non-trivial eigenfunctions as coordinates.

        Args:
            n_components: Number of dimensions for embedding
            normalize: Whether to normalize the embedding

        Returns:
            Embedding coordinates (n_vertices, n_components)
        """
        # Compute eigenpairs (skip first constant eigenfunction)
        eigenvalues, eigenvectors = self.lb.compute_eigenpairs(k=n_components + 1)

        # Skip the first (constant) eigenfunction
        embedding = eigenvectors[:, 1:n_components + 1]

        if normalize:
            # Weight by eigenvalues (optional)
            # This gives more importance to low-frequency components
            weights = 1.0 / (eigenvalues[1:n_components + 1] + 1e-10)
            embedding = embedding * weights

        return embedding

    def spectral_clustering(
        self,
        n_clusters: int,
        n_eigenfunctions: int = 10,
        return_centers: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Perform spectral clustering on the mesh.

        Args:
            n_clusters: Number of clusters
            n_eigenfunctions: Number of eigenfunctions to use
            return_centers: If True, return cluster centers

        Returns:
            Cluster labels for each vertex, or (labels, centers) if return_centers=True
        """
        # Get spectral embedding
        embedding = self.compute_spectral_embedding(
            n_components=n_eigenfunctions,
            normalize=True
        )

        # Apply k-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embedding)

        if return_centers:
            return labels, kmeans.cluster_centers_
        return labels

    def compute_heat_kernel_signature(
        self,
        time_scales: Optional[np.ndarray] = None,
        n_eigenfunctions: int = 100
    ) -> np.ndarray:
        """
        Compute multi-scale Heat Kernel Signature (HKS).

        HKS(x, t) = Σ_i e^(-λ_i * t) * φ_i(x)^2

        Args:
            time_scales: Array of time scales (default: auto-generated)
            n_eigenfunctions: Number of eigenfunctions to use

        Returns:
            HKS values (n_vertices, n_time_scales)
        """
        # Compute eigenpairs
        eigenvalues, eigenvectors = self.lb.compute_eigenpairs(k=n_eigenfunctions)

        if time_scales is None:
            # Auto-generate logarithmically spaced time scales
            lambda_min = eigenvalues[1] if len(eigenvalues) > 1 else eigenvalues[0]
            lambda_max = eigenvalues[-1]

            t_min = 4 * np.log(10) / lambda_max
            t_max = 4 * np.log(10) / lambda_min
            time_scales = np.logspace(np.log10(t_min), np.log10(t_max), 100)

        # Compute HKS for each time scale
        n_times = len(time_scales)
        hks = np.zeros((self.n_vertices, n_times))

        for i, t in enumerate(time_scales):
            exp_lambda = np.exp(-eigenvalues * t)
            hks[:, i] = np.sum(exp_lambda * eigenvectors ** 2, axis=1)

        return hks

    def compute_wave_kernel_signature(
        self,
        energy_scales: Optional[np.ndarray] = None,
        n_eigenfunctions: int = 100
    ) -> np.ndarray:
        """
        Compute Wave Kernel Signature (WKS).

        WKS is similar to HKS but uses wave equation instead of heat equation.

        Args:
            energy_scales: Array of energy scales (default: auto-generated)
            n_eigenfunctions: Number of eigenfunctions to use

        Returns:
            WKS values (n_vertices, n_energy_scales)
        """
        # Compute eigenpairs
        eigenvalues, eigenvectors = self.lb.compute_eigenpairs(k=n_eigenfunctions)

        if energy_scales is None:
            # Auto-generate logarithmically spaced energy scales
            lambda_min = eigenvalues[1] if len(eigenvalues) > 1 else eigenvalues[0]
            lambda_max = eigenvalues[-1]
            energy_scales = np.logspace(
                np.log10(lambda_min),
                np.log10(lambda_max),
                100
            )

        # Compute WKS for each energy scale
        n_energies = len(energy_scales)
        wks = np.zeros((self.n_vertices, n_energies))
        sigma = (energy_scales[-1] - energy_scales[0]) / n_energies  # Gaussian width

        for i, e in enumerate(energy_scales):
            # Gaussian window around energy e
            weights = np.exp(-((eigenvalues - e) ** 2) / (2 * sigma ** 2))
            wks[:, i] = np.sum(weights * eigenvectors ** 2, axis=1)

        return wks

    def smooth_mesh(
        self,
        iterations: int = 1,
        timestep: float = 0.01
    ) -> trimesh.Trimesh:
        """
        Smooth the mesh geometry using Laplacian smoothing.

        Args:
            iterations: Number of smoothing iterations
            timestep: Time step for each iteration

        Returns:
            Smoothed mesh
        """
        # Smooth each coordinate independently
        smoothed_vertices = np.zeros_like(self.mesh.vertices)

        for dim in range(3):
            smoothed_vertices[:, dim] = self.lb.smooth_function(
                self.mesh.vertices[:, dim],
                iterations=iterations,
                timestep=timestep
            )

        # Create new mesh with smoothed vertices
        smoothed_mesh = trimesh.Trimesh(
            vertices=smoothed_vertices,
            faces=self.mesh.faces,
            process=False
        )

        return smoothed_mesh

    def compute_spectral_distance(
        self,
        n_eigenfunctions: int = 50
    ) -> np.ndarray:
        """
        Compute spectral distance matrix between vertices.

        The spectral distance uses the diffusion distance metric.

        Args:
            n_eigenfunctions: Number of eigenfunctions to use

        Returns:
            Distance matrix (n_vertices, n_vertices)
        """
        # Get spectral embedding
        embedding = self.compute_spectral_embedding(
            n_components=n_eigenfunctions,
            normalize=True
        )

        # Compute pairwise distances
        from scipy.spatial.distance import cdist
        distances = cdist(embedding, embedding, metric='euclidean')

        return distances

    def detect_features(
        self,
        feature_type: str = 'hks',
        n_features: int = 10
    ) -> np.ndarray:
        """
        Detect geometric features on the mesh.

        Args:
            feature_type: Type of feature detector ('hks', 'wks', 'curvature')
            n_features: Number of top features to return

        Returns:
            Indices of feature vertices
        """
        if feature_type == 'hks':
            # Use HKS variance as feature detector
            hks = self.compute_heat_kernel_signature()
            feature_values = np.var(hks, axis=1)
        elif feature_type == 'wks':
            # Use WKS variance as feature detector
            wks = self.compute_wave_kernel_signature()
            feature_values = np.var(wks, axis=1)
        elif feature_type == 'curvature':
            # Use Gaussian curvature
            feature_values = np.abs(MeshUtils.compute_gaussian_curvature(self.mesh))
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        # Return top-k features
        feature_indices = np.argsort(feature_values)[-n_features:][::-1]
        return feature_indices

    def compute_shape_dna(
        self,
        n_eigenvalues: int = 100
    ) -> np.ndarray:
        """
        Compute the "Shape DNA" (eigenvalue spectrum).

        The eigenvalue spectrum is a shape descriptor that is
        isometry-invariant.

        Args:
            n_eigenvalues: Number of eigenvalues to compute

        Returns:
            Array of eigenvalues (shape DNA)
        """
        eigenvalues = self.lb.compute_laplacian_spectrum(k=n_eigenvalues)
        return eigenvalues

    def visualize_clustering(
        self,
        labels: np.ndarray,
        title: str = "Spectral Clustering"
    ):
        """
        Visualize clustering results on the mesh.

        Args:
            labels: Cluster labels for each vertex
            title: Title for visualization
        """
        ps.init()

        ps_mesh = ps.register_surface_mesh(
            "mesh",
            self.mesh.vertices,
            self.mesh.faces
        )

        # Convert labels to colors
        ps_mesh.add_scalar_quantity(
            title,
            labels.astype(float),
            defined_on='vertices',
            cmap='viridis',
            enabled=True
        )

        ps.show()

    def visualize_hks(
        self,
        hks: Optional[np.ndarray] = None,
        time_idx: int = 50
    ):
        """
        Visualize Heat Kernel Signature.

        Args:
            hks: HKS values (if None, will be computed)
            time_idx: Time scale index to visualize
        """
        if hks is None:
            hks = self.compute_heat_kernel_signature()

        ps.init()

        ps_mesh = ps.register_surface_mesh(
            "mesh",
            self.mesh.vertices,
            self.mesh.faces
        )

        ps_mesh.add_scalar_quantity(
            f"HKS (t={time_idx})",
            hks[:, time_idx],
            defined_on='vertices',
            cmap='plasma',
            enabled=True
        )

        ps.show()

    def visualize_features(
        self,
        feature_indices: np.ndarray,
        title: str = "Feature Points"
    ):
        """
        Visualize detected feature points.

        Args:
            feature_indices: Indices of feature vertices
            title: Title for visualization
        """
        ps.init()

        ps_mesh = ps.register_surface_mesh(
            "mesh",
            self.mesh.vertices,
            self.mesh.faces
        )

        # Create a binary indicator
        features = np.zeros(self.n_vertices)
        features[feature_indices] = 1.0

        ps_mesh.add_scalar_quantity(
            title,
            features,
            defined_on='vertices',
            cmap='coolwarm',
            enabled=True
        )

        # Add feature points as a point cloud
        ps.register_point_cloud(
            "features",
            self.mesh.vertices[feature_indices],
            radius=0.01
        )

        ps.show()

    def compute_biharmonic_distance(
        self,
        source_idx: int,
        m: int = 2
    ) -> np.ndarray:
        """
        Compute biharmonic distance from a source vertex.

        Solves: L^m φ = δ_s

        Args:
            source_idx: Source vertex index
            m: Order of the biharmonic operator (typically 2)

        Returns:
            Biharmonic distances
        """
        from scipy.sparse.linalg import spsolve

        # Create delta function at source
        delta = np.zeros(self.n_vertices)
        delta[source_idx] = 1.0 / self.lb.vertex_areas[source_idx]

        # Solve L^m φ = δ_s
        L = self.lb.get_laplacian()
        Lm = L.copy()
        for _ in range(m - 1):
            Lm = Lm @ L

        distances = spsolve(Lm, delta)

        # Normalize
        distances = distances - distances[source_idx]
        distances = np.abs(distances)

        return distances
