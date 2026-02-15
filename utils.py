"""Utility functions for mesh processing and geometric computations."""

import numpy as np
import trimesh
from typing import Tuple, Optional


class MeshUtils:
    """Utility functions for mesh operations."""

    @staticmethod
    def compute_vertex_areas(mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Compute the area associated with each vertex (Voronoi area).

        Args:
            mesh: Input triangular mesh

        Returns:
            Array of vertex areas (n_vertices,)
        """
        n_verts = len(mesh.vertices)
        vertex_areas = np.zeros(n_verts)

        # Accumulate 1/3 of each triangle's area to its vertices
        for face in mesh.faces:
            # Get triangle vertices
            v0, v1, v2 = mesh.vertices[face]

            # Compute triangle area
            area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

            # Distribute to vertices
            vertex_areas[face[0]] += area / 3.0
            vertex_areas[face[1]] += area / 3.0
            vertex_areas[face[2]] += area / 3.0

        return vertex_areas

    @staticmethod
    def compute_cotangent_weights(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute cotangent weights for the Laplace-Beltrami operator.

        For each edge (i,j), the weight is: w_ij = (cot α + cot β) / 2
        where α and β are the angles opposite to the edge.

        Args:
            mesh: Input triangular mesh

        Returns:
            Tuple of (row_indices, col_indices, weights) for sparse matrix construction
        """
        vertices = mesh.vertices
        faces = mesh.faces

        rows = []
        cols = []
        weights = []

        # Process each triangle
        for face in faces:
            i, j, k = face

            # Get vertex positions
            vi = vertices[i]
            vj = vertices[j]
            vk = vertices[k]

            # Compute edges
            e_ij = vj - vi
            e_jk = vk - vj
            e_ki = vi - vk

            # Compute cotangents for each angle
            # Angle at vertex k (opposite to edge ij)
            cot_k = MeshUtils._compute_cotangent(-e_ki, e_jk)

            # Angle at vertex i (opposite to edge jk)
            cot_i = MeshUtils._compute_cotangent(e_ij, -e_ki)

            # Angle at vertex j (opposite to edge ki)
            cot_j = MeshUtils._compute_cotangent(e_jk, e_ij)

            # Add contributions for edge (i,j)
            rows.extend([i, j])
            cols.extend([j, i])
            weights.extend([cot_k, cot_k])

            # Add contributions for edge (j,k)
            rows.extend([j, k])
            cols.extend([k, j])
            weights.extend([cot_i, cot_i])

            # Add contributions for edge (k,i)
            rows.extend([k, i])
            cols.extend([i, k])
            weights.extend([cot_j, cot_j])

        return np.array(rows), np.array(cols), np.array(weights)

    @staticmethod
    def _compute_cotangent(v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute cotangent of angle between two vectors.

        cot(θ) = cos(θ) / sin(θ) = (v1 · v2) / ||v1 × v2||

        Args:
            v1, v2: Input vectors

        Returns:
            Cotangent value
        """
        dot_product = np.dot(v1, v2)
        cross_product = np.cross(v1, v2)
        cross_norm = np.linalg.norm(cross_product)

        # Avoid division by zero
        if cross_norm < 1e-10:
            return 0.0

        return dot_product / cross_norm

    @staticmethod
    def compute_face_normals(mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Compute normals for each face.

        Args:
            mesh: Input triangular mesh

        Returns:
            Array of face normals (n_faces, 3)
        """
        return mesh.face_normals

    @staticmethod
    def compute_vertex_normals(mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Compute normals for each vertex (area-weighted average).

        Args:
            mesh: Input triangular mesh

        Returns:
            Array of vertex normals (n_vertices, 3)
        """
        return mesh.vertex_normals

    @staticmethod
    def normalize_function(values: np.ndarray, mode: str = 'unit') -> np.ndarray:
        """
        Normalize a function defined on mesh vertices.

        Args:
            values: Function values at vertices
            mode: Normalization mode ('unit', 'zero_mean', 'range')

        Returns:
            Normalized values
        """
        if mode == 'unit':
            # Normalize to unit norm
            norm = np.linalg.norm(values)
            return values / (norm + 1e-10)
        elif mode == 'zero_mean':
            # Zero mean, unit variance
            mean = np.mean(values)
            std = np.std(values)
            return (values - mean) / (std + 1e-10)
        elif mode == 'range':
            # Normalize to [0, 1] range
            min_val = np.min(values)
            max_val = np.max(values)
            return (values - min_val) / (max_val - min_val + 1e-10)
        else:
            return values

    @staticmethod
    def compute_mean_edge_length(mesh: trimesh.Trimesh) -> float:
        """
        Compute the mean edge length of the mesh.

        Args:
            mesh: Input triangular mesh

        Returns:
            Mean edge length
        """
        edges = mesh.edges_unique
        edge_lengths = np.linalg.norm(
            mesh.vertices[edges[:, 0]] - mesh.vertices[edges[:, 1]],
            axis=1
        )
        return np.mean(edge_lengths)

    @staticmethod
    def compute_gaussian_curvature(mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Compute discrete Gaussian curvature at each vertex using angle deficit.

        K_i = (2π - Σ θ_j) / A_i

        where θ_j are angles at vertex i, and A_i is the vertex area.

        Args:
            mesh: Input triangular mesh

        Returns:
            Array of Gaussian curvatures (n_vertices,)
        """
        n_verts = len(mesh.vertices)
        angle_sum = np.zeros(n_verts)
        vertex_areas = MeshUtils.compute_vertex_areas(mesh)

        vertices = mesh.vertices
        faces = mesh.faces

        # Compute angle sum at each vertex
        for face in faces:
            for idx in range(3):
                i = face[idx]
                j = face[(idx + 1) % 3]
                k = face[(idx + 2) % 3]

                # Compute angle at vertex i
                v1 = vertices[j] - vertices[i]
                v2 = vertices[k] - vertices[i]

                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                angle = np.arccos(cos_angle)

                angle_sum[i] += angle

        # Compute Gaussian curvature
        gaussian_curvature = (2 * np.pi - angle_sum) / (vertex_areas + 1e-10)

        return gaussian_curvature
