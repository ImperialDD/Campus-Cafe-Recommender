"""PCA via NumPy eigendecomposition — no sklearn.

Pipeline (all from scratch):
  mean_center -> covariance -> eigendecomposition -> sort ->
  PC1/PC2 -> explained variance -> project students / new user.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PCAResult:
    means: np.ndarray              # feature means used for centering (n_features,)
    centered: np.ndarray           # mean-centered matrix (n x features)
    covariance: np.ndarray         # covariance matrix (features x features)
    eigenvalues: np.ndarray        # sorted descending
    eigenvectors: np.ndarray       # columns sorted to match eigenvalues
    explained_variance_ratio: np.ndarray
    pc1: np.ndarray                # first eigenvector (features,)
    pc2: np.ndarray                # second eigenvector (features,)
    pc1_variance: float
    pc2_variance: float
    student_coords: np.ndarray     # (n x 2) projection of training students
    feature_names: list[str] = field(default_factory=list)


def mean_center(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the feature-wise mean from each column.

    Returns (centered_matrix, feature_means).
    """
    means = matrix.mean(axis=0)
    centered = matrix - means
    return centered, means


def calculate_covariance(centered: np.ndarray) -> np.ndarray:
    """Covariance matrix of the centered data.

    Uses np.cov with rowvar=False (observations in rows, features in columns).
    """
    return np.cov(centered, rowvar=False, ddof=1)


def perform_pca(matrix: np.ndarray, feature_names: list[str] | None = None) -> PCAResult:
    """Run the full PCA pipeline and return a PCAResult.

    Steps:
      1. Mean-center feature-wise.
      2. Covariance matrix.
      3. Eigendecomposition (np.linalg.eigh for symmetric matrices).
      4. Sort eigenvalues descending + reorder eigenvectors.
      5. Extract PC1, PC2.
      6. Explained variance ratios.
      7. Project training students onto PC1/PC2.
    """
    centered, means = mean_center(matrix)
    cov = calculate_covariance(centered)

    # eigh returns ascending eigenvalues for symmetric matrices; reverse for descending.
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    total = eigvals.sum()
    explained = eigvals / total if total != 0 else np.zeros_like(eigvals)

    pc1 = eigvecs[:, 0]
    pc2 = eigvecs[:, 1]

    # Project: centered @ [pc1 pc2]
    components = np.column_stack([pc1, pc2])
    student_coords = centered @ components

    return PCAResult(
        means=means,
        centered=centered,
        covariance=cov,
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        explained_variance_ratio=explained,
        pc1=pc1,
        pc2=pc2,
        pc1_variance=float(explained[0]),
        pc2_variance=float(explained[1]),
        student_coords=student_coords,
        feature_names=feature_names or [],
    )


def calculate_explained_variance(pca: PCAResult) -> np.ndarray:
    """Explained variance ratio per component (already computed; helper for UI)."""
    return pca.explained_variance_ratio


def project_students(pca: PCAResult) -> np.ndarray:
    """Return the (n x 2) PC1/PC2 coordinates for training students."""
    return pca.student_coords


def project_new_user(ratings: np.ndarray, pca: PCAResult) -> np.ndarray:
    """Project a NEW user into the ORIGINAL PCA space.

    1. Subtract the ORIGINAL training-data feature means.
    2. Project onto ORIGINAL PC1 and PC2.
    """
    centered = ratings - pca.means
    components = np.column_stack([pca.pc1, pca.pc2])
    return centered @ components
