
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_iris()
X = data.data

# Step 1: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Calculate covariance matrix
cov_matrix = np.cov(X_scaled.T)
print("Covariance Matrix:\n", cov_matrix)

# Step 3: Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Step 4: Sort eigenvalues in descending order
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# Step 5: Select first 2 principal components
components = eigenvectors[:, :2]

# Step 6: Transform original data
X_pca = np.dot(X_scaled, components)

print("Transformed Data:\n", X_pca[:5])

# Step 7: Reconstruct original data
X_reconstructed = np.dot(X_pca, components.T)

# Convert back to original scale
X_reconstructed = scaler.inverse_transform(X_reconstructed)

print("Reconstructed Data:\n", X_reconstructed[:5])

# Reconstruction error
error = np.mean((X - X_reconstructed) ** 2)
print("Reconstruction Error:", error)
