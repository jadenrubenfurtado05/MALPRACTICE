
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

# Load dataset
data = load_iris()
X = data.data

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Spectral Clustering
model = SpectralClustering(
    n_clusters=3,
    affinity='nearest_neighbors',
    n_neighbors=10,
    assign_labels='discretize',
    random_state=42
)

# Predict clusters
labels = model.fit_predict(X_scaled)

# Display results
print("Cluster Labels:\n", labels)
print("Silhouette Score:",
      silhouette_score(X_scaled, labels))

# Visualize clusters using first two features
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels,
    cmap='viridis'
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Spectral Clustering")
plt.show()
