import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv("cybersecurity_intrusion_data.csv")

df = df.drop(columns=["session_id"])

categorical_cols = ["protocol_type", "encryption_used", "browser_type"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# побудова дендрограми
linked = linkage(X_scaled, method="ward")
plt.figure(figsize=(12, 6))
dendrogram(linked)
plt.title("Dendrogram - Hierarchical Clustering")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()

# оптимальна кількість кластерів
n_clusters = 3

hclust = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
labels_h = hclust.fit_predict(X_scaled)
df["cluster_hier"] = labels_h

# k-Means для порівняння
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels_k = kmeans.fit_predict(X_scaled)
df["cluster_kmeans"] = labels_k

print("Silhouette Score (Hierarchical):", silhouette_score(X_scaled, labels_h))
print("Silhouette Score (k-Means):", silhouette_score(X_scaled, labels_k))

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_h)
plt.title("Hierarchical Clustering")

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_k)
plt.title("K-Means")

plt.show()

print(df.head())
