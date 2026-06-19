import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
daily_features = pd.read_csv('summarystats.csv')
feature_cols = [
    "average",
    "max",
    "min",
    "stdev",
    "var",
    "below",
    "above"
]
X = daily_features[feature_cols]

daily_features = daily_features.dropna(subset=feature_cols)
X = daily_features[feature_cols]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
silhouette_scores = []

K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Plot")
plt.savefig("Elbow.png")
#plt.show(block=False)
#plt.pause(0.001)

plt.figure(figsize=(8, 5))
plt.plot(K_range, silhouette_scores, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score Plot")
plt.savefig("Silhouette.png")
#plt.show(block=False)
#plt.pause(0.001)

k = 5

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
daily_features["cluster"] = kmeans.fit_predict(X_scaled)


cluster_summary = daily_features.groupby("cluster")[feature_cols].mean()
print(cluster_summary)

cluster_counts = daily_features["cluster"].value_counts().sort_index()
print(cluster_counts)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

daily_features["PC1"] = X_pca[:, 0]
daily_features["PC2"] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
plt.scatter(
    daily_features["PC1"],
    daily_features["PC2"],
    c=daily_features["cluster"],
    alpha=0.7
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Daily Glucose Clusters Visualized with PCA")
plt.colorbar(label="Cluster")
plt.savefig("Clusters.png")
#plt.show(block=False)
#plt.pause(0.001)

daily_features.to_csv("daily_glucose_clusters.csv", index=False)
plt.show()



