import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.figsize"] = (8, 5)

df = pd.read_csv(r"C:\Users\AL SharQ\Downloads\intern\Mall_Customers.csv") 
print(df.head())
print(df.info(), end="\n\n")

plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Gender",
    palette="Set1",
    s=70,
    edgecolor="black"
)
plt.title("Customers by Annual Income vs. Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Gender")
plt.tight_layout()
plt.show()

X = df.drop("CustomerID", axis=1).copy()
X["Gender"] = LabelEncoder().fit_transform(X["Gender"])  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(k_range, inertia, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Inertia")
plt.tight_layout()
plt.show()

k_opt = 5  
kmeans = KMeans(n_clusters=k_opt, n_init="auto", random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df["Cluster"] = clusters

sil_score = silhouette_score(X_scaled, clusters)
print(f"\nSilhouette Score (K={k_opt}): {sil_score:.3f}\n")

pca = PCA(n_components=2, random_state=42)
components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
pca_df["Cluster"] = clusters

sns.scatterplot(data=pca_df, x="PC1", y="PC2",hue="Cluster", palette="Set2", s=70, edgecolor="white")
plt.title(f"K-Means Clusters (K={k_opt})   |   Silhouette = {sil_score:.2f}")
plt.tight_layout()
plt.show()
centroids_scaled = kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)
centroids_df = pd.DataFrame(
    centroids_original,
    columns=X.columns,
    index=[f"Cluster {c}" for c in range(k_opt)]
)
print("Cluster Centroids (original scale):")
print(centroids_df.round(2))
