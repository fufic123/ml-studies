import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

spotify_df = pd.read_csv("SpotifyData_by artists.csv")

print(spotify_df.info())
print(spotify_df.describe())
print(spotify_df.isnull().sum())
print(spotify_df.head())

plt.figure(figsize=(12, 10))
sns.heatmap(spotify_df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix - Spotify Features")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()




from sklearn.preprocessing import StandardScaler

features = [
    "valence", "danceability", "energy", "acousticness",
    "instrumentalness", "liveness", "loudness", "speechiness", "tempo"
]
X = spotify_df[features]
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)




from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
spotify_df["KMeans_Cluster"] = kmeans.fit_predict(scaled_X)

plt.figure(figsize=(10, 6))
plt.scatter(spotify_df["tempo"], spotify_df["danceability"], c=spotify_df["KMeans_Cluster"], cmap='viridis')
plt.xlabel("Tempo")
plt.ylabel("Danceability")
plt.title("K-Means Clustering on Spotify Dataset")
plt.colorbar(label="Cluster")
plt.savefig("kmeans_spotify.png")
plt.close()



from sklearn.metrics import pairwise_distances
import numpy as np

def k_medoids_manual(X, k, max_iter=100):
    np.random.seed(42)
    m = X.shape[0]
    medoid_idxs = np.random.choice(m, k, replace=False)
    medoids = X[medoid_idxs]

    for _ in range(max_iter):
        distances = pairwise_distances(X, medoids)
        labels = np.argmin(distances, axis=1)

        new_medoids = np.array([
            X[labels == i][np.argmin(np.sum(pairwise_distances(X[labels == i], [m]), axis=1))]
            if len(X[labels == i]) > 0 else medoids[i]
            for i, m in enumerate(medoids)
        ])
        if np.all(medoids == new_medoids):
            break
        medoids = new_medoids

    return np.argmin(pairwise_distances(X, medoids), axis=1)

spotify_df["PAM_Cluster"] = k_medoids_manual(scaled_X, k=3)

plt.figure(figsize=(10, 6))
plt.scatter(spotify_df["tempo"], spotify_df["danceability"], c=spotify_df["PAM_Cluster"], cmap='coolwarm')
plt.xlabel("Tempo")
plt.ylabel("Danceability")
plt.title("PAM Clustering on Spotify Dataset")
plt.colorbar(label="Cluster")
plt.savefig("pam_spotify.png")
plt.close()




import random

def clara(X, k, sample_size=40, num_samples=5):
    best_labels = None
    best_cost = float('inf')
    for _ in range(num_samples):
        sample_idxs = random.sample(range(len(X)), sample_size)
        sample_X = X[sample_idxs]
        labels_sample = k_medoids_manual(sample_X, k)
        medoids = sample_X[np.unique(labels_sample, return_index=True)[1]]

        cost = np.sum(np.min(pairwise_distances(X, medoids), axis=1))
        if cost < best_cost:
            best_cost = cost
            best_labels = np.argmin(pairwise_distances(X, medoids), axis=1)

    return best_labels

spotify_df["CLARA_Cluster"] = clara(scaled_X, 3)

plt.figure(figsize=(10, 6))
plt.scatter(spotify_df["tempo"], spotify_df["danceability"], c=spotify_df["CLARA_Cluster"], cmap='coolwarm')
plt.xlabel("Tempo")
plt.ylabel("Danceability")
plt.title("CLARA Clustering on Spotify Dataset")
plt.colorbar(label="Cluster")
plt.savefig("clara_spotify.png")
plt.close()





def claran(X, k, sample_size=40, num_samples=5, replace_fraction=0.2):
    best_labels = None
    best_cost = float('inf')

    for _ in range(num_samples):
        sample_idxs = random.sample(range(len(X)), sample_size)
        sample_X = X[sample_idxs]
        labels_sample = k_medoids_manual(sample_X, k)
        medoids = sample_X[np.unique(labels_sample, return_index=True)[1]]

        # Randomly replace some medoids
        for i in random.sample(range(len(medoids)), int(k * replace_fraction)):
            medoids[i] = X[random.randint(0, len(X)-1)]

        cost = np.sum(np.min(pairwise_distances(X, medoids), axis=1))
        if cost < best_cost:
            best_cost = cost
            best_labels = np.argmin(pairwise_distances(X, medoids), axis=1)

    return best_labels

spotify_df["CLARAN_Cluster"] = claran(scaled_X, 3)

plt.figure(figsize=(10, 6))
plt.scatter(spotify_df["tempo"], spotify_df["danceability"], c=spotify_df["CLARAN_Cluster"], cmap='coolwarm')
plt.xlabel("Tempo")
plt.ylabel("Danceability")
plt.title("CLARAN Clustering on Spotify Dataset")
plt.colorbar(label="Cluster")
plt.savefig("claran_spotify.png")
plt.close()






from sklearn.cluster import AgglomerativeClustering

agg = AgglomerativeClustering(n_clusters=3)
spotify_df["Agglomerative_Cluster"] = agg.fit_predict(scaled_X)

plt.figure(figsize=(10, 6))
plt.scatter(spotify_df["tempo"], spotify_df["danceability"], c=spotify_df["Agglomerative_Cluster"], cmap='coolwarm')
plt.xlabel("Tempo")
plt.ylabel("Danceability")
plt.title("Agglomerative Clustering on Spotify Dataset")
plt.colorbar(label="Cluster")
plt.savefig("agg_spotify.png")
plt.close()





from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

linkage_matrix = linkage(scaled_X, method='ward')
spotify_df["Divisive_Cluster"] = fcluster(linkage_matrix, t=3, criterion='maxclust')

plt.figure(figsize=(12, 6))
dendrogram(linkage_matrix, truncate_mode='lastp', p=30)
plt.title("Divisive Hierarchical Clustering Dendrogram")
plt.savefig("divisive_dendrogram.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(spotify_df["tempo"], spotify_df["danceability"], c=spotify_df["Divisive_Cluster"], cmap='coolwarm')
plt.xlabel("Tempo")
plt.ylabel("Danceability")
plt.title("Divisive Clustering on Spotify Dataset")
plt.colorbar(label="Cluster")
plt.savefig("divisive_spotify.png")
plt.close()




from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.7, min_samples=5)
spotify_df["DBSCAN_Cluster"] = dbscan.fit_predict(scaled_X)

plt.figure(figsize=(10, 6))
plt.scatter(spotify_df["tempo"], spotify_df["danceability"], c=spotify_df["DBSCAN_Cluster"], cmap='coolwarm')
plt.xlabel("Tempo")
plt.ylabel("Danceability")
plt.title("DBSCAN Clustering on Spotify Dataset")
plt.colorbar(label="Cluster")
plt.savefig("dbscan_spotify.png")
plt.close()






from minisom import MiniSom

som_grid = (3, 3)
som = MiniSom(x=som_grid[0], y=som_grid[1], input_len=scaled_X.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(scaled_X)
som.train_random(scaled_X, num_iteration=500)

bmu_indices = np.array([som.winner(x) for x in scaled_X])
som_clusters = [i * som_grid[1] + j for i, j in bmu_indices]
spotify_df["SOM_Cluster"] = som_clusters

plt.figure(figsize=(10, 6))
plt.scatter(spotify_df["tempo"], spotify_df["danceability"], c=spotify_df["SOM_Cluster"], cmap='coolwarm')
plt.xlabel("Tempo")
plt.ylabel("Danceability")
plt.title("SOM Clustering on Spotify Dataset")
plt.colorbar(label="Cluster")
plt.savefig("som_spotify.png")
plt.close()
