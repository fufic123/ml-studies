import json
import re

import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, silhouette_score, davies_bouldin_score



with open("books.json", "r", encoding="utf-8") as json_file:
    books_data = json.load(json_file)


df = pd.DataFrame(books_data)

df["price"] = df["price"].apply(lambda x: float(re.sub(r'[^\d.]', '', x)))

rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
df["rating_numeric"] = df["rating"].map(rating_map)


titles = " ".join(df["title"].tolist())
stopwords = set(STOPWORDS)
wordcloud = WordCloud(
    stopwords=stopwords,
    background_color="white",
    width=800,
    height=400
).generate(titles)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Book Titles")
plt.savefig("wordcloud.png")
plt.show()


X = df[["price", "rating_numeric"]].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
df.loc[X.index, "cluster"] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))

for cluster in sorted(df["cluster"].unique()):
    plt.scatter(
        X_pca[df["cluster"] == cluster, 0],
        X_pca[df["cluster"] == cluster, 1],
        label=f"Cluster {int(cluster)}",
        alpha=0.6
    )

plt.title("KMeans Clustering of Books (PCA-reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("kmeans_clusters_pca.png")
plt.show()




inertia = []
silhouette = []
davies = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    
    inertia.append(kmeans.inertia_)
    silhouette.append(silhouette_score(X_scaled, labels))
    davies.append(davies_bouldin_score(X_scaled, labels))


plt.figure(figsize=(15, 4))

# Elbow plot
plt.subplot(1, 3, 1)
plt.plot(k_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")

# Silhouette
plt.subplot(1, 3, 2)
plt.plot(k_range, silhouette, marker='o', color='green')
plt.title("Silhouette Score")
plt.xlabel("Number of clusters")

# Davies-Bouldin
plt.subplot(1, 3, 3)
plt.plot(k_range, davies, marker='o', color='red')
plt.title("Davies-Bouldin Index")
plt.xlabel("Number of clusters")

plt.tight_layout()
plt.savefig("cluster_evaluation.png")
plt.show()



kmeans = KMeans(n_clusters=4, random_state=42)
df.loc[X.index, "cluster"] = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 5))

for cluster in sorted(df["cluster"].unique()):
    plt.scatter(
        X_pca[df["cluster"] == cluster, 0],
        X_pca[df["cluster"] == cluster, 1],
        label=f"Cluster {int(cluster)}",
        alpha=0.6
    )

plt.title("KMeans Clustering of Books (PCA-reduced)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("kmeans_clusters_pca_4.png")
plt.show()



df["cluster"] = df["cluster"].astype(int)
df["Genre"] = "Unknown" 
df["Genre label"] = df["cluster"]
df.to_csv("books_clustered.csv", index=False)

plt.figure(figsize=(8, 5))
df.boxplot(column="price", by="Genre label", grid=False)

plt.title("Price Distribution by Genre Label")
plt.suptitle("")
plt.xlabel("Genre Label (Cluster ID)")
plt.ylabel("Price (£)")
plt.tight_layout()
plt.savefig("price_boxplot_by_genre.png")
plt.show()



y = df.loc[X.index, "Genre label"]

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

mlp.fit(X_train, y_train)



y_pred = mlp.predict(X_test)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix for Genre Classification")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()