from sklearn.cluster import KMeans
from PCA import dimention_reduction
X_pca=dimention_reduction()
def k_means():
    kmeans = KMeans(n_clusters=20, random_state=42, n_init='auto')
    cluster_labels = kmeans.fit_predict(X_pca)
    return cluster_labels