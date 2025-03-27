from Data_collection import collect_data
from Data_preprocessing import Preprocess_text
from Feature_extraction import vectorizer
from PCA import dimention_reduction
from model import k_means
from Visualization import visualize
from sklearn.metrics import silhouette_score
X_pca=dimention_reduction()
cluster_labels=k_means()
score = silhouette_score(X_pca, cluster_labels)
print(f"Silhouette Score: {score:.4f}")
visualize()

