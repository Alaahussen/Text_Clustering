from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
from Feature_extraction import vectorizer
from PCA import dimention_reduction
from model import k_means
text_to_vector=vectorizer()
X_pca=dimention_reduction()
cluster_labels=k_means()
def visualize():
    plt.figure(figsize=(10,7))
    palette = sns.color_palette("hls", 20)
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette=palette, legend='full')
    plt.title("PCA Visualization of Text Clusters")
    plt.grid(True)
    plt.show()

