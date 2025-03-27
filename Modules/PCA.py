from sklearn.decomposition import PCA
from Feature_extraction import vectorizer

text_to_vector=vectorizer()
def dimention_reduction():
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(text_to_vector.toarray())
    return X_pca