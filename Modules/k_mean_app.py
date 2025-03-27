import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from Data_collection import collect_data
import pickle
import pandas as pd

# Load saved models
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('kmeans.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Simulated DataFrame with cluster names (Modify as needed)
df = pd.DataFrame({
    'target': list(range(kmeans.n_clusters)),  # Assuming each cluster has a unique index
    'target_names': [f"Cluster {i} - Topic Name" for i in range(kmeans.n_clusters)]
})

# Optional: Your preprocessing function
def preprocess(text):
    import re
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer

    stemmer = PorterStemmer()
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Streamlit UI
st.set_page_config(page_title="Text Article Cluster Classifier", layout="centered")
st.title("🧠 Article Topic Classifier (KMeans)")
st.write("Enter any article or paragraph below. The app will predict which article group it belongs to!")

user_input = st.text_area("✍️ Enter Article Text:", height=200)

if st.button("🔍 Predict Cluster"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        preprocessed_text = preprocess(user_input)
        new_vector = vectorizer.transform([preprocessed_text])
        new_vector_pca = pca.transform(new_vector.toarray())
        predicted_cluster = kmeans.predict(new_vector_pca)[0]

        # Get the cluster name from the DataFrame
        df=collect_data()
        out = df[df['target'] == predicted_cluster]
        cluster_name = out['target_names'].iloc[0] if not out.empty else "Unknown Cluster"

        st.success(f"✅ This article belongs to **{cluster_name}**")
