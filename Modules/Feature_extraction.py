from sklearn.feature_extraction.text import TfidfVectorizer
from Data_preprocessing import Preprocess_text
df=Preprocess_text()
def vectorizer():
    vectorizer = TfidfVectorizer(ngram_range=(2, 2), max_features=2**10)
    text_to_vector = vectorizer.fit_transform(df.data_str.values)
    return text_to_vector