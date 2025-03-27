import re
import nltk
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from Data_collection import collect_data

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def Preprocess_text():
    def preprocess(text):
        text = text.lower()

        # Remove emails
        text = re.sub(r'\S+@\S+', ' ', text)

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', ' ', text)

        # Remove HTML tags
        text = BeautifulSoup(text, "html.parser").get_text()

        # Remove digits
        text = re.sub(r'\d+', '', text)

        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Tokenization and stopword removal + stemming
        tokens = text.split()
        tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]

        return ' '.join(tokens)

    df=collect_data()
    df["data_str"] = df.data.apply(lambda row: preprocess(row) )
    return df
