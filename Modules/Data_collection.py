import pandas as pd
import numpy as np
import re
from sklearn.datasets import fetch_20newsgroups
def collect_data():
    dataset = fetch_20newsgroups(subset='all', 
                    shuffle=False, remove=('headers', 'footers', 'quotes'))
    df = pd.DataFrame()
    df["data"] = dataset["data"]
    df["target"] = dataset["target"]
    df["target_names"] = df.target.apply(lambda row: dataset["target_names"][row])
    return df
