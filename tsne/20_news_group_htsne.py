import scipy.io
import scipy
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from collections import defaultdict
from dendrogram_weights import get_dendrogram_weights
from general_htsne import g_htsne
from general_tsne import g_tsne
from importlib import reload 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups

SAMPLE_SIZE=800

np.random.seed(42)

categories = ['rec.sport.baseball','rec.sport.hockey','sci.electronics', 'sci.space','sci.med']

newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)

sample_ids = np.random.permutation(len(newsgroups_train.data))
vectorizer = TfidfVectorizer()

vectors = vectorizer.fit_transform(np.array(newsgroups_train.data)[sample_ids[:SAMPLE_SIZE]])

print(vectors.shape)

y = newsgroups_train.target[sample_ids[:SAMPLE_SIZE]]

print("Starting to iterate")

visualized = g_htsne(vectors.toarray(), y, save_name="20_news_htsne", iterations=1000)