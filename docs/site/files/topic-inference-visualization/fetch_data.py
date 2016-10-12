#!/usr/bin/env python

import pickle

import numpy as np
from sklearn.datasets import fetch_olivetti_faces, fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# Fetch and save the faces dataset first
faces = fetch_olivetti_faces()
with open("faces.npy", "wb") as f:
    # faces.data is multiplied by 255 because it is normalized in [0, 1]
    np.save(f, (faces.data*255).astype(np.int32).T)
    np.save(f, faces.target.astype(np.int32))

# Fetch the newsgroups raw data
newsgroups = fetch_20newsgroups(
    subset="train",
    remove=("headers", "footers", "quotes")
)

# Vectorize them with a vocabulary of 1000 words
tf_vectorizer = CountVectorizer(max_df=0.9, min_df=2, max_features=1000,
                                stop_words="english")
newsdata = tf_vectorizer.fit_transform(newsgroups.data)

# Save the vectorized data
with open("news.npy", "wb") as f:
    np.save(f, newsdata.toarray().astype(np.int32).T)
    np.save(f, newsgroups.target.astype(np.int32))

# Save the feature names in order to visualize the textual topics
feature_names = tf_vectorizer.get_feature_names()
pickle.dump(feature_names, open("fnames.pickle", "w"))
