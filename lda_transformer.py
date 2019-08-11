import scipy
import tensorflow as tf
import numpy as np
from collections import Counter
from gensim.parsing.preprocessing import preprocess_string
from gensim.corpora import Dictionary


class LDATransformer:
    """Preps data for LDA.
    TODO: add options to slim down vocab and filter words. Also make the methods more efficient.
    """

    def fit(self, texts):
        all_words = []
        docs = [preprocess_string(d) for d in texts]
        self.vocab = Dictionary(docs)
        self.vocab.filter_extremes()
        return self

    def transform(self, docs):
        """TODO: speed up for loop."""
        all_docs = []
        i = 0
        for d in docs:
            words = preprocess_string(d)
            id_ct = self.vocab.doc2bow(words)
            if len(id_ct) < 1:
                continue
            else:
                id, ct = zip(*id_ct)
                all_docs.extend([(i, j) for j in id])
                i += 1
        return all_docs
