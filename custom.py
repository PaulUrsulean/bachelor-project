from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle
import trident
import scipy.sparse as sp


class TripleTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, min_df=1, max_df=1.0, unique=True, vectorizer=None, stop_words=None, max_features=None):
		self.min_df 		= min_df
		self.max_df 		= max_df
		self.unique 		= unique
		self.stop_words 	= stop_words
		self.max_features 	= max_features

		self.vec 			= TfidfVectorizer(min_df=self.min_df, max_df=self.max_df, stop_words=self.stop_words, max_features=self.max_features) if vectorizer is None else vectorizer


		with open("short_descriptions.txt", "rb") as f:
			short_desc = pickle.load(f)

		fb = trident.Db("fb15k")
		self.get_desc = np.vectorize(lambda index: short_desc[fb.lookup_str(index)])


	def fit(self, x, y=None):

		if x.shape[1] != 3:
			raise ValueError("The input matrix does not contain 3 columns, and thus it is not a triple.")

		docs = np.unique(np.concatenate((x[:,0], x[:,2]))) if self.unique else np.concatenate((x[:,0], x[:,2]))

		# print(len(docs))

		self.vec.fit(self.get_desc(docs))
		# print(len(self.vec.get_feature_names()))
		return self

	def transform(self, x):

		if x.shape[1] != 3:
			raise ValueError("The input matrix does not contain 3 columns, and thus it is not a triple.")

		# rels = np.empty([len(x),1], dtype=np.int64)
		# rels[:,0] = x[:,1]

		# features = sp.hstack((rels, self.vec.transform(self.get_desc(x[:,0])), self.vec.transform(self.get_desc(x[:,2]))), format='csr')
		# features = sp.hstack((self.__simple_onehot(x[:,1]), self.vec.transform(self.get_desc(x[:,0])), self.vec.transform(self.get_desc(x[:,2]))), format='csr')
		features = sp.hstack((self.vec.transform(self.get_desc(x[:,0])), self.vec.transform(self.get_desc(x[:,2]))), format='csr')

		# print(features.shape)

		return features

	def get_vectorizer(self):
		return self.vec

	def __simple_onehot(self, rels):
		rows = np.arange(len(rels))
		return sp.csr_matrix((np.ones(len(rels), dtype=np.int64), (rows, rels)))


	# def fit_transform(self, x):
	# 	return self.vec.fit(x).transform(x)