# Bag of words classifier with SVM
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error, explained_variance_score
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from custom import TripleTransformer

import pandas as pd
import trident
import dill as pickle
import numpy as np
import nltk
from collections import defaultdict
from operator import itemgetter
import scipy.sparse as sp
from scipy.stats import uniform
import random

import time

fb = trident.Db("fb15k")

r2 = []
mederr = []
weights = []

# NEED TO DEPRECATE
def feature_vector(vectorizer, h, r, t):
	return vectorizer.transform(np.asarray([h, r, t], np.int64).transpose())

def get_model():
	return joblib.load("pipe.pkl")

# Creates a model based on triples received, attempts to make predictions
def train(rel_id):

	n_triples = fb.count_p(rel_id)

	TEST_SPLIT = 0.1

	if n_triples < 10:
		TEST_SPLIT = 0.4
		if n_triples <=6:
			TEST_SPLIT = 0.5
			if n_triples == 1:
				TEST_SPLIT=0

	X, y = generate_sets(rel_id)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT)

	_, X_val, _, y_val = train_test_split(X_train, y_train, test_size=0.4)


	# pipe = Pipeline([('vectorizer', TripleTransformer(min_df=0.0005)), ('estimator', SVR(gamma='auto'))])
	pipe = Pipeline([('vectorizer', TripleTransformer(min_df = 0.002)), ('estimator', SVR(gamma='auto', kernel='linear'))])
	# pipe = Pipeline([('vectorizer', TripleTransformer(min_df=0.13478528733776624, max_df = 0.6783292560120808)), ('estimator', SVR(gamma='auto', kernel='linear'))])

	# print(X_train.shape)

	pipe.fit(X_train, y_train)

	with open("./rel_models/" + str(rel_id) + ".pkl", "wb") as f:
		joblib.dump([pipe.named_steps['vectorizer'].get_vectorizer(), pipe.named_steps['estimator']], f)

	if n_triples > 1:
		y_pred = pipe.predict(X_test)
		r,m = get_accuracy(y_test, y_pred)
		r2.append(r)
		mederr.append(m)
		weights.append(n_triples)
	

	
def get_accuracy(y_test, y_pred):

	y_guess = np.random.rand(len(y_test))

	# print("\nEstimator r2:", r2_score(y_test, y_pred))
	# print("Baseline  r2:", r2_score(y_test, y_guess))

	# print("\nEstimator EVS:", explained_variance_score(y_test, y_pred))
	# print("Baseline  EVS:", explained_variance_score(y_test, y_guess))

	# print("\nEstimator MAE:", mean_absolute_error(y_test, y_pred))
	# print("Baseline  MAE:", mean_absolute_error(y_test, y_guess))

	# print("\nEstimator MSE:", mean_squared_error(y_test, y_pred))
	# print("Baseline  MSE:", mean_squared_error(y_test, y_guess))

	# print("\nEstimator Median Absolute Error", median_absolute_error(y_test, y_pred))
	# print("Baseline  Median Absolute Error", median_absolute_error(y_test, y_guess), "\n")

	return r2_score(y_test, y_pred), median_absolute_error(y_test, y_pred)

def generate_relset(rel_id, use_corrupted, percentage):
	
	positive = [(tup[0], rel_id, tup[1]) for tup in fb.os(rel_id)]


	entities = set(fb.all_s()).union(set(fb.all_o()))
	diff_set = set(fb.all()).difference(set(positive))

	corrections = 0

	if percentage < 1 and percentage > 0:
		positive = random.sample(positive, int(percentage*fb.count_p(rel_id)))

	# negative = np.empty([len(positive),3])

	# I know this is ugly
	if use_corrupted and len(positive) > 1:

		permute = np.random.permutation(positive)
		thresh  = int(len(permute)/2)

		corr_heads = np.asarray(positive)
		corr_tails = np.asarray(positive)

		corr_heads[:,0] = random.sample(entities, len(corr_heads))
		corr_tails[:,2] = random.sample(entities, len(corr_tails))

		for i in range(len(corr_heads)):
			while fb.exists(corr_heads[i][0], corr_heads[i][1], corr_heads[i][2]):
				corr_heads[i][0] = random.sample(entities, 1)[0]
				corrections+=1

		for i in range(len(corr_tails)):
			while fb.exists(corr_tails[i][0], corr_tails[i][1], corr_tails[i][2]):
				corr_tails[i][2] = random.sample(entities, 1)[0]
				corrections+=1

		negative = np.append(corr_heads, corr_tails, axis=0)

	else:
		negative = np.asarray(random.sample(diff_set, len(positive)), np.int64)
		negative[:,1] = np.ones(len(negative), np.int64)*rel_id

		for i in range(len(negative)):
			while fb.exists(negative[i][0], rel_id, negative[i][2]):
				neg = random.sample(diff_set, 1)[0]
				negative[i] = [neg[0], rel_id, neg[2]]
				corrections+=1


	return np.concatenate((positive, negative)), np.append(np.ones(len(positive)), np.zeros(len(negative)))

def generate_sets(rel_ids, use_corrupted=False, percentage=1.):


	if isinstance(rel_ids, int) or isinstance(rel_ids, np.int64) or isinstance(rel_ids, np.int32):
		rel_ids = [rel_ids]

	elif not isinstance(rel_ids, list) and not isinstance(rel_ids, np.ndarray):
		return

	X = np.empty([0,3], dtype=np.int64)
	y = np.empty(0, dtype=np.int64)

	for rel_id in rel_ids:
		X_i, y_i = generate_relset(rel_id, use_corrupted, percentage)
		X = np.append(X, X_i, axis=0)
		y = np.append(y, y_i)


	return X, y


def tuples_consistent(tuples, answers):
	

	for i in range(len(tuples)):
		if not fb.exists(tuples[i][0], tuples[i][1], tuples[i][2]) == bool(answers[i]):
			return False

	return True

# ids = [i for i in random.sample(range(fb.n_relations()), 20) if fb.count_p(i) > 100 and fb.count_p(i) < 5000]

# errors = []

# train(0)

startall = time.time()

for i in np.unique(fb.all_p()):
	try:
		print("Training relation",i)
		start = time.time()
		train(i)

		print("Done. Execution time:", time.time() - start)
	except:
		print("\nERROR AT RELATION ID", i)
		print(sys.exc_info()[0])
		raise
		sys.exit()
		errors.append(i)

if errors != []:
	with open("errors.pkl", "wb") as f:
		joblib.dump(errors, f)





# a = []
# b = []
# weights = []
# for i in ids:
# 	c, d = train(i)
# 	a.append(c)
# 	b.append(d)
# 	weights.append(fb.count_p(i))

weights = np.asarray(weights)
weights = weights/weights.max()

print("Average r2:",np.average(r2, weights=weights), "\nAverge median error:", np.average(mederr, weights=weights))

print("Total execution time:", time.time() - startall)