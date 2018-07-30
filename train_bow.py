# Bag of words classifier with SVM
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR, LinearSVR
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


# NEED TO DEPRECATE
def feature_vector(vectorizer, h, r, t):
	return vectorizer.transform(np.asarray([h, r, t], np.int64).transpose())

def get_model():
	return joblib.load("pipe.pkl")

# Creates a model based on triples received, attempts to make predictions
def train(rel_id, use_corrupted=False):

	n_triples = fb.count_p(rel_id)

	TEST_SPLIT = 0.1

	if n_triples < 10:
		TEST_SPLIT = 0.4
		if n_triples <=6:
			TEST_SPLIT = 0.5
			if n_triples == 1:
				TEST_SPLIT=0

	X, y = generate_sets(rel_id, use_corrupted)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SPLIT)

	n_docs = np.unique(np.concatenate((X_train[:,0], X_train[:,2]))).size

	_, X_val, _, y_val = train_test_split(X_train, y_train, test_size=0.4)


	pipe = Pipeline([('vectorizer', TripleTransformer(min_df=0.002)), ('estimator', SVR(gamma='auto', kernel='linear'))])
	# pipe = Pipeline([('vectorizer', TripleTransformer(min_df = 0.002, stop_words='english')), ('estimator', LinearSVR(epsilon=0.1))])
	# pipe = Pipeline([('vectorizer', TripleTransformer(min_df = 0.002, max_df = (0.008 if int(0.008*n_docs) >= 8 else 8))), ('estimator', SVR(gamma='auto', kernel='linear'))])
	# pipe = Pipeline([('vectorizer', TripleTransformer(min_df=0.13478528733776624, max_df = 0.6783292560120808)), ('estimator', SVR(gamma='auto', kernel='linear'))])

	# print(X_train.shape)



	# search = GridSearchCV(pipe, param_grid, iid=False)
	# search.fit(X_train, y_train)

	# for key in search.best_params_:
	# 	param_freqs[key][search.best_params_[key]] += n_triples

	# r2.append(search.best_score_)
	# weights.append(n_triples)

	# {'vectorizer__min_df': defaultdict(<class 'int'>, {1: 1234, 0.01: 4114, 0.005: 7020, 0.04: 3506, 0.002: 3576}),
	# 'vectorizer__max_df': defaultdict(<class 'int'>, {0.2: 4511, 1.0: 2069, 0.1: 5311, 0.7: 3064, 0.05: 4495}),
	# 'vectorizer__max_features': defaultdict(<class 'int'>, {None: 7064, 2500: 6998, 10000: 5388}),
	# 'estimator__C': defaultdict(<class 'int'>, {2.0: 3544, 1.0: 2263, 0.5: 13643})}

	# pipe = Pipeline([('vectorizer', TripleTransformer(min_df = 0.005, max_df=0.1, max_features=None, stop_words='english')), ('estimator', LinearSVR(C=0.5, epsilon=0.1, max_iter=4000))])




	# print("Best parameters:", search.best_params_)
	# print("R2 score:{}\n".format(search.best_score_))

	pipe.fit(X_train, y_train)
	# print(rel_id, len(pipe.named_steps['vectorizer'].get_vectorizer().get_feature_names()))

	# with open("./rel_models2/" + str(rel_id) + ".pkl", "wb") as f:
	# # with open("pipe.pkl", "wb") as f:
	# 	joblib.dump([pipe.named_steps['vectorizer'].get_vectorizer(), pipe.named_steps['estimator']], f)

	# # # with open("vectorizer.pkl", "wb") as f:
	# # # 	joblib.dump(pipe.named_steps['vectorizer'].get_vectorizer(), f)

	# # # with open("model.pkl", "wb") as f:
	# # # 	joblib.dump(pipe.named_steps['estimator'], f)

	# if n_triples > 1:
	# 	y_pred = pipe.predict(X_test)
	# 	r,m, mean = get_accuracy(y_test, y_pred)
	# 	# print("R2: {}\nMederr: {}".format(r,m))
	# 	r2.append(r)
	# 	mederr.append(m)
	# 	meanerr.append(mean)
	# 	weights.append(n_triples)
	

	
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

	return r2_score(y_test, y_pred), median_absolute_error(y_test, y_pred), mean_absolute_error(y_test, y_pred)

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


startall = time.time()

fb = trident.Db("fb15k")

r2 = []
mederr = []
meanerr = []
weights = []


# for i in np.unique(fb.all_p()):
# 	try:
# 		print("\nTraining relation",i)
# 		start = time.time()
# 		train(i)

# 		print("Done. Execution time:", time.time() - start)
# 	except:
# 		print("\nERROR AT RELATION ID", i)
# 		errors.append(i)

# if errors != []:
# 	with open("errors.pkl", "wb") as f:
# 		joblib.dump(errors, f)

# train(0)

# errors = joblib.load("errors.pkl")

# print(errors)

# for i in errors:

# 	pip = []

# 	with open("./rel_models/" + str(i) + ".pkl", "rb") as f:
# 		pip = joblib.load(f)

# 	with open("./rel_models2/" + str(i) + ".pkl", "wb") as f:
# 		joblib.dump(pip, f)


# while len(joblib.load("errors.pkl")) > 0:
# 	errors = []
# 	for i in joblib.load("errors.pkl"):
# 		try:
# 			print("Training relation",i)
# 			start = time.time()
# 			train(i)

# 			print("Done. Execution time:", time.time() - start)
# 		except:
# 			print("\nERROR AT RELATION ID", i)
# 			errors.append(i)

# 	with open("errors.pkl", "wb") as f:
# 		joblib.dump(errors, f)

# print("\nAverage r2:			", np.average(r2, weights=weights))
# print("Average median error:	", np.average(mederr, weights=weights))
# print("Average mean error:		", np.average(meanerr, weights=weights))

print("\nTotal execution time:", time.time() - startall)


# param_grid = [
# 	{
# 		'vectorizer__min_df': [1, 0.002, 0.005, 0.01, 0.04],
# 		'vectorizer__max_df': [0.05, 0.1, 0.2, 0.7, 1.],
# 		'vectorizer__max_features': [2500, 10000, None],
# 		'estimator__C': [0.5, 1., 2.]
# 		# 'estimator__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']
# 		# 'estimator__max_iter': [1000, 2000]
# 	}
# ]

# param_freqs = {}
# for key in param_grid[0].keys():
# 	param_freqs[key] = defaultdict(int)

# x = [i for i in random.sample(fb.all_p(), 200) if fb.count_p(i) > 500 and fb.count_p(i) < 2500][:10]

# x = [338, 104, 376, 322, 473, 1, 426, 574, 552, 569]
# print("Training on {} relations:\n{}\n".format(len(x), x))
# ranges = [50] + list(range(100, 1600, 100))

# for index in range(len(ranges) - 1):
# 	x = [i for i in fb.all_p() if fb.count_p(i) > ranges[index] and fb.count_p(i) <= ranges[index+1]]
# 	np.random.shuffle(x)
# 	rels_in_range = 0
# 	rels_chosen = []

# 	while rels_in_range<1000:
# 		r = x.pop()
# 		rels_in_range += fb.count_p(r)
# 		rels_chosen.append(r)

# 	for i in rels_chosen:
# 		train(i)

# 	print("Range {} - {}".format(ranges[index], ranges[index+1]))

# 	r2_final.append(np.average(r2, weights=weights))
# 	mederr_final.append(np.average(mederr, weights=weights))
# 	meanerr_final.append(np.average(meanerr, weights=weights))
# 	weights_final.append(sum(weights))

# 	r2 = []
# 	mederr = []
# 	meanerr = []
# 	weights = []

# print("r2: {}\n r2 avg {}\n\nmederr: {}\n, avg {}\n\nmeanerr: {}\n, avg {}\n".format(r2_final, np.average(r2_final, weights=weights_final), mederr_final, np.average(mederr_final, weights=weights_final), meanerr_final, np.average(meanerr_final, weights=weights_final)))



# for i in x:
# 	# train(i)
# 	ad += fb.count_p(i)

# print(len(x))

# print(param_freqs)






# train(0, False)
# a,b = generate_sets(3, True)
# print(a.shape, b.shape)




