# Bag of words classifier with SVM

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

import trident
import pickle
import numpy as np
import nltk
from collections import defaultdict
from operator import itemgetter
import scipy.sparse as sp
from random import sample

TRAINING_SET_PERCENTAGE = 0.9

# IDEA:
# Currently I have descriptions for each entity in my slice of fb

# Make function that:
# 	- Takes a set of sentences (use 1st sentence of descriptions)
# 	- Creates a feature vector from n most common words (maybe sklearn has function for this)
# 	- Train using these feature-vectors. Also need to provide negative examples. To do this,
# pick random triples, check that they do not in fact exist, and feed to model
#	- Apply, to rankings generated by TransE, merge.

# What are the implementation details to this model?
# - Include relId into feature model?
# - This would be instead of making a model for each rel
# - Does this work in combination with one-hot?
# - What can I do to lessen training time?
# - How long should the thesis be?
# - Should I only do fb15k, or try to generalize to whole fb?
# - How to make model that chooses weights for combining?

# - Send email to Jacopo about results of combining rankings in the next 2 days.

# Technique for identifying useful features, remove as many features
# Focus on combining with TransE first


with open("entity_labels.txt", "rb") as f:
	labels = pickle.load(f)

with open("entity_descriptions.txt", "rb") as f:
	desc = pickle.load(f)

with open("short_descriptions.txt", "rb") as f:
	short_desc = pickle.load(f)

fb = trident.Db("fb15k")
d_get = np.vectorize(lambda index: short_desc[fb.lookup_str(index)])


def feature_vector(rel_id):
	triples, negative_triples = generate_sets(rel_id)


# Creates a model based on triples received, attempts to make predictions
def make_model(rel_id):

	vectorizer = TfidfVectorizer()

	triples, negative_triples = generate_sets(rel_id)

	train_pos = triples[:int(len(triples)*TRAINING_SET_PERCENTAGE)]
	train_neg = negative_triples[:int(len(negative_triples)*TRAINING_SET_PERCENTAGE)]

	test_pos = triples[int(len(triples)*TRAINING_SET_PERCENTAGE):]
	test_neg = negative_triples[int(len(negative_triples)*TRAINING_SET_PERCENTAGE):]


	s = np.concatenate((train_pos[:,0], train_neg[:,0]), axis=0)
	s_test = np.concatenate((test_pos[:,0], test_neg[:,0]), axis=0)

	o = np.concatenate((train_pos[:,2], train_neg[:,2]), axis=0)
	o_test = np.concatenate((test_pos[:,2], test_neg[:,2]), axis=0)

	y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
	y_test = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

	# print(len(train_pos), len(train_neg), len(test_pos), len(test_neg))

	ents = d_get(np.unique(np.append(s, o)))

	vectorizer.fit_transform(ents)
	
	# if n_rels > 1:
	# 	pass
	# 	# Create a feature vector that includes the relation

	# else:
		# Create a feature vector that is just the unique 
		# tokens of a sentence twice, one for each entity
	s = d_get(s)
	s_test = d_get(s_test)

	o = d_get(o)
	o_test = d_get(o_test)

	rel = sp.csr_matrix(np.ones([len(s), 1])*rel_id)
	rel_test = sp.csr_matrix(np.ones([len(s_test), 1])*rel_id)

	features = sp.hstack((rel, vectorizer.transform(s), vectorizer.transform(o)), format='csr')
	features_test = sp.hstack((rel_test, vectorizer.transform(s_test), vectorizer.transform(o_test)), format='csr')

	svr = SVR()

	y_pred = svr.fit(features, y).predict(features_test)


	return mean_squared_error(y_test, y_pred), mean_squared_error(y_test, np.random.rand(len(y_pred), 1))


	# print(features.shape, y.shape)
	# print(features_test.shape, y_test.shape)
	
def generate_sets(rel_id):
	positive = np.asarray(fb.os(rel_id))
	positive = [(tup[0], rel_id, tup[1]) for tup in positive]

	all_set = set(fb.all())

	negative = sample(all_set.difference(set(positive)), fb.count_p(rel_id))

	return np.asarray(positive), np.asarray(negative)

# make_model(0)
real, baseline = make_model(0)

print("The mean squared error for our estimator:", real)
print("The baseline mean squared error for a guessing estimator:", baseline)



# # Cleaning up descriptions, extracting the first sentence from each, writing to file

# short_desc = defaultdict(str)

# for key in list(desc.keys()):

# 	corrected = desc[key].replace("\\n", " ").replace("\\r", " ").replace("\\t", " ").replace('\\"', '"').replace("\\", "")

# 	sentence = nltk.sent_tokenize(corrected)

# 	short_desc[key] = sentence[0]

# 	if len(sentence[0]) <= 40 and " is " not in sentence[0] and " was " not in sentence[0] and " are " not in sentence[0] and " were " not in sentence[0]:
# 		short_desc[key] += sentence[1]

# 		if len(sentence[1]) < 5:
# 			short_desc[key] += sentence[2]

# with open("short_descriptions.txt", "wb") as f:
# 	pickle.dump(short_desc, f, protocol = pickle.HIGHEST_PROTOCOL)