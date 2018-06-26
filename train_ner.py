# This script was meant to train a model on the triples
# based on the NER classifications, however I soon realised
# that this would not work. See more in ner.py

import sklearn as sk
import trident
import pickle
import numpy as np
from collections import defaultdict

TRAINING_SET_PERCENTAGE = 0.9


with open("entity_labels.txt", "rb") as f:
	labels = pickle.load(f)

with open("entity_recognition.txt", "rb") as f:
	entities = pickle.load(f)


fb = trident.Db("fb15k")

# # # ENTITIES ENCODING # # #

onehot_e = {}
ent = np.unique(list(entities.values()))

for i in range(len(ent)):
	onehot_e[ent[i]] = i

# # # # # # # # # # # # # ## #

# # # RELATIONS ENCODING # # #

onehot_r = {}
pred = fb.all_p()

for i in range(len(pred)):
	pred[i] = fb.lookup_relstr(pred[i])[1:-1].split("/")[-1]

pred = np.unique(pred)

for i in range(len(pred)):
	onehot_r[pred[i]] = i

# # # # # # # # # # # # # # # #


data = np.asarray(fb.all())

feature_vectors = []
targets = []


for triple in data:

	s = entities[fb.lookup_str(triple[0])]
	p = fb.lookup_relstr(triple[1])[1:-1].split("/")[-1]
	o = entities[fb.lookup_str(triple[2])]

	if s != "" and p != "" and o != "":
		features = np.zeros(len(ent) + len(pred))
		features[onehot_e[s]] = 1
		features[len(ent) + onehot_r[p]] = 1

		feature_vectors.append(features)

		target = np.zeros(len(ent))
		target[onehot_e[o]] = 1
		targets.append(target)

feature_vectors = np.asarray(feature_vectors)
targets = np.asarray(targets)

cutoff = int(len(feature_vectors) * TRAINING_SET_PERCENTAGE)
training_set 	= [feature_vectors[:cutoff], targets[:cutoff]]
test_set		= [feature_vectors[cutoff:], targets[cutoff:]]








		# print(s, p, o)

		# zeros = []
		
		# for i in range(len(features)):
		# 	if features[i] == 1:
		# 		zeros.append(i)

		# print(ent[zeros[0]], pred[zeros[1] - len(ent)], "\n")









# Make arrays one-hot encoded of the subject and object, try to predict the relationship

# heads = triples[:,0]
# rels  = triples[:,1]
# tails = triples[:,2]

# f_e = np.vectorize(lambda index: fb.lookup_str(index))
# f_r = np.vectorize(lambda index: fb.lookup_relstr(index)[1:-1].split("/")[-1])

# triples = [f_e(heads), f_r(rels), f_e(tails)]



