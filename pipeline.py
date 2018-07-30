import tensorflow as tf
import numpy as np
from sklearn.externals import joblib
import trident
import pickle
from collections import defaultdict
from sklearn.pipeline import Pipeline

from custom import TripleTransformer
from sklearn.svm import SVR
import os

import time

# import train_bow as bow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_relations(kb_dir, rel_ids, n_triples):

	batch_size = 100

	if isinstance(rel_ids, int):
		rel_ids = [rel_ids]

	batcher = trident.Batcher(kb_dir, batch_size, 1)
	batcher.start()
	batch = batcher.getbatch()

	heads = np.empty(0, dtype=np.uint64)
	rels = np.empty(0, dtype=np.uint64)
	tails = np.empty(0, dtype=np.uint64)

	while batch is not None and len(heads) < n_triples:

		if rel_ids is not None:

			indices = []

			for rel_id in rel_ids:
				indices += [index for index, elem in enumerate(batch[1]) if elem == rel_id]

			heads = np.append(heads, [batch[0][i] for i in indices])
			rels = np.append(rels, [batch[1][i] for i in indices])
			tails = np.append(tails, [batch[2][i] for i in indices])


		else:


			heads = np.append(heads, batch[0])
			rels = np.append(rels, batch[1])
			tails = np.append(tails, batch[2])

		batch = batcher.getbatch()

	if n_triples > len(heads):
		n_triples = len(heads)

	X = np.column_stack((heads, rels, tails))

	np.random.shuffle(X)




	return X[:,0][:n_triples].astype(np.int64), X[:,1][:n_triples].astype(np.int64), X[:,2][:n_triples].astype(np.int64)

def get_mean_rank(model_dir, kb_dir, rel_ids=None, n_triples=float("inf"), use_transe=True):
	with tf.Session() as sess:

		# Load model file
		saver = tf.train.import_meta_graph(model_dir + '/model-100.meta/')
		saver.restore(sess, tf.train.latest_checkpoint(model_dir))
		graph = tf.get_default_graph()

		# Load embeddings from model file
		emb_e, emb_r = tf.trainable_variables()

		heads, rels, tails = get_relations(kb_dir, rel_ids, n_triples)

		# if rel_ids is not None:
		# 	rel_indices = np.argwhere(rels == rel_ids)

		X = np.column_stack((heads, rels, tails))

		n = len(heads)
		if n == 0:
			return


		# Get embeddings for each relation element
		heads_embedded	= tf.gather(emb_e, heads)
		rels_embedded	= tf.gather(emb_r, rels)
		tails_embedded	= tf.gather(emb_e, tails)

		# Determine translations and targets depending on the 'forward' boolean
		translations_forward	= heads_embedded + rels_embedded
		translations_backward	= tails_embedded - rels_embedded

		# Make a vector of indices
		# Two different assignments so that ranks_forward and ranks_backward don't refer to the same object
		ranks_forward	= tf.cast(tf.linspace(0., float(n-1), n), tf.float32)
		ranks_backward 	= tf.cast(tf.linspace(0., float(n-1), n), tf.float32)

		# # Map the get_translation_rank function to each index
		# ranks_forward = tf.map_fn(lambda index: get_translation_rank(translations_forward[index], index, tails_embedded), ranks_forward)
		# ranks_backward= tf.map_fn(lambda index: get_translation_rank(translations_backward[index], index, heads_embedded), ranks_backward)

		ranks_forward_transe  	= tf.map_fn(lambda index: get_rank(heads, rels, tails, heads_embedded, rels_embedded, tails_embedded, index, True, True), ranks_forward)
		ranks_backward_transe 	= tf.map_fn(lambda index: get_rank(heads, rels, tails, heads_embedded, rels_embedded, tails_embedded, index, False, True), ranks_backward)

		ranks_forward_nlp  	= tf.map_fn(lambda index: get_rank(heads, rels, tails, heads_embedded, rels_embedded, tails_embedded, index, True, False), ranks_forward)
		ranks_backward_nlp 	= tf.map_fn(lambda index: get_rank(heads, rels, tails, heads_embedded, rels_embedded, tails_embedded, index, False, False), ranks_backward)

		return sess.run([tf.reduce_mean(ranks_forward_transe), tf.reduce_mean(ranks_backward_transe), tf.reduce_mean(ranks_forward_nlp), tf.reduce_mean(ranks_backward_nlp)])

		# Get mean rank
		mean_rank_forward  	= tf.reduce_mean(ranks_forward)
		mean_rank_backward 	= tf.reduce_mean(ranks_backward)

		# # ranks_forward, ranks_backward = sess.run([ranks_forward, ranks_backward])

		mean_rank_forward, mean_rank_backward = sess.run([mean_rank_forward, mean_rank_backward])

		# a=[]

		# for i in range(2):
		# 	print(sess.run(apply_model(heads, rels, tails, i, True)))

		# print(np.argsort(a[0]), np.argsort(a[1]))

	# label_ranks_f, label_ranks_b = rank_labels(X)

	# return [ranks_forward, label_ranks_f], [ranks_backward, label_ranks_b]

	return mean_rank_forward, mean_rank_backward

# Sequence model, create sentence, work at character level, sentence is 3 elems of triple, lstm

def get_rank(h, r, t, emb_h, emb_r, emb_t, i, forward=True, use_transe=True):

	# Cast index i to an int. It is originally of type float because it may be replaced
	# with a float by the tf.map_fn function, wherein types need to match
	i = tf.cast(i, tf.int32)

	# TransE
	translation = emb_h[i] + emb_r[i]	if forward else emb_t[i] - emb_r[i]

	distances 	= emb_t - translation 	if forward else emb_h - translation
	distances	= tf.norm(distances, ord=1, axis=1)

	_, transe_indices = tf.nn.top_k(distances, tf.size(distances))
	transe_indices	= tf.reverse(transe_indices, [0])



	# Labels

	_, label_indices = apply_model(h, r, t, i, forward)

	indices = transe_indices if use_transe else label_indices

	return tf.cast(tf.where(tf.equal(indices, i))[0][0], tf.float32)

# pipe = Pipeline([('vectorizer', TripleTransformer(vectorizer=joblib.load("vectorizer.pkl"))), ('estimator', joblib.load("model.pkl"))])

def load_rel_model(i):

	if isinstance(i, int) or isinstance(i, np.int32) or isinstance(i, np.int64):

		with open("./rel_models/" + str(i) + ".pkl", "rb") as f:
			v,m = joblib.load(f)

	elif isinstance(i, str):
		v, m = joblib.load(i)
	return Pipeline([('vectorizer', TripleTransformer(vectorizer=v)), ('estimator', m)])

def load_all_models():
	for i in np.unique(trident.Db("fb15k").all_p()):
		dicc[i] = joblib.load("./rel_models/" + str(i) + ".pkl")

def get_model(i):
	v,m = dicc[i]
	return Pipeline([('vectorizer', TripleTransformer(vectorizer=v)), ('estimator', m)])

def lazy_model_dicc(i):
	if i not in dicc:
		dicc[i] = joblib.load("./rel_models/" + str(i) + ".pkl")

	v,m = dicc[i]

	return Pipeline([('vectorizer', TripleTransformer(vectorizer=v)), ('estimator', m)])



def apply_model(h, r, t, i, forward):

	if forward:
		fun = lambda i: lazy_model_dicc(r[i]).predict(np.column_stack((np.repeat(h[i], len(h)), np.repeat(r[i], len(r)), t)))
		# fun = lambda i: pipe.predict(np.column_stack((np.repeat(h[i], len(h)), np.repeat(r[i], len(r)), t)))

	else:
		fun = lambda i: lazy_model_dicc(r[i]).predict(np.column_stack((h, np.repeat(r[i], len(r)), np.repeat(t[i], len(t)))))
		# fun = lambda i: pipe.predict(np.column_stack((h, np.repeat(r[i], len(r)), np.repeat(t[i], len(t)))))

	predicted = tf.py_func(fun, [i], tf.float64)
	# return predicted

	return tf.nn.top_k(predicted, tf.size(predicted))


start = time.time()

fb = trident.Db("fb15k")

n_triples = 100

ranges = [50, 1500]

dicc = {}

# ids = [1, 2, 10, 13, 14, 16, 18, 19, 22, 23, 24, 25, 26, 28, 30, 31, 33, 37, 38, 39, 43, 44, 45, 47, 48, 51, 52, 54, 61, 66, 77, 78, 79, 80, 85, 86, 89, 92, 99]

ids = [i for i in np.unique(fb.all_p()) if fb.count_p(i) > ranges[0] and fb.count_p(i) <= ranges[1]]
# ids = np.unique(fb.all_p())
# np.random.shuffle(ids)

# v = np.vectorize(lambda i: fb.count_p(i))
# print(max(v(ids)), min(v(ids)))

# i = 221

# pipe = load_rel_model("pipe.pkl")

# get_mean_rank("./models_fb", "./fb15k", rel_ids = None,use_transe=True, n_triples = 5)

print("Range: {} - {}".format(ranges[0], ranges[1]))

f1, b1, f2, b2 = get_mean_rank("./models_fb", "./fb15k", rel_ids=ids,use_transe=True, n_triples = n_triples)
print("\nTransE:\nForward rank: {}\nBackward rank: {}\n".format(f1, b1))
print("\nMy technique:\nForward rank: {}\nBackward rank: {}\n".format(f2, b2))


# # # print("Labels forward rank: {}\nLabels backward rank: {}".format(np.mean(f[1]), np.mean(b[1])))


# print("Execution time:", time.time() - start)

# def rank_labels(X):
# 	forward = np.empty([1, len(X)], np.int32)
# 	back 	= np.empty([1, len(X)], np.int32)

# 	forward[0,:] 	= np.linspace(0, len(X)-1, len(X), dtype=np.int32)
# 	back[0,:]		= np.linspace(0, len(X)-1, len(X), dtype=np.int32)

# 	forward = np.apply_along_axis(lambda i: rank_single(X, i, True), 0, forward)
# 	back 	= np.apply_along_axis(lambda i: rank_single(X, i, False), 0, back)

# 	return forward, back

# def rank_single(X, i, forward=True):

# 	X[:,1] = X[i,1]

# 	if forward:
# 		X[:,0] = X[i,0]
# 	else:
# 		X[:,2] = X[i,2]

# 	where = np.where(np.argsort(pipe.predict(X))[::-1] == i)

# 	return where[0][0]



# # Computes the rank of a translation head+rel, by checking how close
# # the vector is to the actual target, in comparison with other embeddings.
# def get_translation_rank(translation, target_index, entities):

# 	distances = tf.norm(entities-translation, ord=1, axis=1)

# 	values, indices = tf.nn.top_k(distances, tf.size(distances))

# 	# Use source_index and evaluate (source_index, entities[i]) for each i, get ranking
# 	# Maybe need to find a way to wrap the real model in a tensorflow wrapper, so that I don't
# 	# have to translate from tf to real values here, since I am in an outside function

# 	values = tf.reverse(values, [0])
# 	indices = tf.reverse(indices, [0])
# 	rank = tf.where(tf.equal(indices, target_index))

# 	return tf.cast(rank[0][0], tf.int32)

# def merge_rankings(a, b, target_index, alpha):

# 	if b is None:
# 		return None, a

# 	# print(a.dtype, b.dtype)

# 	a = tf.to_float(a)
# 	b = tf.to_float(b)


# 	alpha = tf.to_float(tf.convert_to_tensor(alpha))

# 	# merged = []

# 	ranks = tf.linspace(0., tf.to_float(tf.size(a)-1), tf.size(a))

# 	ranks = tf.map_fn(lambda i: average_element(a, b, i, alpha), ranks)

# 	# for i in range(batch_size):
# 	# 	a_i = tf.to_float(tf.where(tf.equal(a, i))[0][0])
# 	# 	b_i = tf.to_float(tf.where(tf.equal(b, i))[0][0])

# 	# 	merged.append(tf.scalar_mul(alpha, a_i) + tf.scalar_mul(1-alpha, b_i))

# 	# v, i = tf.nn.top_k(tf.stack(merged), batch_size)

# 	v, i = tf.nn.top_k(ranks, tf.size(ranks))

# 	return tf.reverse(v, [0]), tf.reverse(i, [0])


# def average_element(a, b, i, alpha):

# 	# print(tf.where(tf.equal(a, i)))

# 	a_i = tf.to_float(tf.where(tf.equal(a, i))[0][0])
# 	# print(a_i.dtype)
# 	b_i = tf.to_float(tf.where(tf.equal(b, i))[0][0])
# 	# print(b_i.dtype)

# 	# print(alpha.dtype)

# 	c_i = tf.scalar_mul(alpha, a_i) + tf.scalar_mul(1-alpha, b_i)

# 	# print(c_i.dtype)

# 	return c_i