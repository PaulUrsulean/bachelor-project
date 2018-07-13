import tensorflow as tf
import numpy as np
from sklearn.externals import joblib
import trident
import pickle
from collections import defaultdict
from sklearn.pipeline import Pipeline

from custom import TripleTransformer
from sklearn.svm import SVR

import time

# import train_bow as bow

def get_relations(kb_dir, rel_ids=None):

	if isinstance(rel_ids, int):
		rel_ids = [rel_ids]

	batcher = trident.Batcher(kb_dir, 200, 1)
	batcher.start()
	batch = batcher.getbatch()


	heads = np.empty(0, dtype=np.uint64)
	rels = np.empty(0, dtype=np.uint64)
	tails = np.empty(0, dtype=np.uint64)



	while batch is not None:

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


	return heads.astype(np.int64), rels.astype(np.int64), tails.astype(np.int64)

def get_mean_rank(model_dir, kb_dir, rel_ids=None):
	with tf.Session() as sess:

		# Load model file
		saver = tf.train.import_meta_graph(model_dir + '/model-100.meta/')
		saver.restore(sess, tf.train.latest_checkpoint(model_dir))
		graph = tf.get_default_graph()

		# Load embeddings from model file
		emb_e, emb_r = tf.trainable_variables()

		heads, rels, tails = get_relations(kb_dir, rel_ids)

		n = len(heads)
		if n == 0:
			return


		# Separate subjects, predicates, objects, convert to tensors.
		t_heads		 = tf.convert_to_tensor(heads, tf.int64)
		t_rels 		 = tf.convert_to_tensor(rels, tf.int64)
		t_tails		 = tf.convert_to_tensor(tails, tf.int64)

		# Get embeddings for each relation element
		heads_embedded	= tf.gather(emb_e, t_heads)
		rels_embedded	= tf.gather(emb_r, t_rels)
		tails_embedded	= tf.gather(emb_e, t_tails)

		# Determine translations and targets depending on the 'forward' boolean
		translations_forward	= heads_embedded + rels_embedded
		translations_backward	= tails_embedded - rels_embedded

		# Make a vector of indices
		# Two different assignments so that ranks_forward and ranks_backward don't refer to the same object
		ranks_forward	= tf.cast(tf.linspace(0., float(n-1), n), tf.float32)
		ranks_backward 	= tf.cast(tf.linspace(0., float(n-1), n), tf.float32)

		# Map the get_translation_rank function to each index
		# ranks_forward = tf.map_fn(lambda index: get_translation_rank(translations_forward[index], index, tails_embedded), ranks_forward)
		# ranks_backward= tf.map_fn(lambda index: get_translation_rank(translations_backward[index], index, heads_embedded), ranks_backward)

		ranks_forward  	= tf.map_fn(lambda index: get_rank(heads, rels, tails, heads_embedded, rels_embedded, tails_embedded, index, True), ranks_forward)
		ranks_backward 	= tf.map_fn(lambda index: get_rank(heads, rels, tails, heads_embedded, rels_embedded, tails_embedded, index, False), ranks_backward)

		# Get mean rank
		mean_rank_forward  	= tf.reduce_mean(tf.cast(ranks_forward, tf.float32))
		mean_rank_backward 	= tf.reduce_mean(tf.cast(ranks_backward, tf.float32))

		mean_rank_forward, mean_rank_backward = sess.run([mean_rank_forward, mean_rank_backward])

		# a = []

		# for i in range(100):
		# 	a.append(sess.run(get_rank(heads, rels, tails, heads_embedded, rels_embedded, tails_embedded, tf.convert_to_tensor(i), True)))

		# for i in range(len(a)-1):
		# 	print(np.all(a[i].argsort() == a[i+1].argsort()))


	return mean_rank_forward, mean_rank_backward

# Sequence model, create sentence, work at character level, sentence is 3 elems of triple, lstm

# Computes the rank of a translation head+rel, by checking how close
# the vector is to the actual target, in comparison with other embeddings.
def get_translation_rank(translation, target_index, entities, source_index=None):

	distances = tf.norm(entities-translation, ord=1, axis=1)

	values, indices = tf.nn.top_k(distances, tf.size(distances))

	# Use source_index and evaluate (source_index, entities[i]) for each i, get ranking
	# Maybe need to find a way to wrap the real model in a tensorflow wrapper, so that I don't
	# have to translate from tf to real values here, since I am in an outside function

	values = tf.reverse(values, [0])
	indices = tf.reverse(indices, [0])
	rank = tf.where(tf.equal(indices, target_index))

	if source_index is not None:
		return values, indices

	return tf.cast(rank[0][0], tf.int32)

def get_rank(h, r, t, emb_h, emb_r, emb_t, i, forward=True, alpha=1):

	# Cast index i to an int. It is originally of type float because it may be replaced
	# with a float by the tf.map_fn function, wherein types need to match
	i = tf.cast(i, tf.int32)

	# TransE
	translation = emb_h[i] + emb_r[i]	if forward else emb_t[i] - emb_r[i]

	distances 	= emb_t - translation 	if forward else emb_h - translation
	distances	= tf.norm(distances, ord=1, axis=1)

	values, indices = tf.nn.top_k(distances, tf.size(distances))
	values 	= tf.reverse(values, [0])
	indices	= tf.reverse(indices, [0])



	# Labels
	pipe = Pipeline([('vectorizer', TripleTransformer(vectorizer=joblib.load("vectorizer.pkl"))), ('estimator', joblib.load("model.pkl"))])

	label_values, label_indices = apply_model(h, r, t, i, forward, alpha, pipe)

	if alpha != 1:
		return tf.cast(tf.where(tf.equal(label_indices, i))[0][0], tf.float32)
	else:
		return tf.cast(tf.where(tf.equal(indices, i))[0][0], tf.float32)


	# Merging the two
	avg_values, avg_indices = merge_rankings(indices, label_indices, i, alpha)

	# return avg_indices

	rank = tf.where(tf.equal(avg_indices, i))
	return tf.cast(rank[0][0], tf.int32)



def apply_model(h, r, t, i, forward, alpha, predictor):

	if alpha==1:
		return None, None


	if forward:

		# POSSIBLE THAT I DID IT WRONG HERE
		fun = lambda i: pipe.predict(np.asarray([np.repeat(h[i], len(h)), np.repeat(r[i], len(r)), t], np.int64).transpose())

		# fun = lambda i: pipe.predict(np.asarray([h, r, [t[i]] * len(t)], np.int64).transpose())

	else:

		fun = lambda i: pipe.predict(np.asarray([h, np.repeat(r[i], len(r)), np.repeat(t[i], len(t))], np.int64).transpose())

	predicted = tf.py_func(fun, [i], tf.float64)
	# return predicted

	return tf.nn.top_k(predicted, tf.size(predicted))


def merge_rankings(a, b, target_index, alpha):

	if b is None:
		return None, a

	# print(a.dtype, b.dtype)

	a = tf.to_float(a)
	b = tf.to_float(b)


	alpha = tf.to_float(tf.convert_to_tensor(alpha))

	# merged = []

	ranks = tf.linspace(0., tf.to_float(tf.size(a)-1), tf.size(a))

	ranks = tf.map_fn(lambda i: average_element(a, b, i, alpha), ranks)

	# for i in range(batch_size):
	# 	a_i = tf.to_float(tf.where(tf.equal(a, i))[0][0])
	# 	b_i = tf.to_float(tf.where(tf.equal(b, i))[0][0])

	# 	merged.append(tf.scalar_mul(alpha, a_i) + tf.scalar_mul(1-alpha, b_i))

	# v, i = tf.nn.top_k(tf.stack(merged), batch_size)

	v, i = tf.nn.top_k(ranks, tf.size(ranks))

	return tf.reverse(v, [0]), tf.reverse(i, [0])


def average_element(a, b, i, alpha):

	# print(tf.where(tf.equal(a, i)))

	a_i = tf.to_float(tf.where(tf.equal(a, i))[0][0])
	# print(a_i.dtype)
	b_i = tf.to_float(tf.where(tf.equal(b, i))[0][0])
	# print(b_i.dtype)

	# print(alpha.dtype)

	c_i = tf.scalar_mul(alpha, a_i) + tf.scalar_mul(1-alpha, b_i)

	# print(c_i.dtype)

	return c_i






# print(get_mean_rank("./models_fb", "./fb15k", 0))

# TO DO:

# TRY TransE on lubm1, both random and same relation
# Do the comparison on all entities, not just 50

# f, b = get_mean_rank("./models", "./lubm1", 3)
# print("The mean rank forward is: {}\nThe mean rank backward is: {}".format(f, b))

# When taking IDS, I think i'm comparing only to the list of IDs, not whole set of rels.

start = time.time()

# get_mean_rank("./models_fb", "./fb15k", ids)
f, b = get_mean_rank("./models_fb", "./fb15k", 0)
print("The mean rank forward is: {}\nThe mean rank backward is: {}".format(f, b))

print("Execution time:", time.time() - start)




