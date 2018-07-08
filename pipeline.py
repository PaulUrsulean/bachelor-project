import tensorflow as tf
import numpy as np
import trident
import pickle
from collections import defaultdict

import train_bow as bow

batch_size = 200

# Sequence model, create sentence, work at character level, sentence is 3 elems of triple, lstm

# Computes the rank of a translation head+rel, by checking how close
# the vector is to the actual target, in comparison with other embeddings.
def get_translation_rank(translation, target_index, entities, source_index=None):

	distances = tf.norm(entities-translation, ord=1, axis=1)

	values, indices = tf.nn.top_k(distances, batch_size)

	# Use source_index and evaluate (source_index, entities[i]) for each i, get ranking
	# Maybe need to find a way to wrap the real model in a tensorflow wrapper, so that I don't
	# have to translate from tf to real values here, since I am in an outside function

	values = tf.reverse(values, [0])
	indices = tf.reverse(indices, [0])
	rank = tf.where(tf.equal(indices, target_index))

	if source_index is not None:
		return values, indices

	return tf.cast(rank[0][0], tf.int32)

def get_rank(h, r, t, emb_h, emb_r, emb_t, i, forward=True):


	# TransE
	translation = emb_h[i] + emb_r[i]	if forward else emb_t[i] - emb_r[i]

	distances 	= emb_t - translation 	if forward else emb_h - translation
	distances	= tf.norm(distances, ord=1, axis=1)

	values, indices = tf.nn.top_k(distances, batch_size)
	values 	= tf.reverse(values, [0])
	indices	= tf.reverse(indices, [0])

	# Labels
	label_values, label_indices = apply_model(h, r, t, i, forward)
	# return apply_model(h, r, t, i, forward)

	# return indices, label_indices

	# return tf.cast(tf.where(tf.equal(indices, i))[0][0], tf.int32)

	# Merging the two
	avg_values, avg_indices = merge_rankings(indices, label_indices, i)

	# return avg_indices

	rank = tf.where(tf.equal(avg_indices, i))
	return tf.cast(rank[0][0], tf.int32)


def apply_model(h, r, t, i, forward):


	model, vectorizer = bow.get_model()

	if forward:

		fun = lambda i: model.predict(bow.feature_vector(vectorizer, h, r, [t[i]] * len(t)))

	else:
		fun = lambda i: model.predict(bow.feature_vector(vectorizer, [h[i]] * len(h), r, t))

	predicted = tf.py_func(fun, [i], tf.float64)
	# return predicted

	return tf.nn.top_k(predicted, batch_size)


def merge_rankings(a, b, target_index, alpha = 1):

	if alpha == 1:
		return [], a

	# print(a.dtype, b.dtype)

	a = tf.to_float(a)
	b = tf.to_float(b)


	alpha = tf.to_float(tf.convert_to_tensor(alpha))

	# merged = []

	ranks = tf.linspace(0., float(batch_size-1), batch_size)

	ranks = tf.map_fn(lambda i: average_element(a, b, i, alpha), ranks)

	# for i in range(batch_size):
	# 	a_i = tf.to_float(tf.where(tf.equal(a, i))[0][0])
	# 	b_i = tf.to_float(tf.where(tf.equal(b, i))[0][0])

	# 	merged.append(tf.scalar_mul(alpha, a_i) + tf.scalar_mul(1-alpha, b_i))

	# v, i = tf.nn.top_k(tf.stack(merged), batch_size)

	v, i = tf.nn.top_k(ranks, batch_size)

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


def get_mean_rank(model_dir, kb_dir, test_rel_id=None):
	with tf.Session() as sess:

		# Load model file
		saver = tf.train.import_meta_graph(model_dir + '/model-100.meta/')
		saver.restore(sess, tf.train.latest_checkpoint(model_dir))
		graph = tf.get_default_graph()

		# Load embeddings from model file
		emb_e, emb_r = tf.trainable_variables()

		# Initialize batcher
		batcher = trident.Batcher(kb_dir, batch_size, 1)
		batcher.start()
		batch = batcher.getbatch()

		if test_rel_id is not None and isinstance(test_rel_id, int):

			heads 	= []
			rels 	= []
			tails 	= []

			while len(rels) <= batch_size:

				indices	 = [index for index, elem in enumerate(batch[1]) if elem == test_rel_id]

				heads 	+= [batch[0][i] for i in indices]
				rels 	+= [batch[1][i] for i in indices]
				tails 	+= [batch[2][i] for i in indices]

				batch 	 = batcher.getbatch()

			heads = heads[:batch_size]
			rels = rels[:batch_size]
			tails = tails[:batch_size]

			# return heads[:batch_size], rels[:batch_size], tails[:batch_size]

		else:

			heads 		= batch[0]
			rels 		= batch[1]
			tails 		= batch[2]

		# Separate subjects, predicates, objects, convert to tensors.
		t_heads		 = tf.convert_to_tensor(heads[:batch_size], tf.int64)
		t_rels 		 = tf.convert_to_tensor(rels[:batch_size], tf.int64)
		t_tails		 = tf.convert_to_tensor(tails[:batch_size], tf.int64)

		# Get embeddings for each relation element
		heads_embedded	= tf.gather(emb_e, t_heads)
		rels_embedded	= tf.gather(emb_r, t_rels)
		tails_embedded	= tf.gather(emb_e, t_tails)

		# Determine translations and targets depending on the 'forward' boolean
		translations_forward	= heads_embedded + rels_embedded
		translations_backward	= tails_embedded - rels_embedded

		# Make a vector of indices
		# Two different assignments so that ranks_forward and ranks_backward don't refer to the same object
		ranks_forward	= tf.cast(tf.linspace(0., float(batch_size-1), batch_size), tf.int32)
		ranks_backward 	= tf.cast(tf.linspace(0., float(batch_size-1), batch_size), tf.int32)

		# # Map the get_translation_rank function to each index
		# ranks_forward = tf.map_fn(lambda index: get_translation_rank(translations_forward[index], index, tails_embedded), ranks_forward)
		# ranks_backward= tf.map_fn(lambda index: get_translation_rank(translations_backward[index], index, heads_embedded), ranks_backward)

		ranks_forward  	= tf.map_fn(lambda index: get_rank(heads, rels, tails, heads_embedded, rels_embedded, tails_embedded, index, True), ranks_forward)
		ranks_backward 	= tf.map_fn(lambda index: get_rank(heads, rels, tails, heads_embedded, rels_embedded, tails_embedded, index, False), ranks_backward)

		# Get mean rank
		mean_rank_forward  	= tf.reduce_mean(tf.cast(ranks_forward, tf.float32))
		mean_rank_backward 	= tf.reduce_mean(tf.cast(ranks_backward, tf.float32))

		mean_rank_forward, mean_rank_backward = sess.run([mean_rank_forward, mean_rank_backward])


		# return(sess.run(get_rank(heads, rels, tails, heads_embedded, rels_embedded, tails_embedded, tf.convert_to_tensor(0), True)))
		# a = np.empty([0, batch_size])
		# for i in range(batch_size):
		# 	a = np.append(a, [sess.run(get_rank(heads, rels, tails, heads_embedded, rels_embedded, tails_embedded, tf.convert_to_tensor(i), True))], axis=0)

		# return a

	return mean_rank_forward, mean_rank_backward

# print(get_mean_rank("./models_fb", "./fb15k", 0))



f, b = get_mean_rank("./models_fb", "./fb15k", 0)
print("The mean rank forward is: {}\nThe mean rank backward is: {}".format(f, b))




