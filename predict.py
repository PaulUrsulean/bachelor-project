import tensorflow as tf
import numpy as np
import trident
import os

batch_size = 50

# Computes the rank of a translation head+rel, by checking how close
# the vector is to the actual target, in comparison with other embeddings.
def get_rank(translation, target_index, entities):

	distances = tf.norm(entities-translation, ord=1, axis=1)

	values, indices = tf.nn.top_k(distances, batch_size)

	values = tf.reverse(values, [0])
	indices = tf.reverse(indices, [0])
	rank = tf.where(tf.equal(indices, target_index))

	return tf.cast(rank[0][0], tf.int32)

def rank_translations(model_dir, kb_dir):
	with tf.Session() as sess:

		# Load model file
		saver = tf.train.import_meta_graph(model_dir + '/model-100.meta/')
		saver.restore(sess, tf.train.latest_checkpoint(model_dir))
		graph = tf.get_default_graph()

		# Load embeddings from model file
		emb_e, emb_r = tf.trainable_variables()

		# Initialize batcher
		batcher = trident.Batcher(kb_dir, batch_size, 1) # can add valid, train or test to determine which dataset to use.
		batcher.start()
		batch = batcher.getbatch()

		# Get relations from batch
		heads		= tf.convert_to_tensor(batch[0], tf.int64)
		relations 	= tf.convert_to_tensor(batch[1], tf.int64)
		tails		= tf.convert_to_tensor(batch[2], tf.int64)

		# Get embeddings for each relation element
		heads_embedded	= tf.gather(emb_e, heads)
		rel_embedded	= tf.gather(emb_r, relations)
		tails_embedded	= tf.gather(emb_e, tails)

		# Determine translations and targets depending on the 'forward' boolean
		translations_forward	= heads_embedded + rel_embedded
		translations_backward	= tails_embedded - rel_embedded

		# Make a vector of indices
		ranks_forward = tf.cast(tf.linspace(0., float(batch_size-1), batch_size), tf.int32)
		ranks_backward= tf.cast(tf.linspace(0., float(batch_size-1), batch_size), tf.int32)

		# Map the get_rank function to each index
		ranks_forward = tf.map_fn(lambda index: get_rank(translations_forward[index], index, tails_embedded), ranks_forward)
		ranks_backward= tf.map_fn(lambda index: get_rank(translations_backward[index], index, heads_embedded), ranks_backward)

		# Get mean rank
		mean_rank_forward = tf.reduce_mean(tf.cast(ranks_forward, tf.float32))
		mean_rank_backward= tf.reduce_mean(tf.cast(ranks_backward, tf.float32))

		return sess.run([mean_rank_forward, mean_rank_backward])

f, b = rank_translations("./models_fb", "./fb15k")
print("The mean rank forward is: {}\nThe mean rank backward is: {}".format(f, b))
