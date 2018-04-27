import tensorflow as tf
import numpy as np
import trident
import os

batch_size = 50

def get_rank(translation, target_index, tails_embedded):

	distances = tf.norm(tails_embedded-translation, ord=1, axis=1)

	values, indices = tf.nn.top_k(distances, batch_size)

	values = tf.reverse(values, [0])
	indices = tf.reverse(indices, [0])
	rank = tf.where(tf.equal(indices, target_index))

	return tf.cast(rank[0][0], tf.int32)


with tf.Session() as sess:

	# Load model file
	saver = tf.train.import_meta_graph('./models/model-100.meta/')
	saver.restore(sess, tf.train.latest_checkpoint('./models/'))
	graph = tf.get_default_graph()

	# Load embeddings from model file
	emb_e, emb_r = tf.trainable_variables()

	# Initialize batcher
	inputdir = "./lubm1"
	batcher = trident.Batcher(inputdir, batch_size, 1) # can add valid, train or test to determine which dataset to use.
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

	# Get translations by adding head and relation embeddings
	translations = heads_embedded + rel_embedded

	# Make a vector of indices
	ranks = tf.cast(tf.linspace(0., float(batch_size-1), batch_size), tf.int32)

	# Map the get_rank function to each index
	ranks = tf.map_fn(lambda index: get_rank(translations[index], index, tails_embedded), ranks)

	# Get mean rank
	mean_rank = tf.reduce_mean(tf.cast(ranks, tf.float32))

	print(sess.run(mean_rank))
