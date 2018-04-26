import tensorflow as tf
import numpy as np
import trident
import os

def get_rank(translation, target_index, all_tails_e):

	# Change to L1
	distances = tf.norm(all_tails_e-translation, ord='euclidean', axis=1)

	values, indices = tf.nn.top_k(distances, 20)

	values = tf.reverse(values, [0])
	indices = tf.reverse(indices, [0])
	return values, indices, tf.where(tf.equal(indices, target_index))



with tf.Session() as sess:

	# Load model file
	saver = tf.train.import_meta_graph('./models/model-100.meta/')
	saver.restore(sess, tf.train.latest_checkpoint('./models/'))
	graph = tf.get_default_graph()

	# Load embeddings from model file
	emb_e, emb_r = tf.trainable_variables()

	# Initialize batcher
	inputdir = "./lubm1"
	batch_size = 20
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

	# translation = translations[0]

	# distances = tf.norm(tails_embedded-translation, ord='euclidean', axis=1)
	# values, indices = tf.nn.top_k(distances, 20)

	# rank = get_rank(translations[0], 0, tails_embedded)

	# look up how to do for loop in tf.

	values, indices, rank = get_rank(translations[3], 3, tails_embedded)


	print(sess.run([values, indices, rank, ]))



	# with graph.as_default():
	# 	with tf.device('/cpu:0'):
	# 		filename_queue = tf.train.string_input_producer("???")
	# 		reader = tf.TFRecordReader()
	# 		key, value = reader.read(filename_queue)
	# 		features = tf.parse_single_example(value,
	# 			features={
	# 				'triple': tf.FixedLenFeature([3], tf.int64),
	# 			})
	# 		triples = features['triple']
			# return triples

	# For each triple
	#

	# To get database:
	# make database again with adjustment Jacopo made
	# Once trained, throw away _batch and rename _batch_test to _batch