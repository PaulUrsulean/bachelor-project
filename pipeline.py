import tensorflow as tf
import numpy as np
import trident
import pickle
from collections import defaultdict

batch_size = 200

# Sequence model, create sentence, work at character level, sentence is 3 elems of triple, lstm

# Computes the rank of a translation head+rel, by checking how close
# the vector is to the actual target, in comparison with other embeddings.
def get_translation_rank(translation, target_index, entities):

	distances = tf.norm(entities-translation, ord=1, axis=1)

	values, indices = tf.nn.top_k(distances, batch_size)

	# Here is where we should apply the labels regression, on the top k
	# distances found, and return an average rank

	values = tf.reverse(values, [0])
	indices = tf.reverse(indices, [0])
	rank = tf.where(tf.equal(indices, target_index))

	return tf.cast(rank[0][0], tf.int32)


def get_mean_rank(model_dir, kb_dir, labels_rank_function=None):
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

		# Separate subjects, predicates, objects, convert to tensors.
		heads		= tf.convert_to_tensor(batch[0], tf.int64)
		rels 		= tf.convert_to_tensor(batch[1], tf.int64)
		tails		= tf.convert_to_tensor(batch[2], tf.int64)

		# # # # # # # # # TransE # # # # # # # # #

		# Get embeddings for each relation element
		heads_embedded	= tf.gather(emb_e, heads)
		rels_embedded	= tf.gather(emb_r, rels)
		tails_embedded	= tf.gather(emb_e, tails)

		# Determine translations and targets depending on the 'forward' boolean
		translations_forward	= heads_embedded + rels_embedded
		translations_backward	= tails_embedded - rels_embedded

		# Make a vector of indices
		# Two different assignments so that ranks_forward and ranks_backward don't refer to the same object
		ranks_forward = tf.cast(tf.linspace(0., float(batch_size-1), batch_size), tf.int32)
		ranks_backward= tf.cast(tf.linspace(0., float(batch_size-1), batch_size), tf.int32)

		# Map the get_translation_rank function to each index
		ranks_forward = tf.map_fn(lambda index: get_translation_rank(translations_forward[index], index, tails_embedded), ranks_forward)
		ranks_backward= tf.map_fn(lambda index: get_translation_rank(translations_backward[index], index, heads_embedded), ranks_backward)

		# Get mean rank
		mean_rank_forward = tf.reduce_mean(tf.cast(ranks_forward, tf.float32))
		mean_rank_backward= tf.reduce_mean(tf.cast(ranks_backward, tf.float32))

		# # # # # # # # # #    # # # # # # # # # #

		# # # # # # # # # Labels # # # # # # # # #

		# Get labels for each triple element
		labels_h, labels_r, labels_t = get_labels_fb(batch)

		mean_rank_forward, mean_rank_backward = sess.run([mean_rank_forward, mean_rank_backward])		

	return mean_rank_forward, mean_rank_backward

def get_labels_fb(triples):
	with open("entity_labels.txt", "rb") as f:
		labels = pickle.load(f)

	db = trident.Db("fb15k")

	f_ent = np.vectorize(lambda index: labels[db.lookup_str(index)])
	f_rel = np.vectorize(lambda index: db.lookup_relstr(index)[1:-1].split("/")[-1])

	return [tf.convert_to_tensor(f_ent(triples[0])), tf.convert_to_tensor(f_rel(triples[1])), tf.convert_to_tensor(f_ent(triples[2]))]


f, b = get_mean_rank("./models_fb", "./fb15k", get_labels_fb)
print("The mean rank forward is: {}\nThe mean rank backward is: {}".format(f, b))





