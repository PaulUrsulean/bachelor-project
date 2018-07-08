import tensorflow as tf
import trident
import train_bow as bow
import pipeline as pipe
import numpy as np



def f(batch_size, test_rel_id=None):

	# Initialize batcher
	batcher = trident.Batcher("fb15k", batch_size, 1)
	batcher.start()
	batch = batcher.getbatch()

	model, vectorizer = bow.get_model()

	indices = np.linspace(0, batch_size-1, batch_size, dtype=np.int64)


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

	else:

		heads 		= batch[0]
		rels 		= batch[1]
		tails 		= batch[2]

	fun = lambda i: model.predict(bow.feature_vector(vectorizer, heads, rels, [tails[i]] * len(tails)))

	a = np.empty([0,batch_size])

	for i in range(batch_size):
		a = np.append(a, [fun(i)], axis=0)

	return a


print(f(10, 0) == pipe.get_mean_rank("./models_fb", "./fb15k", 0))


# if forward:

# 	fun = lambda i: model.predict(bow.feature_vector(vectorizer, h, r, [t[i]] * len(t)))

# else:
# 	fun = lambda i: model.predict(bow.feature_vector(vectorizer, [h[i]] * len(h), r, t))

# predicted = tf.py_func(fun, [i], tf.float64)


# return tf.nn.top_k(predicted, batch_size)