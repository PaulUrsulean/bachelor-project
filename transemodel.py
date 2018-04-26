import tensorflow as tf
from math import sqrt

class TranseModel:

    def readTensorFromTFRecords(graph, files):
        with graph.as_default():
            with tf.device('/cpu:0'):
                filename_queue = tf.train.string_input_producer(files)
                reader = tf.TFRecordReader()
                key, value = reader.read(filename_queue)
                features = tf.parse_single_example(value,
                    features={
                        'triple': tf.FixedLenFeature([3], tf.int64),
                    })
                triples = features['triple']
                return triples

    def readTensorFromCSV(graph, files):
        with graph.as_default():
            with tf.device('/cpu:0'):
                filename_queue = tf.train.string_input_producer(files)
                reader = tf.TextLineReader()
                key, value = reader.read(filename_queue)
                record_defaults = [[1], [1], [1]]
                col1, col2, col3 = tf.decode_csv(value, record_defaults=record_defaults)
                triples = tf.stack([col1, col2, col3])
                return triples

    def readTensorFromPlaceholder(graph):
        with graph.as_default():
            with tf.device('/cpu:0'):
                return tf.placeholder(tf.int32, shape=[None, 3])

    def readTensorsFromPlaceholder(graph):
        with graph.as_default():
            with tf.device('/cpu:0'):
                s = tf.placeholder(tf.int64, shape=[None])
                p = tf.placeholder(tf.int64, shape=[None])
                o = tf.placeholder(tf.int64, shape=[None])
                return (s, p, o)

    def getTrainer(graph, device, n, r, dim, batch_size, margin, learning_rate, train_heads, train_rels, train_tails):
        with graph.as_default():
            with tf.device('/cpu:0'):
                emb_e = tf.Variable(tf.random_uniform([n, dim], -6.0/sqrt(dim), 6.0/sqrt(dim))) # Embeddings of the entities
                emb_r = tf.Variable(tf.random_uniform([r, dim], -1.0, 1.0)) # Embeddings of the relations
                train_heads_neg = tf.random_uniform([batch_size], minval=0, maxval=n, dtype=tf.int64)
                train_tails_neg = tf.random_uniform([batch_size], minval=0, maxval=n, dtype=tf.int64)

        with graph.as_default():
            with tf.device(device):
                batch_heads_pos = tf.gather(emb_e, train_heads)
                batch_heads_neg = tf.gather(emb_e, train_heads_neg)
                batch_tails_pos = tf.gather(emb_e, train_tails)
                batch_tails_neg = tf.gather(emb_e, train_tails_neg)
                batch_rels = tf.gather(emb_r, train_rels)

        with graph.as_default():
            with tf.device(device):
                # L1 Distance head + rel, tail on the positive triples
                l1_sum_h2t_pos = tf.abs(batch_heads_pos + batch_rels - batch_tails_pos)
                l1_dist_pos = tf.reduce_sum(l1_sum_h2t_pos, 1)
                # L1 Distance head + rel, tail on the negative triples
                l1_sum_h2t_neg = tf.abs(batch_heads_pos + batch_rels - batch_tails_neg)
                l1_dist_neg = tf.reduce_sum(l1_sum_h2t_neg, 1)
                # L1 Distance tail - rel, head on the negative triples
                l1_sum_t2h_neg = tf.abs(batch_tails_pos - batch_rels - batch_heads_neg)
                l1_dist2_neg = tf.reduce_sum(l1_sum_t2h_neg, 1)
                # I consider only the positive value of the distance by picking the maximum with 0
                l1_t2h = tf.maximum(0.0, l1_dist_pos - l1_dist2_neg + margin)
                l1_h2t = tf.maximum(0.0, l1_dist_pos - l1_dist_neg + margin)
                loss = tf.reduce_sum(tf.concat([l1_h2t, l1_t2h],0))
                # Minimize using adaptive gradient descent
                optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

        return optimizer, loss, emb_e, emb_r
