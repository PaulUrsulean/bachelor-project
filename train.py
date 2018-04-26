from transemodel import TranseModel
import sys
import time
import tensorflow as tf
import numpy as np
import logging
import trident
import os

# Set logger
loglevel = 'INFO'
logger = logging.getLogger()
logger.setLevel(loglevel)
handler = logging.StreamHandler()
handler.setLevel(loglevel)
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

inputdir = sys.argv[1] # The (trident) KG
outputdir = sys.argv[2] # Where to dump the models
if not os.path.exists(outputdir):
    os.makedirs(outputdir)
modelfile = outputdir + '/model'

db = trident.Db(inputdir)
n = db.n_terms()
r = db.n_relations()
ntriples = db.n_triples()
db = None
dim = 50 # I want embeddings of size 50
batch_size = 1000
learning_rate = 0.1
epochs = 100
margin = 2
dump_after_n_epochs = 10
device = '/cpu:0' # Device to use for the computation

logger.info("# Entities: %d" % n)
logger.info("# Relations: %d" % r)
logger.info("# Training triples: %d" % ntriples)

graph = tf.Graph()
subjs, preds, objs = TranseModel.readTensorsFromPlaceholder(graph)
optimizer, loss, emb_e, emb_r = TranseModel.getTrainer(graph, device,
                                                       n,
                                                       r,
                                                       dim,
                                                       batch_size,
                                                       margin,
                                                       learning_rate,
                                                       subjs, preds, objs)

with graph.as_default():
    init_g = tf.global_variables_initializer()
    init_l = tf.local_variables_initializer()
    saver = tf.train.Saver()

# Running TransE
start_time = time.time()
session = tf.Session(graph=graph) # Start a session
init_g.run(session=session) # Initialize global vars
init_l.run(session=session) # Initialize local vars
coord = tf.train.Coordinator() # Default coordinator
threads = tf.train.start_queue_runners(session, coord) # Start queue runners
logger.info("Initialization time: %0.2f sec." % (time.time() - start_time))

# Let's do the training!
for epoch in range(epochs):
    start_time = time.time()
    batcher = trident.Batcher(inputdir, batch_size, 1) # can add valid, train or test to determine which dataset to use.
    batcher.start()
    losses = [] # Record the losses
    i = 0
    batch = batcher.getbatch()
    while batch is not None and len(batch[0]) == batch_size:
        _, l = session.run([optimizer, loss], feed_dict={subjs: batch[0], preds: batch[1], objs: batch[2]})
        losses.append(l)
        i += 1
        if i % 10000 == 0:
            logger.info("   Batch %d: Time so far %0.2f sec." % (time.time() - start_time))
        batch = batcher.getbatch()
    logger.info("Epoch: %d Loss: %0.2f Time: %0.2f sec." % (epoch, np.average(np.array(losses)), (time.time() - start_time)))
    if (epoch + 1) % dump_after_n_epochs == 0:
        saver.save(session, modelfile, global_step=epoch+1)
        logger.info("Model saved in " + modelfile)

# Make sure all threads are finished ...
coord.request_stop()
coord.join(threads)

# Close everything...
session.close()
