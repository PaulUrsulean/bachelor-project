# This script converts identifiers from the fb15k .nt file on the repository
# To their respective labels from the full freebase. It needs to be run on DAS4

import sys
sys.path.insert(0, '/home/jurbani/trident/build')

import trident
import numpy as np
import pickle

fb = trident.Db("./fb15k")
big= trident.Db("/home/jurbani/data/motherkb-trident")

def id_to_label(Id, entity=True, language="en"):
	# Use lookup_relstr only for fb15k, motherkb only uses lookup_str
	pseudo = fb.lookup_str(Id) if entity else fb.lookup_relstr(Id)

	# If the label from fb15k is an identifier
	if pseudo[0:4] == "</m/":

		# Creat the URI for access in motherkb
		uri = "<http://rdf.freebase.com/ns/m." + pseudo[4:-1] + ">"
		
		# Lookup URI in motherkb
		mother_id = big.lookup_id(uri)
		mother_label_ids = big.o(mother_id, 6)

		for label_id in mother_label_ids:
			label = big.lookup_str(label_id).split("@")
			if len(label) < 2:
				raise RuntimeError("Language delimiter not detected")
			if label[1] == language:
				return label[0]

	# Return empty string if nothing found
	return ""

entities = list(set().union(fb.all_s(), fb.all_o()))
relations= fb.all_p()

e_id_to_english_label = np.vectorize(lambda Id: id_to_label(Id))
r_id_to_english_label = np.vectorize(lambda Id: id_to_label(Id, entity=False))

with open("entity_labels.txt", "wb") as f:
	pickle.dump(e_id_to_english_label(entities), f)

with open("relation_labels.txt", "wb") as f:
	pickle.dump(r_id_to_english_label(relations), f)

