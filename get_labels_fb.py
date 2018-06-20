# This script converts identifiers from the fb15k .nt file on the repository
# To their respective labels from the full freebase. It needs to be run on DAS4

import sys
sys.path.insert(0, '/home/jurbani/trident/build')

import trident
import numpy as np
from collections import defaultdict
import pickle

fb = trident.Db("./fb15k")
big= trident.Db("/home/jurbani/data/motherkb-trident")

def ids_to_labels(Id_list, entity=True, language="en", descriptions=True):

	code_to_label = defaultdict(str)

	rel_id = 42 if descriptions else 6

	for Id in Id_list:

		# Use lookup_relstr only for fb15k, motherkb only uses lookup_str
		pseudo = fb.lookup_str(Id) if entity else fb.lookup_relstr(Id)

		# If the label from fb15k is an identifier
		if pseudo[0:4] == "</m/":

			# Create the URI for access in motherkb
			uri = "<http://rdf.freebase.com/ns/m." + pseudo[4:-1] + ">"
			
			# Lookup URI in motherkb
			mother_id = big.lookup_id(uri)
			mother_label_ids = big.o(mother_id, rel_id)

			for label_id in mother_label_ids:
				label = big.lookup_str(label_id).split("@")
				if len(label) < 2:
					raise RuntimeError("Language delimiter not detected")
				if label[1] == language:
					code_to_label[pseudo] = label[0][1:-1]

		# Insert nothing in the dict if nothing is found
		# Defaultdict defaults missing values to '' for (str).
	return code_to_label


entities = list(set().union(fb.all_s(), fb.all_o()))

# This operation will need to be performed on batches for actual datasets
# with open("entity_labels.txt", "wb") as f:
# 	pickle.dump(ids_to_labels(entities), f, protocol = pickle.HIGHEST_PROTOCOL)

with open("entity_descriptions.txt", "wb") as f:
	pickle.dump(ids_to_labels(entities, descriptions=True), f, protocol = pickle.HIGHEST_PROTOCOL)