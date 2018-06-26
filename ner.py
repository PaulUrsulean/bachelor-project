# Script used for entity recognition via Stanford NLP's Named
# Entity Recognizer. This approach was a dead end because it
# could only resolve about 60% of the entities, which means
# that only 36% of the triples could be properly evaluated.
# Additionally, the categories it produced were too abstract for my purposes.

import pickle
import trident
from stanfordcorenlp import StanfordCoreNLP
from collections import defaultdict
from operator import itemgetter

nlp = StanfordCoreNLP(r'/Users/paul/Desktop/bachelor-project/stanford-corenlp-full-2018-02-27/', lang="en")

with open("entity_labels.txt", "rb") as f:
	dicc = pickle.load(f)

fb = trident.Db("fb15k")

entities = list(set().union(fb.all_s(), fb.all_o()))
named_entities = defaultdict(str)

sample = fb.all()



def entity_recognition(label):
	ner = nlp.ner(dicc[label]);

	if not ner:
		return
	else:
		first = ner[0][1];

	consistent = True

	for tup in ner:
		if tup[1] != first or tup[1] == "O" or tup[1] == "MISC":
			consistent = False
			break

	if consistent:
		named_entities[label] = first
		return first

for e in list(set().union(fb.all_s(), fb.all_o())):
	entity_recognition(fb.lookup_str(e))


with open("entity_recognition.txt", "wb") as f:
	pickle.dump(named_entities, f, protocol = pickle.HIGHEST_PROTOCOL)


# for s, p, o in sample:

# 	s_ner = entity_recognition(fb.lookup_str(s))

# 	# Checked individually for efficiency purposes
# 	if s_ner is not None:
# 		o_ner = entity_recognition(fb.lookup_str(o))

# 		if o_ner is not None:
# 			rel = fb.lookup_relstr(p)[1:-1].split("/")[-1]

# 			sentence = [s_ner, rel, o_ner]
# 			frequencies[s_ner] += 1
# 			frequencies[rel] += 1
# 			frequencies[o_ner] += 1

# sorted_freqs = sorted(frequencies.items(), key=itemgetter(1), reverse=True)


# for i in range(len(sorted_freqs)):
# 	feature_dicc[sorted_freqs[i][0]] = i+1;


# # for i in range(len(feature_vector)):
# # 	feature_vector[i] = feature_vector[i][0]

# # print(copy)

# for s, p, o in sample:

# 	s = copy[fb.lookup_str(s)]
# 	p = fb.lookup_relstr(p)
# 	o = copy[fb.lookup_str(o)]

# 	print(s, p, o)

# 	if feature_dicc[s] and feature_dicc[o]:
# 		print(feature_dicc[s], feature_dicc[p], feature_dicc[o])

# print(feature_dicc)


# for entity in entities[:10]:
# 	print(entity);
	# rec = nlp.ner(entity[1:-1])
	
	# if not rec:
	# 	continue
	# else:
	# 	first = rec[0][1]

	# consist = True

	# for tup in rec:
	# 	if tup[1] != first or tup[1] == "O" or tup[1] == "MISC":
	# 		consist = False
	# 		print(rec)
	# 		break

	# consistent.append(consist)

# print(consistent.count(True)/(consistent.count(False)+consistent.count(True)))
# print(len(consistent))


nlp.close()

# >>> def fb_to_mkb(Id):
# ...     pseudo = fb.lookup_str(Id)
# ...     pseudo = "<http://rdf.freebase.com/ns/m." + pseudo[4:-1] + ">"
# ...     pseudo = mkb.lookup_id(pseudo)
# ...     return pseudo

# >>> def get_label(Id):
# ...     labels = mkb.o(Id, 6)
# ...     for label in labels:
# ...             label = mkb.lookup_str(label).split("@")
# ...             if label[1] == "en":
# ...                     return label[0][1:-1]
# ... 
# >>> get_label(fb_to_mkb(85))
# 'Rod Serling'
# >>> def get_description(Id):
# ...     descriptions = mkb.o(Id, 42)
# ...     for des in descriptions:
# ...             des = mkb.lookup_str(des).split("@")
# ...             if(des[1] == "en"):
# ...                     return des[0]
