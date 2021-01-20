import spacy
import neuralcoref


def coref_resol(p1, p2, p3):
    nlp = spacy.load('en')
    neuralcoref.add_to_pipe(nlp)
    doc = p1 + ' ' + p2 + ' ' + p3
    doc = nlp(doc)
    return doc._.coref_resolved


# p1 = "What is a physician's assistant? "
# p2 = "What are the educational requirements required to become one? "
# p3 = "What does it cost?"
# doc = p1 + p2 + p3
# print(doc)
# nlp = spacy.load('en')
# neuralcoref.add_to_pipe(nlp)
# doc = nlp("What breed of goat is good for meat? Are angora goats good for it?")
# print(doc._.coref_scores)
# print(doc._.coref_clusters)
# print(doc._.coref_resolved)
# print(doc)
# if doc._.has_coref:
#     for ent in doc.ents:
#         print(ent._.coref_cluster)
#     print(doc._.coref_resolved)
# doc1 = nlp('My sister has a dog. She loves him.')
# print(doc1._.coref_scores)
# print(doc1._.coref_clusters)
# print(doc1._.coref_resolved)
#
# doc2 = nlp('Angela lives in Boston. She is quite happy in that city.')
# for ent in doc2.ents:
#     print(ent._.coref_cluster)
# print(doc2._.coref_resolved)
