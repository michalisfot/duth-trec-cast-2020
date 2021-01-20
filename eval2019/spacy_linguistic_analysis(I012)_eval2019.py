import spacy
import csv
import re


def main():
    file_name = "allennlp_resolved"
    # file_name = "allennlp_resolved_v4"
    # file_name = "allennlp_resolved_v5"
    file_path = "/home/michalis/Desktop/CAsT/testing/eval/automatic/" + file_name + ".txt"
    nlp = spacy.load("en_core_web_lg")
    with open(file_path) as tsv:
        all_nouns = []
        all_adjectives = []
        all_adverbs = []
        for line in csv.reader(tsv, delimiter="\t"):
            num = line[0]
            text = line[1]
            for i in text.split("\n"):
                text = re.sub(r"[^a-zA-Z0-9]+", ' ', i)
                # print(text)
                nouns, adjectives, prepositions, adverbs, verbs = pos_tagging(nlp, text)
                all_nouns.append(nouns)
                all_adjectives.append(adjectives)
                all_adverbs.append(adverbs)
                # all_prepositions.append(prepositions)
        # print(all_nouns)
        # print(all_adjectives)
        all_tokens = [i + j + k for i, j, k in zip(all_nouns, all_adjectives, all_adverbs)]
        print(all_tokens)

    convert_to_trec(file_name, all_tokens, 'I012b')


def pos_tagging(nlp, sentence):
    # nlp = spacy.load("en_core_web_lg")
    sentence = nlp(sentence)
    # print(sentence)
    nouns = []
    adjectives = []
    prepositions = []
    adverbs = []
    verbs = []
    for token in sentence:
        if token.pos_ is 'NOUN':
            nouns.append(token.text)
        elif token.pos_ is 'PROPN':
            nouns.append(token.text)
        elif token.pos_ is 'ADJ':
            adjectives.append(token.text)
        elif token.pos_ is 'ADP':
            prepositions.append(token.text)
        elif token.pos_ is 'ADV':
            adverbs.append(token.text)
        elif token.pos_ is 'VERB':
            verbs.append(token.text)
    return nouns, adjectives, prepositions, adverbs, verbs


def pos_tagging_original_query(nlp, sentence):
    # nlp = spacy.load("en_core_web_lg")
    sentence = nlp(sentence)
    # print(sentence)
    query = []
    for token in sentence:
        if token.pos_ is 'NOUN':
            query.append(token.text)
        elif token.pos_ is 'PROPN':
            query.append(token.text)
        elif token.pos_ is 'ADJ':
            query.append(token.text)
        elif token.pos_ is 'ADV':
            query.append(token.text)
        elif token.pos_ is 'VERB':
            query.append(token.text)
    return query


def convert_to_trec(file_name, context_tokens, export):
    nlp = spacy.load("en_core_web_lg")
    file_path = "/home/michalis/Desktop/CAsT/testing/eval/automatic/" + file_name + ".txt"
    queries = "<parameters>\n<index>/home/michalis/Desktop/CAsT/index/marco_index</index>\n" \
              "<index>/home/michalis/Desktop/CAsT/index/car_index</index>\n" \
              "<rule>method:dirichlet,mu:1000</rule>\n<count>1000</count>\n<trecFormat>true</trecFormat>"
    p1 = "<query> <type>indri</type> <number>"
    p2 = "</number> <text>"
    p3 = "</text> </query>"

    with open(file_path) as tsv:
        j = 0
        for line in csv.reader(tsv, delimiter="\t"):
            context = ''
            num = line[0]
            text = line[1]
            for i in text.split("\n"):
                text = re.sub(r"[^a-zA-Z0-9]+", ' ', i)
                text = pos_tagging_original_query(nlp, text)
                if int(num.split('_')[1]) > 2:
                    duplicates = set(text) & set(context_tokens[j-1])
                    for d in duplicates:
                        context_tokens[j-1].pop(context_tokens[j-1].index(d))
                    context = '0.5 #combine(' + ' '.join(context_tokens[j-1]) + ')'
                    duplicates = set(text) & set(context_tokens[j-2])
                    for d in duplicates:
                        context_tokens[j-2].pop(context_tokens[j-2].index(d))
                    context = context + ' 0.25 #combine(' + ' '.join(context_tokens[j-2]) + ')'
                elif int(num.split('_')[1]) > 1:
                    duplicates = set(text) & set(context_tokens[j-1])
                    for d in duplicates:
                        context_tokens[j-1].pop(context_tokens[j-1].index(d))
                    context = '0.5 #combine(' + ' '.join(context_tokens[j-1]) + ')'
            query = p1 + num + p2 + '#weight( 1.0 #combine(' + ' '.join(text) + ') ' + context + ')' + p3
            queries = queries + "\n" + query
            print(query)
            j = j + 1

    queries = queries + "\n</parameters>"
    output = open("/home/michalis/Desktop/CAsT/testing/eval/automatic/" + export, "w")
    output.write(queries)
    output.close()


if __name__ == "__main__":
    main()
