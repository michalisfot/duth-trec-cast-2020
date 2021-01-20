import spacy
import csv
import re


def main():
    file_name = "stanfordnlp_resolved"
    # file_name = "test"
    file_path = "/home/michalis/Desktop/CAsT/testing/train/queries/" + file_name + ".txt"
    with open(file_path) as tsv:
        all_nouns = []
        all_adjectives = []
        all_prepositions = []
        for line in csv.reader(tsv, delimiter="\t"):
            num = line[0]
            text = line[1]
            for i in text.split("\n"):
                text = re.sub(r"[^a-zA-Z0-9]+", ' ', i)
                # print(text)
                nouns, adjectives, prepositions, verbs = pos_tagging(text)
                all_nouns.append(nouns)
                all_adjectives.append(adjectives)
                # all_prepositions.append(prepositions)
        # print(all_nouns)
        # print(all_adjectives)
        all_na = [i + j for i, j in zip(all_nouns, all_adjectives)]
        print(all_na)

    convert_to_trec(file_name, all_na, 'I010')


def pos_tagging(sentence):
    nlp = spacy.load("en_core_web_lg")
    sentence = nlp(sentence)
    # print(sentence)
    nouns = []
    adjectives = []
    prepositions = []
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
        elif token.pos_ is 'VERB':
            verbs.append(token.text)
    return nouns, adjectives, prepositions, verbs


def pos_tagging_original_query(sentence):
    nlp = spacy.load("en_core_web_lg")
    sentence = nlp(sentence)
    # print(sentence)
    query = ''
    for token in sentence:
        if token.pos_ is 'NOUN':
            query = query + ' ' + token.text
        elif token.pos_ is 'PROPN':
            query = query + ' ' + token.text
        elif token.pos_ is 'ADJ':
            query = query + ' ' + token.text
        elif token.pos_ is 'ADV':
            query = query + ' ' + token.text
        elif token.pos_ is 'VERB':
            query = query + ' ' + token.text
    return query


def convert_to_trec(file_name, all_nouns, export):
    file_path = "/home/michalis/Desktop/CAsT/testing/train/queries/" + file_name + ".txt"
    queries = "<parameters>\n<index>/home/michalis/Desktop/CAsT/index/marco_index</index>\n" \
              "<index>/home/michalis/Desktop/CAsT/index/car_index</index>\n" \
              "<rule>method:dirichlet,mu:1000</rule>\n<count>1000</count>\n<trecFormat>true</trecFormat>"
    p1 = "<query> <type>indri</type> <number>"
    p2 = "</number> <text>"
    p3 = "</text> </query>"

    with open(file_path) as tsv:
        j = 0
        for line in csv.reader(tsv, delimiter="\t"):
            noun_str = ''
            nouns = ''
            num = line[0]
            text = line[1]
            for i in text.split("\n"):
                text = re.sub(r"[^a-zA-Z0-9]+", ' ', i)
                text = pos_tagging_original_query(text)
                if int(num.split('_')[1]) > 2:
                    for n in all_nouns[j-1]:
                        nouns = nouns + ' ' + n
                    if all_nouns[j-1]:
                        noun_str = '0.5 #combine(' + nouns + ')'
                    nouns = ''
                    for n in all_nouns[j-2]:
                        nouns = nouns + ' ' + n
                    if all_nouns[j-2]:
                        noun_str = noun_str + ' 0.25 #combine(' + nouns + ')'
                elif int(num.split('_')[1]) > 1:
                    for n in all_nouns[j-1]:
                        nouns = nouns + ' ' + n
                    if all_nouns[j-1]:
                        noun_str = '0.5 #combine(' + nouns + ')'
            query = p1 + num + p2 + '#weight( 1.0 #combine(' + text + ') ' + noun_str + ')' + p3
            queries = queries + "\n" + query
            print(query)
            j = j + 1

    queries = queries + "\n</parameters>"
    output = open("/home/michalis/Desktop/CAsT/testing/train/queries/" + export, "w")
    output.write(queries)
    output.close()


if __name__ == "__main__":
    main()
