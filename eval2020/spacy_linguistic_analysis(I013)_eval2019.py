import spacy
import csv
import re


def main():
    # file_name = "allennlp_resolved_v5.txt"
    file_name = "2020_manual_evaluation_topics_v1.0.json_reformed.txt"
    # file_name = "2020_automatic_evaluation_topics_v1.0.json_reformed.txt"
    export = 'I013_manual'
    file_path = "/home/michalis/Desktop/CAsT/2020/origin/" + file_name
    nlp_model = spacy.load("en_core_web_lg")
    queries = "<parameters>\n<index>/home/michalis/Desktop/CAsT/index/marco_index</index>\n" \
              "<index>/home/michalis/Desktop/CAsT/index/car_index_v2</index>\n<runID>duth_manual</runID>\n" \
              "<rule>method:dirichlet,mu:1000</rule>\n<count>1000</count>\n<trecFormat>true</trecFormat>\n"
    with open(file_path) as tsv:
        j = 0
        context = []
        curr_query = 81
        for line in csv.reader(tsv, delimiter="\t"):
            num = line[0]
            # if num.split('_')[0] != curr_query:
            #     context = []
            original_query = line[1]
            # print(original_query)
            for i in original_query.split("\n"):
                original_query = re.sub(r"[^a-zA-Z0-9]+", ' ', i)
                nouns, adjectives, prepositions, adverbs, verbs = pos_tagging(nlp_model, original_query)
                original_query = nouns + adjectives + adverbs + verbs
                # print(original_query)
                if int(num.split('_')[1]) > 2:
                    duplicates = set(original_query) & set(context[j - 1])
                    for d in duplicates:
                        context[j - 1].pop(context[j - 1].index(d))
                    duplicates = set(original_query) & set(context[j - 2])
                    for d in duplicates:
                        context[j - 2].pop(context[j - 2].index(d))
                    query = convert_to_trec(original_query, num, context[j - 2:])
                    queries = queries + query
                elif int(num.split('_')[1]) > 1:
                    duplicates = set(original_query) & set(context[j - 1])
                    for d in duplicates:
                        context[j - 1].pop(context[j - 1].index(d))
                    query = convert_to_trec(original_query, num, context[j - 1:])
                    queries = queries + query
                else:
                    query = convert_to_trec(original_query, num, context[j:])
                    queries = queries + query
                context.append(nouns + adjectives + adverbs)
                # print(context)
            # curr_query = num.split('_')[0]
            j = j + 1
            # print(j)

    queries = queries + "\n</parameters>"
    print(queries)
    output = open("/home/michalis/Desktop/CAsT/2020/post_competition/" + export, "w")
    output.write(queries)
    output.close()


def pos_tagging(nlp_model, sentence):
    sentence = nlp_model(sentence)
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


def convert_to_trec(original_query, num, context_tokens):
    p1 = "<query> <type>indri</type> <number>"
    p2 = "</number> <text>"
    p3 = "</text> </query>"
    context = ''
    # print(context_tokens)
    if len(context_tokens) > 1:
        if len(context_tokens[1]) > 0:
            context = '0.5 #combine(' + ' '.join(context_tokens[1]) + ')'
        if len(context_tokens[0]) > 0:
            context = context + ' 0.25 #combine(' + ' '.join(context_tokens[0]) + ')'
    elif len(context_tokens) > 0:
        if len(context_tokens[0]) > 0:
            context = '0.5 #combine(' + ' '.join(context_tokens[0]) + ')'
    query = p1 + num + p2 + '#weight( 1.0 #combine(' + ' '.join(original_query) + ') ' + context + ')' + p3 + '\n'
    # print(query)
    return query


if __name__ == "__main__":
    main()
