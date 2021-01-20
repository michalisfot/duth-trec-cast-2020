from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers import spacy_tokenizer
import json
import warnings
warnings.filterwarnings("ignore")


def coref_resol(model, doc, indexes):
    tokenizer = spacy_tokenizer.SpacyTokenizer()
    results = model.predict(document=" ".join(doc))
    temp = results["document"]
    print("No. of clusters: " + str(results["clusters"]))
    # print(temp)

    for cluster in results["clusters"]:
        base_entity = " ".join(temp[slice(cluster[0][0], cluster[0][1] + 1)])
        base_entity = tokenizer.tokenize(text=base_entity)
        base_entity_len = len(base_entity)
        base_entity = " ".join(str(t) for t in base_entity)
        print("Base entity: " + base_entity)
        for entity in cluster:
            print("Entity: " + str(entity) + " : " + " ".join(temp[slice(entity[0], entity[1] + 1)]))
            if entity[0] == entity[1]:
                index = entity[0]
                temp[index] = base_entity
                for j in range(len(indexes)):
                    if index < indexes[j]:
                        indexes[j:] = [(i + base_entity_len - 1) for i in indexes[j:]]
                        break
    resolved = " ".join(temp)
    return resolved, indexes


def indexer(doc):
    tokenizer = spacy_tokenizer.SpacyTokenizer()
    index = []
    temp = 0
    for p in doc:
        # print(entity_resol(p))
        tokens = tokenizer.tokenize(text=p)
        # print(tokens)
        temp = len(tokens) + temp
        index.append(temp)
    index = [i - 1 for i in index]
    return index


def update_utterances(resolved, indexes):
    tokenizer = spacy_tokenizer.SpacyTokenizer()
    tokens = tokenizer.tokenize(text=resolved)
    start_index = 0
    doc = []
    for i in range(len(indexes)):
        doc.append(" ".join(str(t) for t in tokens[start_index:indexes[i]+1]))
        start_index = indexes[i]+1
    return doc


def entity_resol(doc):
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz")
    prediction = predictor.predict(sentence=doc)
    return prediction['tags']


def main():
    with open('/home/michalis/Desktop/CAsT/testing/eval/origin/evaluation_topics_v1.0.json') as json_file:
        pretrained = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")
        window = 4  # Coreference resolution on up to 5 previous queries
        data = json.load(json_file)
        out = ""
        # print(data)
        for p in data:
            # print(p)
            # desc = p['description']
            current_conv = p['number']
            # print("Description: " + desc + '\n')
            # print(current_conv)
            # out = out + '\n' + 'Number: ' + str(current_conv) + '\n' + 'Description: ' + desc + '\n'
            u = [p['turn'][0]['raw_utterance']]
            out = out + str(current_conv) + '_1' + '\t' + u[0] + '\n'
            for t in p['turn'][1:]:
                current_turn = t['number']
                if len(u) < window:
                    u.append(t['raw_utterance'])
                else:
                    u.pop(0)
                    u.append(t['raw_utterance'])
                # print(str(current_conv) + '_' + str(current_turn) + ": " + current_utterance)
                print('\n' + " ".join(u))
                indexes = indexer(u)
                resolved, indexes = coref_resol(pretrained, u, indexes)
                u = update_utterances(resolved, indexes)
                out = out + str(current_conv) + '_' + str(current_turn) + '\t' + u[-1] + '\n'
            print('--- --- ---')
        print(out)

    output = open("/home/michalis/Desktop/CAsT/testing/eval/automatic/allennlp_resolved.txt", "w")
    output.write(out)
    output.close()


if __name__ == "__main__":
    main()
