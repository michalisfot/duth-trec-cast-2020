from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers import spacy_tokenizer
import json
import warnings
warnings.filterwarnings("ignore")


def coref_resol(model, pos_tagging_model, doc):
    tokenizer = spacy_tokenizer.SpacyTokenizer()
    # print(doc)
    if len(doc) > 3:
        # print(" ".join(doc[2:]))
        results = model.predict(document=" ".join(doc[2:]))
        indexes = indexer(doc[2:])
        if len(results["clusters"]) == 0:
            # print(" ".join(doc[1:]))
            results = model.predict(document=" ".join(doc[1:]))
            indexes = indexer(doc[1:])
            if len(results["clusters"]) == 0:
                # print(" ".join(doc))
                results = model.predict(document=" ".join(doc))
                indexes = indexer(doc)
    elif len(doc) > 2:
        # print(" ".join(doc[1:]))
        results = model.predict(document=" ".join(doc[1:]))
        indexes = indexer(doc[1:])
        if len(results["clusters"]) == 0:
            # print(" ".join(doc))
            results = model.predict(document=" ".join(doc))
            indexes = indexer(doc)
    else:
        results = model.predict(document=" ".join(doc))
        indexes = indexer(doc)
    temp = results["document"]
    print("No. of clusters: " + str(results["clusters"]))
    # print(results)

    offset = 0
    for cluster in results["clusters"]:
        base_entity = " ".join(temp[slice(cluster[0][0], cluster[0][1] + 1)])
        base_entity = tokenizer.tokenize(text=base_entity)
        base_entity_len = len(base_entity)
        base_entity = " ".join(str(t) for t in base_entity)
        print("Base entity: " + base_entity)
        for entity in cluster[1:]:
            print("Entity: " + str(entity) + " : " + " ".join(temp[slice(entity[0] + offset, entity[1] + offset + 1)]))
            if len(" ".join(temp[slice(entity[0] + offset, entity[1] + offset + 1)])) > 2 and \
                    contains_entity(pos_tagging_model, " ".join(temp[slice(entity[0] + offset, entity[1] + offset + 1)])):
                continue

            if entity[0] == entity[1]:
                temp_entity = " ".join(temp[slice(entity[0] + offset, entity[1] + offset + 1)])
                temp_entity = tokenizer.tokenize(text=temp_entity)
                temp_entity_len = len(temp_entity)

                index = entity[0] + offset
                temp[index] = base_entity
                temp = tokenize(" ".join(temp))
                offset = offset + base_entity_len - temp_entity_len
                # print(temp)
                for j in range(len(indexes)):
                    if index < indexes[j]:
                        indexes[j:] = [(i + base_entity_len - 1) for i in indexes[j:]]
                        break
            else:
                temp_entity = " ".join(temp[slice(entity[0] + offset, entity[1] + offset + 1)])
                temp_entity = tokenizer.tokenize(text=temp_entity)
                temp_entity_len = len(temp_entity)

                for i in range(entity[1]-entity[0]):
                    temp.pop(entity[0] + offset)
                    # print(temp.pop(entity[0]))
                index = entity[0] + offset
                temp[index] = base_entity
                temp = tokenize(" ".join(temp))
                offset = offset + base_entity_len - temp_entity_len
                # print(temp)
                print("Initial indexes: "+str(indexes))
                for j in range(len(indexes)):
                    if index < indexes[j]:
                        indexes[j:] = [(i + base_entity_len - temp_entity_len) for i in indexes[j:]]
                        print("Updated indexes: "+str(indexes))
                        break
    resolved = " ".join(temp)
    return resolved, indexes


def indexer(doc):  # A function that indexes the end of each sentence
    tokenizer = spacy_tokenizer.SpacyTokenizer()
    index = []
    temp = 0
    for p in doc:
        tokens = tokenizer.tokenize(text=p)
        temp = len(tokens) + temp
        index.append(temp)
    index = [i - 1 for i in index]
    return index


def tokenize(doc):
    tokenizer = spacy_tokenizer.SpacyTokenizer()
    tokens = tokenizer.tokenize(text=doc)
    for i in range(len(tokens)):
        tokens[i] = str(tokens[i])
    return tokens


def update_utterances(resolved, indexes):
    tokenizer = spacy_tokenizer.SpacyTokenizer()
    tokens = tokenizer.tokenize(text=resolved)
    start_index = 0
    doc = []
    for i in range(len(indexes)):
        doc.append(" ".join(str(t) for t in tokens[start_index:indexes[i]+1]))
        start_index = indexes[i]+1
    return doc


def contains_entity(predictor, doc):
    prediction = predictor.predict(sentence=doc)
    pos = prediction['tags']
    for t in pos:
        if len(t) > 1:  # If len of POS tag is greater than 1 then its a named entity
            return True
    return False


def main():
    with open('/home/michalis/Desktop/CAsT/testing/train/queries/origin/train_topics_v1.0.json') as json_file:
        coref_model = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz")
        pos_tagging_model = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz")
        window = 4  # Coreference resolution on up to 5 previous queries
        data = json.load(json_file)
        out = ""
        # print(data)
        for p in data[23:24]:
            # print(p)
            desc = p['description']
            current_conv = p['number']
            # print("Description: " + desc + '\n')
            print("Current conv: " + str(current_conv) +'\n')
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
                print("Input: " + " ".join(u))
                # indexes = indexer(u)
                resolved, indexes = coref_resol(coref_model, pos_tagging_model, u)
                u = update_utterances(resolved, indexes)
                print("Output: " + str(u))
                print("---")
                out = out + str(current_conv) + '_' + str(current_turn) + '\t' + u[-1] + '\n'
            print('\n --- --- --- \n')
        print(out)

    # output = open("/home/michalis/Desktop/CAsT/testing/train/queries/allennlp_resolved_v3.txt", "w")
    # output.write(out)
    # output.close()


if __name__ == "__main__":
    main()
