import spacy
import neuralcoref
import json
from neural_coref import coref_resol

with open('../testing/train/queries/origin/train_topics_v1.0.json') as json_file:
    data = json.load(json_file)[0]  # TODO remove [0]
    # print(data)
    # for p in data:
    p = data
    print(p)
    desc = p['description']
    current_conv = p['number']
    print("Description: " + desc + '\n')
    # print(current_conv)
    pre_previous_utterance = p['turn'][0]['raw_utterance']
    previous_utterance = p['turn'][1]['raw_utterance']
    for t in p['turn'][2:]:
        current_turn = t['number']
        current_utterance = t['raw_utterance']
        # current_utterance = re.sub(r"[^a-zA-Z0-9]+", ' ', current_utterance)
        # temp = str(current_conv) + '_' + str(current_turn) + ": " + current_utterance
        temp = pre_previous_utterance + ' ' + previous_utterance + ' ' + current_utterance
        print(coref_resol(pre_previous_utterance, previous_utterance, current_utterance))
        pre_previous_utterance = previous_utterance
        previous_utterance = current_utterance
