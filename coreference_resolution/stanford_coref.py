import os
import json
from stanza.server import CoreNLPClient
os.environ["CORENLP_HOME"] = r'/home/michalis/stanford-corenlp-4.0.0'


def coref_resol(doc, client):
    # submit the request to the server
    ann = client.annotate(doc)

    # access the coref chain
    print('---')
    # print('coref chains for the given passage')
    chains = ann.corefChain

    # return chains
    for c in chains:
        # an array containing all the words that need to be resolved
        mentions = []
        # the best mention of the entity to be resolved (index)
        representative = c.representative
        # Mention details provided by StanfordNLP analysis
        r_mention = c.mention[representative]
        sentence = ann.sentence[r_mention.sentenceIndex]
        begin_index = r_mention.beginIndex
        end_index = r_mention.endIndex
        token = []
        for index in range(begin_index, end_index):
            token.append(sentence.token[index].word)
        mentions.append(' '.join(token))
        for m in c.mention:
            if m.mentionID is not r_mention.mentionID:
                sentence = ann.sentence[m.sentenceIndex]
                begin_index = m.beginIndex
                end_index = m.endIndex
                token = []
                for index in range(begin_index, end_index):
                    token.append(sentence.token[index].word)
                mentions.append(' '.join(token))
                sentence.token[begin_index].word = mentions[0]
                for index in range(begin_index + 1, end_index):
                    sentence.token[index].word = ''
        # print(mentions)
        # print('---')
    updated = []
    for sentence in ann.sentence:
        s = []
        for token in sentence.token:
            s.append(token.word)
        updated.append(' '.join(s))
    return updated


def entity_resol(doc):
    # set up the client
    print('---')
    print('starting up Java Stanford CoreNLP Server...')

    # set up the client
    with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'depparse', 'coref'], timeout=30000, memory='16G') as client:
        # submit the request to the server
        ann = client.annotate(doc)

        # print(ann.sentence[0].mentions[0])

        # get the named entity tag
        # print('---')
        # print('named entity tag of token')
        # for sentence in ann.sentence:
        #     for token in sentence.token:
        #         # print(token.pos)
        #         print(token.ner)

        # get an entity mention from the first sentence
        print('---')
        print('entities mentions in sentence')
        for sentence in ann.sentence:
            for mention in sentence.mentions:
                print(mention.ner + ' : ' + mention.entityMentionText)


def main():
    text = "Chris Manning is a nice person. Chris wrote a simple sentence. He also gives oranges to people."
    p1 = "What is a physician's assistant? "
    p2 = "What are the educational requirements required to become one? "
    p3 = "What does it cost? "
    p4 = "What school subjects are needed to become a registered nurse? "
    p5 = "What is the PA average salary vs an RN? "
    # p1 = "What are the main breeds of goat? "
    # p2 = "Tell me about boer goats. "
    # p3 = "What breed is good for their meat? "
    # p4 = "Are angora goats good for it? "
    # doc = p1 + p2 + p3 +p4

    # print(doc)
    # updated = coref_resol(doc)
    # print(updated)
    # entity_resol(doc)

    with open('/home/michalis/Desktop/CAsT/testing/train/queries/origin/train_topics_v1.0.json') as json_file:
        # set up the client
        print('---')
        print('starting up Java Stanford CoreNLP Server...')

        # set up the client
        with CoreNLPClient(annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse', 'depparse', 'coref'],
                           timeout=30000, memory='16G') as client:
            data = json.load(json_file)
            out = ""
            for p in data:
                # print(p)
                desc = p['description']
                current_conv = p['number']
                # print("Description: " + desc + '\n')
                # print(current_conv)
                # out = out + '\n' + 'Number: ' + str(current_conv) + '\n' + 'Description: ' + desc + '\n'
                u = [p['turn'][0]['raw_utterance']]
                out = out + str(current_conv) + '_1' + '\t' + u[0] + '\n'
                for t in p['turn'][1:]:
                    current_turn = t['number']
                    if len(u) < 2:
                        u.append(t['raw_utterance'])
                    else:
                        u.pop(0)
                        u.append(t['raw_utterance'])
                    # print(str(current_conv) + '_' + str(current_turn) + ": " + current_utterance)
                    doc = " ".join(u)
                    # print('---')
                    # print(doc)
                    u = coref_resol(doc, client)
                    print(u)
                    out = out + str(current_conv) + '_' + str(current_turn) + '\t' + u[-1] + '\n'
                    print('---')
                print(out)

    output = open("/home/michalis/Desktop/resolved.txt", "w")
    output.write(out)
    output.close()


if __name__ == "__main__":
    main()
