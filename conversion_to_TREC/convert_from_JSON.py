import re
import json

queries = "<parameters>\n<index>/home/michalis/Desktop/CAsT/indices</index>\n" \
     "<rule>method:dirichlet,mu:1000</rule>\n<count>1000</count>\n<trecFormat>true</trecFormat>"
p1 = "<query> <type>indri</type> <number>"
p2 = "</number> <text>"
p3 = "</text> </query>"

with open('../testing/train/queries/train_topics_v1.0.json') as json_file:
    data = json.load(json_file)
    for p in data:
        desc = p['description']
        current_conv = p['number']
        for t in p['turn']:
            current_turn = t['number']
            current_utterance = t['raw_utterance']
            current_utterance = re.sub(r"[^a-zA-Z0-9]+", ' ', current_utterance)
            num = str(current_conv) + '_' + str(current_turn)
            query = p1 + num + p2 + current_utterance + p3
            queries = queries + "\n" + query

queries = queries + "\n</parameters>"
print(queries)
output = open("../testing/train/queries/train_topics_v1.0.json_trec_format", "w")
output.write(queries)
output.close()
