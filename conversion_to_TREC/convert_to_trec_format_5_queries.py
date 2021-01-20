import csv
import re

window = 4  # 5 history queries will be concatenated
file_name = "resolved.txt"
file_path = "../testing/train/queries/" + file_name
queries = "<parameters>\n<index>/home/michalis/Desktop/CAsT/index/marco_index</index>\n" \
          "<index>/home/michalis/Desktop/CAsT/index/car_index</index>\n" \
          "<index>/home/michalis/Desktop/CAsT/index/wapo_index</index>\n" \
     "<rule>method:dirichlet,mu:1000</rule>\n<count>1000</count>\n<trecFormat>true</trecFormat>"
p1 = "<query> <type>indri</type> <number>"
p2 = "</number> <text>"
p3 = "</text> </query>"

with open(file_path) as tsv:
    history = []
    for line in csv.reader(tsv, delimiter="\t"):
        num = line[0]
        text = line[1]
        conv_turn = num.split('_')[1]
        if int(conv_turn) is 1:
            history.clear()
        for i in text.split("\n"):
            text = re.sub(r"[^a-zA-Z0-9]+", ' ', i)
            if len(history) > window:
                history.pop(0)
                history.append(text)
            else:
                history.append(text)
        query = p1 + num + p2 + " ".join(history) + p3
        queries = queries + "\n" + query

queries = queries + "\n</parameters>"
print(queries)
output = open("../testing/train/queries/" + file_name + ".trec", "w")
output.write(queries)
output.close()
