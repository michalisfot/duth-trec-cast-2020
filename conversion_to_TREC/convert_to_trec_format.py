import csv
import re

file_name = "stanfordnlp_resolved.txt"
file_path = "/home/michalis/Desktop/CAsT/testing/train/queries/" + file_name
queries = "<parameters>\n<index>/home/michalis/Desktop/CAsT/index/marco_index</index>\n" \
          "<index>/home/michalis/Desktop/CAsT/index/car_index</index>\n" \
     "<rule>method:dirichlet,mu:1000</rule>\n<count>1000</count>\n<trecFormat>true</trecFormat>"
p1 = "<query> <type>indri</type> <number>"
p2 = "</number> <text>"
p3 = "</text> </query>"

with open(file_path) as tsv:
    for line in csv.reader(tsv, delimiter="\t"):
        num = line[0]
        text = line[1]
        for i in text.split("\n"):
            text = re.sub(r"[^a-zA-Z0-9]+", ' ', i)
        query = p1 + num + p2 + text + p3
        queries = queries + "\n" + query

queries = queries + "\n</parameters>"
print(queries)
output = open("/home/michalis/Desktop/CAsT/testing/train/queries/" + file_name + ".trec", "w")
output.write(queries)
output.close()
