import re
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", help="Input file to be processed")
parser.add_argument("-o", "--output", help="Output file to save the processed queries")

args = parser.parse_args()


def main():
    if not(args.input and args.output):
        raise RuntimeError('You have to define both input and output files paths')

    queries = "<parameters>\n<index>/home/michalis/Desktop/CAsT/indices</index>\n" \
         "<rule>method:dirichlet,mu:1000</rule>\n<count>1000</count>\n<trecFormat>true</trecFormat>"
    p1 = "<query> <type>indri</type> <number>"
    p2 = "</number> <text>"
    p3 = "</text> </query>"

    with open(args.input) as json_file:
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
    output = open(args.output, "w")
    output.write(queries)
    output.close()


if __name__ == "__main__":
    main()
