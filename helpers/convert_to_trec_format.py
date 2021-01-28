import csv
import re
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", help="Input file to be processed")
parser.add_argument("-o", "--output", help="Output file to save the processed queries")

args = parser.parse_args()


def main():
    if not(args.input and args.output):
        raise RuntimeError('You have to define both input and output files paths')

    queries = "<parameters>\n<index>/home/michalis/Desktop/CAsT/index/marco_index</index>\n" \
              "<index>/home/michalis/Desktop/CAsT/index/car_index</index>\n" \
         "<rule>method:dirichlet,mu:1000</rule>\n<count>1000</count>\n<trecFormat>true</trecFormat>"
    p1 = "<query> <type>indri</type> <number>"
    p2 = "</number> <text>"
    p3 = "</text> </query>"

    with open(args.input) as tsv:
        for line in csv.reader(tsv, delimiter="\t"):
            num = line[0]
            text = line[1]
            for i in text.split("\n"):
                text = re.sub(r"[^a-zA-Z0-9]+", ' ', i)
            query = p1 + num + p2 + text + p3
            queries = queries + "\n" + query

    queries = queries + "\n</parameters>"
    print(queries)
    output = open(args.output, "w")
    output.write(queries)
    output.close()


if __name__ == "__main__":
    main()
