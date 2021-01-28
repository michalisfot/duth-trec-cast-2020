import csv
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", help="Input file to be processed")
parser.add_argument("-o", "--output", help="Output file to save the processed queries")
parser.add_argument("-t", "--type", help="Specifies the type of reformation (txt or json)")

args = parser.parse_args()


def main():
    if not(args.input and args.output and args.type):
        raise RuntimeError('You have to define both input and output files paths')
    if args.type == 'txt':
        reform_txt_file(args.input, args.output)
    elif args.type == 'json':
        reform_json_file(args.input, args.output)
    else:
        raise RuntimeError('You have to specify either "txt" or "json" types of reformation')


def reform_txt_file(file_path, output_file_path):
    with open(file_path) as tsv:
        current_conv = '1'
        output = ''
        for line in csv.reader(tsv, delimiter="\t"):
            # print(line[0].split("_")[0])
            if current_conv != line[0].split("_")[0]:
                output = output + '\n'
            current_conv = line[0].split("_")[0]
            raw_utterance = line[1]
            output = output + str(line[0]) + '\t' + raw_utterance + '\n'
        print(output)
        output_file = open(output_file_path, "w")
        output_file.write(output)
        output_file.close()


def reform_json_file(file_path, output_file_path):
    with open(file_path) as json_file:
        data = json.load(json_file)
        current_conv = '1'
        output = ''
        for p in data:
            if current_conv != p['number']:
                output = output
            current_conv = p['number']
            for t in p['turn']:
                current_turn = t['number']
                raw_utterance = t['automatic_rewritten_utterance']
                output = output + str(current_conv) + '_' + str(current_turn) + '\t' + raw_utterance + '\n'
        print(output)
        output_file = open(output_file_path, "w")
        output_file.write(output)
        output_file.close()


if __name__ == "__main__":
    main()
