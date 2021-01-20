import csv, json


def main():
    file_name = "allennlp_resolved"
    # file_name = 'test'
    file_path = "/home/michalis/Desktop/CAsT/2020/" + file_name + ".txt"
    output_file_name = file_name + '_reformed'
    output_file_path = "/home/michalis/Desktop/CAsT/2020/" + output_file_name + ".txt"
    # reform_txt_file(file_path, output_file_path)

    file_name = "2020_automatic_evaluation_topics_v1.0.json"
    file_path = "/home/michalis/Desktop/CAsT/2020/" + file_name
    output_file_name = file_name + '_reformed'
    output_file_path = "/home/michalis/Desktop/CAsT/2020/" + output_file_name + ".txt"
    reform_json_file(file_path, output_file_path)


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
