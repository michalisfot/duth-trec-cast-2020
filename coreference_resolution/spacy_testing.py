import spacy
import csv


def main():
    file_name = "test"
    file_path = "/home/michalis/Desktop/CAsT/testing/train/queries/" + file_name + ".txt"
    nlp = spacy.load("en_core_web_lg")
    with open(file_path) as tsv:
        for line in csv.reader(tsv, delimiter="\t"):
            text = line[1]
            doc = nlp(text)
            for chunk in doc.noun_chunks:
                # print(chunk.text, chunk.root.text, chunk.root.dep_,
                #       chunk.root.head.text)
                print(chunk.text)
            print('---')


if __name__ == "__main__":
    main()
