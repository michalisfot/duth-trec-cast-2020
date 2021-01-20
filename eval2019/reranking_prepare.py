import collections


def load_run(path):

    # We want to preserve the order of runs so we can pair the run file with the
    # TFRecord file.
    run = collections.OrderedDict()
    with open(path) as f:
        curr_doc = None
        curr_q = '31_1'
        i = 0
        for line in f:
            query_id, _, doc_id, rank, _, _ = line.split(' ')
            if curr_q != query_id:
                i = 0
                curr_doc = None
            if i < 10:
                if curr_doc != doc_id.split('_')[0] and curr_doc is not None:
                    print(query_id)
                    print("ALERT!")
                curr_doc = doc_id.split('_')[0]
                i += 1
            curr_q = query_id


def main():
    path = "/home/michalis/Desktop/CAsT/testing/eval/automatic/I013b.run"
    load_run(path)


if __name__ == "__main__":
    main()
