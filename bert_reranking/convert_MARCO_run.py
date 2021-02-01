import collections
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-q", "--queries", help="Queries to be processed")
parser.add_argument("-r", "--run", help="TREC run file path")
parser.add_argument("-o", "--output", help="Output file to save the processed queries")

args = parser.parse_args()


def load_queries(path):
    queries = collections.defaultdict()
    with open(path) as f:
        for line in f:
            query_id, query = line.rstrip().split('\t')
            if query_id not in queries:
                queries[query_id] = ''
            # query = re.sub('[^a-zA-Z0-9\n\.]', ' ', query)
            query = query.replace(' ', '%20')
            queries[query_id] = query
    return queries


def load_run(path):

    run = collections.OrderedDict()
    with open(path) as f:
        for i, line in enumerate(f):
            topic, _, doc_title, rank, score, _ = line.split(' ')
            collection = doc_title.rstrip().split('_')[0]
            if topic not in run:
                run[topic] = []
            if collection == 'MARCO':
                doc_id = doc_title.rstrip().split('_')[1]
                run[topic].append((doc_id, int(rank), float(score)))
    return run


def merge(queries, run, output_path):
    output = ''
    for query_id, data in run.items():
        print(query_id)
        # print(queries[query_id])
        for i, (doc_id, _, score) in enumerate(data):
            if i > 99:
                break
            output = output + queries[query_id] + ' Q0 ' + doc_id + ' ' + str(i+1) + ' ' + str(score) + ' indri' '\n'

    output_file = open(output_path, "w")
    output_file.write(output)
    output_file.close()


def main():
    if not(args.queries and args.run and args.output):
        raise RuntimeError('You have to specify all 3 required arguments')

    queries = load_queries(args.queries)
    run = load_run(args.run)
    merge(queries, run, args.output)
    print('Done!')


if __name__ == '__main__':
    main()
