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
            query = query.replace(' ', '%20')
            if query_id not in queries:
                queries[query] = ''
            # query = re.sub('[^a-zA-Z0-9\n\.]', ' ', query)
            queries[query] = query_id
    return queries


def load_run(path):

    run = collections.OrderedDict()
    with open(path) as f:
        for i, line in enumerate(f):
            topic, _, doc_id, rank, score, _ = line.split(' ')
            if topic not in run:
                run[topic] = []
            run[topic].append((doc_id, int(rank), float(score)))
    return run


def replace(queries, run):
    new_run = collections.defaultdict()
    for query_id, data in run.items():
        # print(query_id)
        # print(queries[query_id])
        for i, (doc_id, rank, score) in enumerate(data):
            if i > 99:
                break
            if queries[query_id] not in new_run:
                new_run[queries[query_id]] = []
            new_run[queries[query_id]].append((doc_id, rank, score))
    return new_run


def merge(original_run, msmarco_run, car_run, output_path):
    output = ''
    for query_id, data in original_run.items():
        for i, (doc_id, rank, score) in enumerate(data):
            if i > 99:
                break
            # query_id, _, doc_id, rank, score, _ = line.split(' ')
            if doc_id.split('_')[0] == 'MARCO':
                try:
                    doc_id = msmarco_run[query_id].pop(0)
                except:
                    pass
                output = output + query_id + ' Q0 MARCO_' + doc_id + ' ' + str(rank) + ' ' + str(score) + ' BERT' '\n'
            else:
                try:
                    doc_id = car_run[query_id].pop(0)
                except:
                    pass
                output = output + query_id + ' Q0 CAR_' + doc_id + ' ' + str(rank) + ' ' + str(score) + ' BERT' '\n'

    output_file = open(output_path, "w")
    output_file.write(output)
    output_file.close()


def convert_to_trec_run(car_msmarco_run, output_path):
    output = ''
    for query_id, data in car_msmarco_run.items():
        for doc_id, rank, score in data:
            output = output + query_id + ' Q0 ' + doc_id + ' ' + str(rank) + ' ' + str(score) + ' BERT' '\n'
    output_file = open(output_path, "w")
    output_file.write(output)
    output_file.close()


def main():
    if not(args.queries and args.run and args.output):
        raise RuntimeError('You have to specify all 3 required arguments')

    queries = load_queries(args.queries)
    car_msmarco_run = replace(queries, load_run(args.run))
    convert_to_trec_run(car_msmarco_run, args.output)
    print('Done!')


if __name__ == '__main__':
    main()
