"""Converts TREC-CAR and MS MARCO queries and corpora into TFRecord that will be consumed by BERT.

The main necessary inputs are:
- Paragraph Corpus (CBOR file) 
- Pairs of Query-Relevant Paragraph (called qrels in TREC's nomenclature)
- Pairs of Query-Candidate Paragraph (called run in TREC's nomenclature)

The outputs are 3 TFRecord files, for training, dev and test.
"""
import collections
import os
import tensorflow as tf
import time
import csv
# local modules
import tokenization
import trec_car_classes

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "output_folder", './CAsT_tf_record/Y1',
    "Folder where the TFRecord files will be writen.")

flags.DEFINE_string(
    "vocab_file",
    "/home/michalis/Desktop/dl4marco-bert/data/uncased_L-24_H-1024_A-16/vocab.txt",
    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "car_corpus", "/home/michalis/Desktop/CAsT/Collection/TRECCAR/dedup.articles-paragraphs.cbor",
    "Path to the cbor file containing the Wikipedia paragraphs.")

flags.DEFINE_string(
    "msmarco_corpus", "/home/michalis/Desktop/CAsT/Collection/MSMARCO/MSMARCO_collection.tsv",
    "Path to the cbor file containing the MS MARCO paragraphs.")

flags.DEFINE_string(
    "topics", "/home/michalis/Desktop/CAsT/2019/eval/origin/queries/evaluation_topics_annotated_resolved_v1.0.tsv",
    "Path to the full queries.")

flags.DEFINE_string(
    "qrels_train", "./data/train.qrels",
    "Path to the topic / relevant doc ids pairs for training.")

flags.DEFINE_string(
    "qrels_dev", "./data/dev.qrels",
    "Path to the topic / relevant doc ids pairs for dev.")

flags.DEFINE_string(
    "qrels_test", "/home/michalis/Desktop/CAsT/2019/eval/2019qrels.txt",
    "Path to the topic / relevant doc ids pairs for test.")

flags.DEFINE_string(
    "run_train", "./data/train.run",
    "Path to the topic / candidate doc ids pairs for training.")

flags.DEFINE_string(
    "run_dev", "./data/dev.run",
    "Path to the topic / candidate doc ids pairs for dev.")

flags.DEFINE_string(
    "run_test", "/home/michalis/Desktop/CAsT/2019/eval/manual/I013.run",
    "Path to the topic / candidate doc ids pairs for test.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum query sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated.")

flags.DEFINE_integer(
    "num_train_docs", 10,
    "The number of docs per query for the training set.")

flags.DEFINE_integer(
    "num_dev_docs", 10,
    "The number of docs per query for the development set.")

flags.DEFINE_integer(
    "num_test_docs", 100,
    "The number of docs per query for the test set.")


def convert_dataset(data, corpus, set_name, tokenizer, output_name):
    output_path = FLAGS.output_folder + '/CAR_MSMARCO_dataset_' + output_name + set_name + '.tf'  # CHANGE
    print('Converting {} to tfrecord'.format(set_name))
    start_time = time.time()
    random_title = list(corpus.keys())[0]
    with tf.io.TFRecordWriter(output_path) as writer:
        for i, query in enumerate(data):
            qrels, doc_titles = data[query]
            query = tokenization.convert_to_unicode(query)
            print('query', query)
            query_ids = tokenization.convert_to_bert_input(
                text=query,
                max_seq_length=FLAGS.max_query_length,
                tokenizer=tokenizer,
                add_cls=True)
            query_ids_tf = tf.train.Feature(
                int64_list=tf.train.Int64List(value=query_ids))
            if set_name == 'train':
                max_docs = FLAGS.num_train_docs
            elif set_name == 'dev':
                max_docs = FLAGS.num_dev_docs
            elif set_name == 'test':
                max_docs = FLAGS.num_test_docs
            doc_titles = doc_titles[:max_docs]
            # Add fake docs so we always have max_docs per query.
            doc_titles += max(0, max_docs - len(doc_titles)) * [random_title]
            labels = [
                1 if doc_title in qrels else 0
                for doc_title in doc_titles
            ]
            doc_token_ids = [
                tokenization.convert_to_bert_input(
                    text=tokenization.convert_to_unicode(corpus[doc_title]),
                    max_seq_length=FLAGS.max_seq_length - len(query_ids),
                    tokenizer=tokenizer,
                    add_cls=False)
                for doc_title in doc_titles
            ]
            for rank, (doc_token_id, label) in enumerate(zip(doc_token_ids, labels)):
                doc_ids_tf = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=doc_token_id))
                labels_tf = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[label]))
                len_gt_titles_tf = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[len(qrels)]))
                features = tf.train.Features(feature={
                    'query_ids': query_ids_tf,
                    'doc_ids': doc_ids_tf,
                    'label': labels_tf,
                    'len_gt_titles': len_gt_titles_tf,
                })
                example = tf.train.Example(features=features)
                writer.write(example.SerializeToString())
            print('wrote {}, {} of {} queries'.format(set_name, i, len(data)))
            time_passed = time.time() - start_time
            est_hours = (len(data) - i) * time_passed / (max(1.0, i) * 3600)
            print('estimated total hours to save: {}'.format(est_hours))


def load_qrels(path):
    """Loads qrels into a dict of key: topic, value: list of relevant doc ids."""
    qrels = collections.defaultdict(set)
    with open(path) as f:
        for i, line in enumerate(f):
            topic, _, doc_id, relevance = line.rstrip().split(' ')
            if int(relevance) >= 1:
                qrels[topic].add(doc_id)
            if i % 1000000 == 0:
                print('Loading qrels {}'.format(i))
    return qrels


def load_run(path):
    """Loads run into a dict of key: topic, value: list of candidate doc ids."""
    # We want to preserve the order of runs so we can pair the run file with the
    # TFRecord file.

    run = collections.OrderedDict()
    with open(path) as f:
        for i, line in enumerate(f):
            topic, _, doc_id, rank, _, _ = line.split(' ')
            if topic not in run:
                run[topic] = []
            run[topic].append((doc_id, int(rank)))
            if i % 1000000 == 0:
                print('Loading run {}'.format(i))
    # Sort candidate docs by rank.
    sorted_run = collections.OrderedDict()
    for topic, doc_titles_ranks in run.items():
        sorted(doc_titles_ranks, key=lambda x: x[1])
        doc_titles = [doc_titles for doc_titles, _ in doc_titles_ranks]
        sorted_run[topic] = doc_titles
    return sorted_run


def load_queries(path):
    """Replace topic ids with full topic"""
    queries = collections.defaultdict()
    with open(path) as f:
        for line in f:
            query_id, query = line.rstrip().split('\t')
            if query_id not in queries:
                queries[query_id] = ''
            queries[query_id] = query
    return queries


def load_corpus(car_path, masmarco_path):
    """Loads TREC-CAR's and MS-MARCO's paraghaphs into a dict of key: title, value: paragraph."""
    corpus = {}
    start_time = time.time()
    approx_total_paragraphs = 30000000

    with open(car_path, 'rb') as f:
        for i, p in enumerate(trec_car_classes.iter_paragraphs(f)):
            para_txt = [elem.text if isinstance(elem, trec_car_classes.ParaText)
                        else elem.anchor_text
                        for elem in p.bodies]
            corpus['CAR_'+p.para_id] = ' '.join(para_txt)
            if i % 10000 == 0:
                print('Loading paragraph {} of {}'.format(i, approx_total_paragraphs))
                time_passed = time.time() - start_time
                hours_remaining = (
                                          approx_total_paragraphs - i) * time_passed / (max(1.0, i) * 3600)
                print('Estimated hours remaining to load corpus: {}'.format(
                    hours_remaining))

    approx_total_paragraphs = 9000000

    with open(masmarco_path) as tsv:
        for i, line in enumerate(csv.reader(tsv, delimiter="\t")):
            para_id = line[0]
            para_txt = line[1]
            corpus['MARCO_'+para_id] = para_txt
            if i % 10000 == 0:
                print('Loading paragraph {} of {}'.format(i, approx_total_paragraphs))
                time_passed = time.time() - start_time
                hours_remaining = (
                                          approx_total_paragraphs - i) * time_passed / (max(1.0, i) * 3600)
                print('Estimated hours remaining to load corpus: {}'.format(
                    hours_remaining))

    return corpus


def merge(topics, qrels, run):
    """Merge qrels and runs into a single dict of key: topic,
    value: tuple(relevant_doc_ids, candidate_doc_ids)"""
    data = collections.OrderedDict()
    new_data = collections.defaultdict()
    for topic, candidate_doc_ids in run.items():
        data[topic] = (qrels[topic], candidate_doc_ids)
        new_key = topics[topic]
        new_data[new_key] = data.pop(topic)

    return new_data


def main():
    print('Loading Tokenizer...')
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=True)

    if not os.path.exists(FLAGS.output_folder):
        os.mkdir(FLAGS.output_folder)

    print('Loading Corpora...')
    corpus = load_corpus(FLAGS.car_corpus, FLAGS.msmarco_corpus)

    for set_name, qrels_path, run_path in [('test', FLAGS.qrels_test, FLAGS.run_test)]:
        print('Converting {}'.format(set_name))
        qrels = load_qrels(path=qrels_path)
        run = load_run(path=run_path)
        queries = load_queries(path=FLAGS.topics)
        data = merge(topics=queries, qrels=qrels, run=run)
        convert_dataset(data=data, corpus=corpus, set_name=set_name, tokenizer=tokenizer, output_name='I013_manual_')

    # set_name = 'test'
    # qrels_path = "/home/michalis/Desktop/CAsT/2020/TREC 2020 results/qrels-reduced-cast-2020.txt"
    #
    # print('Converting {}'.format(set_name))
    # qrels = load_qrels(path=qrels_path)
    # run = load_run(path="/home/michalis/Desktop/CAsT/2020/post_competition/I013_automatic.run")
    # queries = load_queries(path='/home/michalis/Desktop/CAsT/2020/origin/2020_automatic_evaluation_topics_v1.0.json_reformed.txt')
    # data = merge(topics=queries, qrels=qrels, run=run)
    # convert_dataset(data=data, corpus=corpus, set_name=set_name, tokenizer=tokenizer, output_name='I013_automatic_')

    # print('Done!')

    # print('Converting {}'.format(set_name))
    # qrels = load_qrels(path=qrels_path)
    # run = load_run(path="/home/michalis/Desktop/CAsT/2020/post_competition/I013_manual.run")
    # queries = load_queries(path='/home/michalis/Desktop/CAsT/2020/origin/2020_manual_evaluation_topics_v1.0.json_reformed.txt')
    # data = merge(topics=queries, qrels=qrels, run=run)
    # convert_dataset(data=data, corpus=corpus, set_name=set_name, tokenizer=tokenizer, output_name='I013_manual_')

    print('Done!')


if __name__ == '__main__':
    main()
