# Standard imports
import argparse
import os

# Packages
import gensim

from scdb_issue_model.data.opinion import TarTokenExtractor

if __name__ == "__main__":
    # setup command line argument parser
    arg_parser = argparse.ArgumentParser(description="build a doc2vec model from a CourtListener archive")
    arg_parser.add_argument("--data_type", type=str, default="opinions", required=False,
                            help="type of data; opinions, audio, dockets")
    arg_parser.add_argument("--court", type=str, required=False, default="scotus",
                            help="court ID; see https://www.courtlistener.com/api/bulk-info/")
    arg_parser.add_argument("--vector_size", type=int, required=False, default=100,
                            help="doc2vec vector size")
    arg_parser.add_argument("--window_size", type=int, required=False, default=5,
                            help="doc2vec window size")
    arg_parser.add_argument("--dm", type=int, required=False, default=1,
                            help="doc2vec training type (1=DV, 0=DBOW)")
    arg_parser.add_argument("--alpha", type=float, required=False, default=0.01,
                            help="doc2vec learning rate")
    arg_parser.add_argument("--epochs", type=int, required=False, default=10,
                            help="doc2vec training epoch count")
    arg_parser.add_argument("--min_count", type=int, required=False, default=10,
                            help="doc2vec vocabulary min count")
    arg_parser.add_argument("--seed", type=int, required=False, default=None,
                            help="doc2vec training seed")
    arg_parser.add_argument("--sample_size", type=int, required=False, default=None,
                            help="number of opinions to use for model training")

    arg_parser.add_argument("--path", type=str, required=False, default="data", help="path to files")

    # get args and handle defaults
    args = arg_parser.parse_args()

    # get tar file name
    input_file_name = os.path.join(args.path, f"{args.court}_{args.data_type}.tar.gz")

    # track features, targets, and doc labels
    doc_label_list = []
    feature_vector_list = []
    target_list = []

    # materialize list given impact of tgz/json/tika for each pass
    document_list = [gensim.models.doc2vec.TaggedDocument(doc, []) for doc in
                     TarTokenExtractor(input_file_name, sample_size=args.sample_size)]

    # build model from parameters
    doc2vec_model = gensim.models.doc2vec.Doc2Vec(documents=document_list,
                                                  vector_size=args.vector_size,
                                                  window=args.window_size,
                                                  dm=args.dm,
                                                  alpha=args.alpha,
                                                  epochs=args.epochs,
                                                  min_count=args.min_count,
                                                  seed=args.seed)
    output_file_name = os.path.join(args.path,
                                    f"doc2vec-{args.vector_size}-{args.window_size}-{args.dm}-{args.min_count}")
    print(f"writing d2v model to {output_file_name}")
    doc2vec_model.save(output_file_name)
