# Standard imports
import argparse
import json
import os
import tarfile

# Packages
from typing import Iterable

import gensim
import spacy
import tika
import tika.parser


class TarTokenExtractor(Iterable):
    """
    Iterable tar file token extraction for gensim input
    """

    def __init__(self, tar_file_name: str, spacy_model: str = "en_core_web_sm",
                 sample_size: int = 1000, progress_step: int = 100):
        """
        Constructor with tar file name
        :param tar_file_name:
        """
        self.tar_file_name = tar_file_name
        self.nlp = nlp = spacy.load(spacy_model)
        self.sample_size = sample_size
        self.progress_step = progress_step

    def __iter__(self):
        n = 0
        with tarfile.open(self.tar_file_name, 'r:gz') as scotus_tar:
            if self.sample_size is None:
                sample = scotus_tar.getmembers()
            else:
                sample = scotus_tar.getmembers()[0:self.sample_size]

            for member in sample:
                if n % self.progress_step == 0:
                    print(f"completed: {n}/{self.sample_size}")
                n += 1

                # set member ID
                member_id = int(member.name.replace(".json", ""))

                # read the file and parse to JSON
                try:
                    # get the contents and convert HTML to plain text
                    file_data = json.loads(scotus_tar.extractfile(member).read())
                except Exception as e:
                    print(f"unable to parse {member.name} as JSON", e)

                try:
                    tika_response = tika.parser.from_buffer(f"<html>{file_data['html']}</html>")
                    file_content = tika_response['content']
                except Exception as e:
                    print(f"unable to extract {member.name} text with tika", e)

                # process text
                try:
                    file_doc = self.nlp(file_content)
                    file_tokens = [token.lemma_.lower() for token in file_doc if
                                   not token.is_stop and not token.is_space and not token.is_punct]
                except Exception as e:
                    # print(f"unable to parse file {member.name}")
                    continue

                yield gensim.models.doc2vec.TaggedDocument(file_tokens, [])


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
    arg_parser.add_argument("--sample_size", type=int, required=False, default=10000,
                            help="number of opinions to use for model training")

    arg_parser.add_argument("--path", type=str, required=False, default="data", help="path to files")

    # get args and handle defaults
    args = arg_parser.parse_args()

    # get file
    input_file_name = os.path.join(args.path, f"{args.court}_{args.data_type}.tar.gz")

    # track features, targets, and doc labels
    doc_label_list = []
    feature_vector_list = []
    target_list = []

    # materialize list given impact of tgz/json/tika for each pass
    document_list = [doc for doc in TarTokenExtractor(input_file_name, sample_size=args.sample_size)]

    # build model from parameters
    doc2vec_model = gensim.models.doc2vec.Doc2Vec(documents=document_list,
                                                  vector_size=args.vector_size,
                                                  window=args.window_size,
                                                  dm=args.dm,
                                                  alpha=args.alpha,
                                                  epochs=args.epochs,
                                                  min_count=args.min_count)
    output_file_name = os.path.join(args.path,
                                    f"doc2vec-{args.vector_size}-{args.window_size}-{args.dm}-{args.min_count}")
    print(f"writing d2v model to {output_file_name}")
    doc2vec_model.save(output_file_name)
