# Standard imports
import argparse
import datetime
import os

# Packages
import gensim.models.doc2vec
import pandas
from gensim.test.test_parsing import doc2

from scdb_issue_model.data.opinion import TarTokenExtractor
from scdb_issue_model.data.scdb import load_scdb_data

if __name__ == "__main__":
    # setup command line argument parser
    arg_parser = argparse.ArgumentParser(
        description="compute doc2vec vectors from a given model and CourtListener archive")
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
    arg_parser.add_argument("--sample_size", type=int, required=False, default=None,
                            help="number of opinions to use for model training")
    arg_parser.add_argument("--year", type=int, required=False, help="SCDB release year; defaults to prior year")
    arg_parser.add_argument("--release", type=int, required=False, default=1,
                            help="SCDB release version; defaults to 1")
    arg_parser.add_argument("--record", type=str, required=False, default="case",
                            help="SCDB record type; case or justice")
    arg_parser.add_argument("--grouping", type=str, required=False, default="Citation",
                            help="SCDB grouping type (CASE SENSITIVE); Citation, Docket, LegalProvision, or Vote")
    arg_parser.add_argument("--start_year", type=int, required=False, default=None,
                            help="first year to use for training")
    arg_parser.add_argument("--end_year", type=int, required=False, default=None,
                            help="first year to use for training")
    arg_parser.add_argument("--path", type=str, required=False, default="data", help="path to files")

    # get args and handle defaults
    args = arg_parser.parse_args()

    # if release year is not provided, default to *last* year because coding is not completed until the term is over and
    # the coding release process is completed.
    if args.year is None:
        args.year = datetime.date.today().year - 1

    # load SCDB data
    scdb_df = load_scdb_data(args.path, args.year, args.release)

    # load map file
    cl_scdb_df = pandas.read_csv(
        os.path.join(args.path, f"map-{args.year}_{args.release}-{args.court}_{args.data_type}.csv"),
        encoding="utf-8", index_col=0)

    # get reverse mapping for lookup by year
    good_cl_scdb_df = cl_scdb_df[-pandas.isnull(cl_scdb_df['scdb_case_id'])]
    scdb_cl_dict = dict(list(zip(good_cl_scdb_df.loc[:, 'scdb_case_id'], good_cl_scdb_df.index)))

    # get start and end years
    if args.start_year is None:
        start_year = scdb_df.loc[:, "yearDecided"].min()
    else:
        start_year = args.start_year

    if args.end_year is None:
        end_year = scdb_df.loc[:, "yearDecided"].max()
    else:
        end_year = args.end_year

    # get tar file name
    input_file_name = os.path.join(args.path, f"{args.court}_{args.data_type}.tar.gz")

    # read all documents from tarball
    document_token_list = []
    document_label_list = []
    for doc_tokens, doc_name in TarTokenExtractor(input_file_name, return_member=True, sample_size=args.sample_size):
        member_id = int(doc_name.replace('.json', ''))
        if not pandas.isnull(cl_scdb_df.loc[member_id]).any():
            document_token_list.append(doc_tokens)
            document_label_list.append(member_id)

    # iterate over years, load each model, and store results
    doc_vector_list = []
    doc_name_list = []
    walkforward_path = os.path.join(args.path,
                                    f"doc2vec-{args.vector_size}-{args.window_size}-{args.dm}-{args.min_count}-walkforward")

    for year in range(start_year, end_year + 1):
        # get current year cases
        year_case_index = scdb_df.index[scdb_df.loc[:, "yearDecided"] == year]
        year_document_index = [i for i in range(len(document_label_list))
                               if scdb_df.loc[
                                   cl_scdb_df.loc[document_label_list[i], 'scdb_case_id'], 'yearDecided'] == year]

        print(year, len(year_document_index))
        if len(year_document_index) == 0:
            continue

        # load model
        year_doc2vec_path = os.path.join(walkforward_path, f"{year}")
        if not os.path.exists(year_doc2vec_path):
            raise RuntimeError(f"doc2vec model {year_doc2vec_path} does not exist")
        year_doc2vec_model = gensim.models.doc2vec.Doc2Vec.load(year_doc2vec_path)

        for i in year_document_index:
            doc_vector_list.append(year_doc2vec_model.infer_vector(document_token_list[i]))
            doc_name_list.append(cl_scdb_df.loc[document_label_list[i], 'scdb_case_id'])

    # output feature data
    doc_vector_df = pandas.DataFrame(doc_vector_list, index=doc_name_list)
    doc_vector_df.to_csv(os.path.join(args.path,
                                      f"features-doc2vec-walkforward-{args.vector_size}-{args.window_size}-{args.dm}-{args.min_count}-{args.court}-{args.data_type}-{args.year}-{args.release}.csv.gz"))
