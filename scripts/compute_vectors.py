# imports
import argparse
import datetime
import json
import tarfile
import os

# packages
import pandas
import spacy

import tika
import tika.parser


def load_scdb_data(path: str, release_year: int, release_version: int, record: str = "case",
                   grouping: str = "citation"):
    scdb_file_name = f"{release_year}_{release_version:02}_{record.lower()}_{grouping.lower()}.csv"
    return pandas.read_csv(os.path.join(path, scdb_file_name), low_memory="False", encoding="latin1",
                           index_col='caseId')


if __name__ == "__main__":
    # setup command line argument parser
    arg_parser = argparse.ArgumentParser(description="create a pre-computed mapping between Court Listener and SCDB")
    arg_parser.add_argument("--vector_type", type=str, default="spacy-en-large", required=False,
                            help="vector type to calculate; see README")
    arg_parser.add_argument("--target_name", type=str, default="issueArea", required=False,
                            help="SCDB target column name")
    arg_parser.add_argument("--data_type", type=str, default="opinions", required=False,
                            help="type of data; opinions, audio, dockets")
    arg_parser.add_argument("--court", type=str, required=False, default="scotus",
                            help="court ID; see https://www.courtlistener.com/api/bulk-info/")
    arg_parser.add_argument("--year", type=int, required=False, help="SCDB release year; defaults to prior year")
    arg_parser.add_argument("--release", type=int, required=False, default=1,
                            help="SCDB release version; defaults to 1")
    arg_parser.add_argument("--record", type=str, required=False, default="case",
                            help="SCDB record type; case or justice")
    arg_parser.add_argument("--grouping", type=str, required=False, default="Citation",
                            help="SCDB grouping type (CASE SENSITIVE); Citation, Docket, LegalProvision, or Vote")
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

    if args.vector_type == "spacy-en-large":
        nlp = spacy.load("en_core_web_lg")
    elif args.vector_type == "spacy-en-trf":
        nlp = spacy.load("en_core_web_trf")
    else:
        raise NotImplementedError(f"vector_type={args.vector_type} is not available; see README")

    # get file
    tar_file_name = os.path.join(args.path, f"{args.court}_{args.data_type}.tar.gz")

    # track features, targets, and doc labels
    doc_label_list = []
    feature_vector_list = []
    target_list = []

    with tarfile.open(tar_file_name, 'r:gz') as scotus_tar:
        for member in scotus_tar.getmembers():
            # set member ID
            member_id = int(member.name.replace(".json", ""))

            # skip opinions without an issue area
            if member_id not in cl_scdb_df.index or pandas.isnull(cl_scdb_df.loc[member_id, 'scdb_case_id']):
                continue

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
            file_doc = nlp(file_content)

            try:
                doc_label = cl_scdb_df.loc[member_id].values[0]
                target_value = int(scdb_df.loc[cl_scdb_df.loc[member_id], args.target_name].values[0])
            except Exception as e:
                print(f"invalid label or target data; skipping {member_id}")
                continue

            # track label and target
            doc_label_list.append(doc_label)
            target_list.append(target_value)

            # handle feature
            if args.vector_type == "spacy-en-large":
                feature_vector_list.append(file_doc.vector)
            elif args.vector_type == "spacy-en-trf":
                feature_vector_list.append(file_doc._.trf_data.tensors[1].mean(axis=0))

    # save outputs
    feature_df = pandas.DataFrame(feature_vector_list, index=doc_label_list)
    target_series = pandas.Series(target_list, index=doc_label_list, name=args.target_name)
    feature_df.to_csv(os.path.join(args.path,
                                   f"features-{args.vector_type}-{args.court}-{args.data_type}-{args.year}-{args.release}.csv"))
    target_series.to_csv(
        os.path.join(args.path, f"targets-{args.court}-{args.data_type}-{args.year}-{args.release}.csv"))
