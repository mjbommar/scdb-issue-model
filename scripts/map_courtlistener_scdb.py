# standard imports
import argparse
import datetime
import json
import os
import tarfile

# packages
import lxml.html
import pandas

from scdb_issue_model.data.scdb import load_scdb_data

if __name__ == "__main__":
    # setup command line argument parser
    arg_parser = argparse.ArgumentParser(description="create a pre-computed mapping between Court Listener and SCDB")
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

    # get file
    tar_file_name = os.path.join(args.path, f"{args.court}_{args.data_type}.tar.gz")

    # create the mapping
    cl_scdb_mapping = {}

    with tarfile.open(tar_file_name, 'r:gz') as scotus_tar:
        for member in scotus_tar.getmembers():
            # set member ID
            member_id = member.name.replace(".json", "")

            # read the file and parse to JSON
            try:
                file_data = json.loads(scotus_tar.extractfile(member).read())
            except Exception as e:
                print(f"unable to parse {member.name} as JSON", e)
                cl_scdb_mapping[member_id] = None

            # get HTML doc
            try:
                if file_data['html'] is not None and len(file_data['html']) > 0:
                    file_html = lxml.html.fromstring(file_data['html'])
                elif file_data['html_with_citations'] is not None and len(file_data['html_with_citations']) > 0:
                    file_html = lxml.html.fromstring(file_data['html_with_citations'])
                elif file_data['html_lawbox'] is not None and len(file_data['html_lawbox']) > 0:
                    file_html = lxml.html.fromstring(file_data['html_lawbox'])
                elif file_data['html_columbia'] is not None and len(file_data['html_columbia']) > 0:
                    file_html = lxml.html.fromstring(file_data['html_columbia'])
                else:
                    cl_scdb_mapping[member_id] = None
                    continue

                scdb_id_match = None

                for case_cite in file_html.xpath("//p[@class='case_cite']"):
                    if case_cite.text is None:
                        continue
                    
                    if ' U.S. ' in case_cite.text:
                        scdb_match_records = scdb_df.loc[scdb_df.loc[:, 'usCite'] == case_cite.text, :]
                        if scdb_match_records.shape[0] == 0:
                            continue
                        elif scdb_match_records.shape[0] > 1:
                            print(f"multiple matching records for {case_cite.text}")
                        else:
                            scdb_id_match = scdb_match_records.index[0]
                    elif 'S. Ct. ' in case_cite.text:
                        scdb_match_records = scdb_df.loc[scdb_df.loc[:, 'sctCite'] == case_cite.text, :]
                        if scdb_match_records.shape[0] == 0:
                            continue
                        elif scdb_match_records.shape[0] > 1:
                            print(f"multiple matching records for {case_cite.text}")
                        else:
                            scdb_id_match = scdb_match_records.index[0]
                    elif 'L. Ed. ' in case_cite.text:
                        scdb_match_records = scdb_df.loc[scdb_df.loc[:, 'ledCite'] == case_cite.text, :]
                        if scdb_match_records.shape[0] == 0:
                            continue
                        elif scdb_match_records.shape[0] > 1:
                            print(f"multiple matching records for {case_cite.text}")
                        else:
                            scdb_id_match = scdb_match_records.index[0]

                if scdb_id_match is not None:
                    cl_scdb_mapping[member_id] = scdb_id_match
                else:
                    cl_scdb_mapping[member_id] = None

            except Exception as e:
                raise e
                print(f"unable to parse {member.name}['html'] as HTML", e)
                cl_scdb_mapping[member_id] = None

    # create dataframe and save
    cl_scdb_df = pandas.DataFrame.from_dict(cl_scdb_mapping, orient="index", columns=["scdb_case_id"])
    cl_scdb_df.to_csv(os.path.join(args.path, f"map-{args.year}_{args.release}-{args.court}_{args.data_type}.csv"),
                      encoding="utf-8", index_label="cl_id")
