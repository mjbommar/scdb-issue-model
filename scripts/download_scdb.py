# imports
import argparse
import datetime
import io
import os
import pandas
import requests
import zipfile


def get_release_url(base_url: str, release_year: int, release_version: int, record: str = "case",
                    grouping: str = "Citation") -> str:
    """
    Get a release URL in SCDB website format.
    :param base_url: base URL for current SCDB site, including protocol
    :param release_year: year of database release
    :param release_version: version of database release
    :param record: record type; case or justice
    :param grouping: grouping type (case sensitive); Citation, Docket, LegalProvision, or Vote
    :return: str, URL
    """
    return f"{base_url.rstrip('/')}/{release_year:02}_{release_version:02}/" + \
           f"SCDB_{release_year:02}_{release_version:02}_{record}Centered_{grouping}.csv.zip"


def extract_csv_from_release(release_url: str) -> bytes:
    """
    Get the CSV database contents from release URL.
    :param release_url:
    :return:
    """
    with requests.get(release_url) as req:
        with zipfile.ZipFile(file=io.BytesIO(req.content)) as release_zip:
            for file_name in release_zip.namelist():
                if file_name.lower().endswith(".csv"):
                    return release_zip.read(file_name)


if __name__ == "__main__":
    # setup command line argument parser
    arg_parser = argparse.ArgumentParser(description="retrieve and extract SCDB data")
    arg_parser.add_argument("--year", type=int, required=False, help="SCDB release year; defaults to prior year")
    arg_parser.add_argument("--release", type=int, required=False, default=1,
                            help="SCDB release version; defaults to 1")
    arg_parser.add_argument("--record", type=str, required=False, default="case",
                            help="SCDB record type; case or justice")
    arg_parser.add_argument("--grouping", type=str, required=False, default="Citation",
                            help="SCDB grouping type (CASE SENSITIVE); Citation, Docket, LegalProvision, or Vote")
    arg_parser.add_argument("--base_url", type=str, required=False, default="http://scdb.wustl.edu/_brickFiles/")
    arg_parser.add_argument("--path", type=str, required=False, default="data", help="path to save CSV data")

    # get args and handle defaults
    args = arg_parser.parse_args()

    # if release year is not provided, default to *last* year because coding is not completed until the term is over and
    # the coding release process is completed.
    if args.year is None:
        args.year = datetime.date.today().year - 1

    # check if we already have it downloaded
    output_file_name = f"{args.year}_{args.release:02}_{args.record.lower()}_{args.grouping.lower()}.csv"
    if os.path.exists(os.path.join(args.path, output_file_name)):
        raise RuntimeError(f"{output_file_name} already exists; please delete if you would like to re-download")

    # get url
    url = get_release_url(args.base_url, args.year, args.release, args.record, args.grouping)
    print(f"downloading year={args.year}, release={args.release}, record={args.record},"
          f"grouping={args.grouping} from {url}")

    # download and store
    csv_bytes = extract_csv_from_release(url)
    print(f"retrieved {len(csv_bytes)} bytes of CSV data")

    # make sure it's clean with pandas and some arbitrary thresholds
    csv_data = pandas.read_csv(io.BytesIO(csv_bytes), encoding="latin1", low_memory=False)
    if csv_data.shape[0] < 100 or csv_data.shape[1] < 10:
        raise RuntimeError("unexpected CSV data encountered; please review release URL")

    print(f"CSV data contains {csv_data.shape[0]} rows, {csv_data.shape[1]} columns")

    with open(os.path.join(args.path, output_file_name), "wb") as output_file:
        output_file.write(csv_bytes)
    print(f"saved to {output_file_name}")
