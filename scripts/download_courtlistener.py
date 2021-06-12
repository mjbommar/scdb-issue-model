# https://www.courtlistener.com/api/bulk-data/opinions/all.tar
# https://www.courtlistener.com/api/bulk-data/dockets/all.tar

# imports
import argparse
import datetime
import io
import os
import pandas
import requests
import zipfile

REQUESTS_CHUNK_SIZE = 2 ** 16


def get_data_url(base_url: str, court: str, data_type: str) -> str:
    """
    Get the URL for a bulk data source.
    :param base_url:
    :param court:
    :param data_type:
    :return:
    """

    # get the extension based on arguments
    if data_type in ["opinions", "dockets"] and court == "all":
        extension = "tar"
    else:
        extension = "tar.gz"

    return f"{base_url.rstrip('/')}/{data_type}/{court}.{extension}"


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
    arg_parser = argparse.ArgumentParser(description="retrieve and extract Court Listener data")
    arg_parser.add_argument("--data_type", type=str, required=True, help="type of data; opinions, audio, dockets")
    arg_parser.add_argument("--court", type=str, required=False, default="scotus",
                            help="court ID; see https://www.courtlistener.com/api/bulk-info/")
    arg_parser.add_argument("--base_url", type=str, required=False,
                            default="https://www.courtlistener.com/api/bulk-data/")
    arg_parser.add_argument("--path", type=str, required=False, default="data", help="path to save CSV data")

    # get args and handle defaults
    args = arg_parser.parse_args()

    # get url
    url = get_data_url(args.base_url, args.court, args.data_type)
    print(f"downloading court={args.court}, data_type={args.data_type} from {url}")

    # setup output file name
    output_file_extension = os.path.splitext(url)[1].lstrip('.')
    output_file_name = f"{args.court}_{args.data_type}.{output_file_extension}"
    if os.path.exists(os.path.join(args.path, output_file_name)):
        raise RuntimeError(f"{output_file_name} already exists; please delete if you would like to re-download")

    # stream contents to file and count bytes along the way
    output_file_size = 0
    with open(os.path.join(args.path, output_file_name), 'wb') as output_file:
        with requests.get(url, stream=True) as req:
            for stream_buffer in req.iter_content(chunk_size=REQUESTS_CHUNK_SIZE):
                req.raise_for_status()
                output_file.write(stream_buffer)
                output_file_size += len(stream_buffer)

    print(f"wrote {output_file_size} bytes to {output_file_name}")
