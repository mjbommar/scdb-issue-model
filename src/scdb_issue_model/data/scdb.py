# Packages
import os
import pandas


def load_scdb_data(path: str, release_year: int, release_version: int, record: str = "case",
                   grouping: str = "citation"):
    scdb_file_name = f"{release_year}_{release_version:02}_{record.lower()}_{grouping.lower()}.csv"
    scdb_df = pandas.read_csv(os.path.join(path, scdb_file_name), low_memory="False", encoding="latin1",
                              index_col='caseId')
    scdb_df.loc[:, "yearDecided"] = pandas.to_datetime(scdb_df.loc[:, 'dateDecision']).apply(lambda x: x.year)
    return scdb_df
