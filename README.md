# SCDB Issue Modeling

This research project attempts to develop models to assist in the prediction of Supreme Court cases; initially, the
goal is to predict the issue and issue area coding of a case.

Due to the _ex post_ nature of SCDB coding and its focus on the published opinion, there are a number of challenges to
out-of-sample, _ex ante_ application.  However, we will attempt to evaluate the degree of the problem through a historical
walk-forward simulation of such a model to _ex ante_ documents such as lower court opinions, briefs, and docket entries.

# Replication

### Step 1: Download SCOTUS opinions from CourtListener
```
$ python scripts/download_courtlistener.py --data_type opinions
downloading court=scotus, data_type=opinions from https://www.courtlistener.com/api/bulk-data/opinions/scotustar.gz
wrote 709371554 bytes to scotus_opinions.tar.gz
```

### Step 2: Download SCDB release
```
$ python scripts/download_scdb.py 
downloading year=2020, release=1, record=case,grouping=Citation from http://scdb.wustl.edu/_brickFiles/2020_01/SCDB_2020_01_caseCentered_Citation.csv.zip
retrieved 2917231 bytes of CSV data
CSV data contains 9030 rows, 53 columns
saved to 2020_01_case_citation.csv
```

### Step 3: Map SCDB caseId to CourtListener document IDs
```
$ export PYTHONPATH=src
$ python scripts/map_courtlistener_scdb.py
multiple matching records for 352 U.S. 862
multiple matching records for 415 U.S. 289
multiple matching records for 415 U.S. 289
multiple matching records for 352 U.S. 862
```

### Step 4: Build doc2vec model using full dataset
```
$ python scripts/build_doc2vec_model.py
```

### Step 5: Build walkforward, time-indexed doc2vec models
```
$ python scripts/build_doc2vec_model.py
```

### Step 6: Compute doc2vec model vectors


### Step 7: Compute GloVe and (Al)BERT model tensors/vectors



