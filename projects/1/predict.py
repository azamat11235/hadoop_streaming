#!/opt/conda/envs/dsenv/bin/python

import sys, os
import logging
from joblib import load
import pandas as pd

sys.path.append('.')
from model import fields

#
# Init the logger
#
logging.basicConfig(level=logging.DEBUG)
logging.info("CURRENT_DIR {}".format(os.getcwd()))
logging.info("SCRIPT CALLED AS {}".format(sys.argv[0]))
logging.info("ARGS {}".format(sys.argv[1:]))

#load the model
model = load("1.joblib")

fields = [fields[0]] + fields[2:]
#read and infere
read_opts=dict(
        sep='\t', names=fields, index_col=False, header=None,
        iterator=True, chunksize=10000
)

for df in pd.read_csv(sys.stdin, **read_opts):
    logging.info(f"DF_type {type(df)}")
    logging.info(f"DF_cols {df.columns}")
    logging.info(f"DF_row {df.iloc[1]}")
    pred = model.predict_proba(df)[:, 1]
    out = zip(df.id, pred)
    print("\n".join(["{0}\t{1}".format(*i) for i in out]))
