"""Convenience tools for interacting with the database or csv files."""

import os
import apsw
from collections import OrderedDict

import pandas as pd


def read_csv_profiles(file):
    return pd.read_csv(file, parse_dates=["valid"], index_col="valid")

def read_csv_covariance(file):
    return pd.read_csv(file, index_col=0)

def read_csv_mean(file):
    df = pd.read_csv(file, index_col=0)
    return df[df.columns[0]]


def iter_profiles(pattern, tryvars=("bt", "p", "T", "qvap", "qliq", "lnq")):
    """Iterate over profiles obtained from multiple csv files.
    
    Assumes that columns and index are identical.
    """
    assert "<VAR>" in pattern
    variables = [var for var in tryvars if os.path.exists(pattern.replace("<VAR>", var))]
    assert len(variables) > 0
    data = OrderedDict()
    index = None
    for var in variables:
        data[var] = read_csv_profiles(pattern.replace("<VAR>", var))
        if index is None:
            index = data[var].index
    for valid in index:
        df = pd.concat([d.loc[valid] for d in data.values()], axis=1)
        df.columns = list(data.keys())
        yield valid, df


def split_bands(df):
    kband, vband = df.copy(), df.copy()
    if df.index.dtype == "O" and any("TB_" in idx for idx in df.index):
        kband = kband.ix[[idx for idx in kband.index if int(idx[3:8]) < 40000]]
        vband = vband.ix[[idx for idx in vband.index if int(idx[3:8]) > 40000]]
    if df.columns.dtype == "O" and any("TB_" in idx for idx in df.columns):
        kband = kband.drop([idx for idx in kband.columns if int(idx[3:8]) > 40000], axis=1)
        vband = vband.drop([idx for idx in vband.columns if int(idx[3:8]) < 40000], axis=1)
    return kband, vband


class Database:

    def __init__(self, database):
        self.connection = apsw.Connection(database)

    def execute(self, query, data=None):
        if data is None: data = []
        with self.connection as db:
            return self.connection.cursor().execute(query, data)

    def executemany(self, query, data=None):
        if data is None: raise ValueError("data must contain at least one row.")
        with self.connection as db:
            return self.connection.cursor().executemany(query, data)

    def lastid(self):
        return self.connection.last_insert_rowid()

    def as_dataframe(self, query):
        """Get the results of a query as a pandas DataFrame.
    
        kwargs are given to pd.read_sql_query.
        
        Catches the ExecutionCompleteError that arises when no rows are selected by
        the query and pd.read_sql_query cannot access the description attribute of
        the cursor. See: https://github.com/rogerbinns/apsw/issues/160.
        """
        try:
            return pd.read_sql_query(query, self.connection)
        except apsw.ExecutionCompleteError:
            return None

    def create_tables(self):
        self.execute(scheme)

