import apsw


class DatabaseToolbox:

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

    def as_dataframe(self, query, **kwargs):
        """Get the results of a query as a pandas DataFrame.
    
        kwargs are given to pd.read_sql_query.
        
        Catches the ExecutionCompleteError that arises when no rows are selected by
        the query and pd.read_sql_query cannot access the description attribute of
        the cursor. See: https://github.com/rogerbinns/apsw/issues/160.
        """
        import pandas as pd
        try:
            return pd.read_sql_query(query, self.connection)
        except apsw.ExecutionCompleteError:
            return None

    def create_tables(self):
        self.execute(scheme)


scheme = """
CREATE TABLE IF NOT EXISTS raso(
    id INTEGER PRIMARY KEY,
    valid NUMERIC,
    metadata JSON
);

CREATE TABLE IF NOT EXISTS rasodata(
    id INTEGER PRIMARY KEY,
    parent INTEGER REFERENCES raso(id),
    p NUMERIC,
    z NUMERIC,
    T NUMERIC,
    Td NUMERIC
);

CREATE TABLE IF NOT EXISTS cosmo7(
    id INTEGER PRIMARY KEY,
    valid NUMERIC,
    run NUMERIC,
    metadata JSON
);

CREATE TABLE IF NOT EXISTS cosmo7data(
    id INTEGER PRIMARY KEY,
    parent INTEGER REFERENCES cosmo7(id),
    p NUMERIC,
    z NUMERIC,
    T NUMERIC,
    Td NUMERIC,
    q NUMERIC,
    qcld NUMERIC
);

CREATE TABLE IF NOT EXISTS hatpro(
    id INTEGER PRIMARY KEY,
    kind TEXT, 
    angle NUMERIC,
    f22240 NUMERIC,
    f23040 NUMERIC,
    f23840 NUMERIC,
    f25440 NUMERIC,
    f26240 NUMERIC,
    f27840 NUMERIC,
    f31400 NUMERIC,
    f51260 NUMERIC,
    f52280 NUMERIC,
    f53860 NUMERIC,
    f54940 NUMERIC,
    f56660 NUMERIC,
    f57300 NUMERIC,
    f58000 NUMERIC,
    T NUMERIC,
    p NUMERIC,
    RH NUMERIC,
    ff NUMERIC,
    dd NUMERIC,
    precip INTEGER
);
"""
