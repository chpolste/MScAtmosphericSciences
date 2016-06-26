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

