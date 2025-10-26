

from sql_connector import SQLConnector



@tool
def get_table_schema(tablename: str) -> TableSchema:
    """Tool that returns the table schema Given a table name"""
    try:
        
        SQLConnector.execute_query("SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (tablename,))
        schema_sql = cursor.fetchone()
        if schema_sql:
            return schema_sql[0]
        else:
            return None
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return None
