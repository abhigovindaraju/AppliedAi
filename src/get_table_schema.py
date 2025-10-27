

from sql_connector import SQLConnector
import pandas as pd



@tool
def get_table_schema(tablename: str) -> TableSchema:
    """Tool that returns the table schema Given a table name"""
    try:
        
        data = SQLConnector.execute_query("SELECT sql FROM sqlite_master WHERE type='table' AND name=?;", (tablename,))
        if data:
            return data.iloc[0]
        else:
            return None 
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return None
