from typing import Dict, Any
import os
from dotenv import load_dotenv
from rich import print
import typer
from sql_connector import SQLConnector


def get_tables_from_db(db_path: str) -> Dict[str, Any]:
    """Retrieve table names and their columns from the SQLite database."""

    
    '''
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    db_schema = {}
    for table_name_tuple in tables:
        table_name = table_name_tuple[0]
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        db_schema[table_name] = [col[1] for col in columns]  # col[1] is the column name
    conn.close()
    '''
  
    return db_schema

load_dotenv()



