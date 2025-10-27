import os
from typing import List, TypedDict
from pprint import pprint

import sqlite3

class SQLConnector:
    def __init__(self, db: str):
        try:
            self.conn = sqlite3.connect(db)
        except sqlite3.OperationalError as e:
            print("Failed to open database:", e)
    def get_tables(self) -> dict:
        """Retrieve table names and their columns from the SQLite database."""
        db_schema = {}
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            for table_name_tuple in tables:
                table_name = table_name_tuple[0]
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns_info = cursor.fetchall()
                columns = [col[1] for col in columns_info]
                db_schema[table_name] = columns
        except sqlite3.Error as e:
            print("An error occurred while retrieving the database schema:", e)

        return db_schema

    def execute_query(self, sql: str):
        """Execute a SQL query on the database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(sql)
            self.conn.commit()
            return cursor.fetchall()
        except sqlite3.Error as e:
            print("An error occurred while executing the query:", e)
            return None