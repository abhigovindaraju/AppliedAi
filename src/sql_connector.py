import os
from typing import List, TypedDict

os.environ['GRPC_VERBOSITY'] = 'NONE'

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.documents import Document
from uuid import uuid4

from langgraph.graph import END, StateGraph
from pprint import pprint

import sqlite3
import pandas as pd

class SQLConnector:
    def __init__(self, db: str):
        try:
            self.conn = sqlite3.connect(db)
        except sqlite3.OperationalError as e:
            print("Failed to open database:", e)
    def execute_query(self, sql: str) -> pd.DataFrame:
        return pd.read_sql_query(sql=sql, con=self.conn, chunksize=5)
        