from typing import Dict, Any
import os
from dotenv import load_dotenv
from rich import print
import typer
from sql_connector import SQLConnector
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, Tool
from langchain.tools import BaseTool, tool


load_dotenv()

db_path = os.getenv("DATABASE_PATH")
if db_path is None:
    raise ValueError("DATABASE_PATH environment variable is not set")

connector = SQLConnector(db_path)

@tool
def get_tables(tool_input: str = "") -> Dict[str, Any]:
    """Retrieve table names and their columns from the SQLite database."""
    if not db_path:
        raise ValueError("Database path must be provided")
    
    db_schema = connector.get_tables()
    return db_schema


# Call the tool with an empty string as input
print(get_tables(""))
