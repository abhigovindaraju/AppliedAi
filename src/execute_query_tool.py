from langchain_core.tools import tool
from sql_connector import SQLConnector

load_dotenv()

db_path = os.getenv("DATABASE_PATH")
if db_path is None:
    raise ValueError("DATABASE_PATH environment variable is not set")

connector = SQLConnector(db_path)

@tool
def execute_query(sql: str) -> str:
    """
    Executes the given SQL query and returns the results as a string.

    Args:
        sql (str): The SQL query to execute.
    """
    results = connector.execute_query(sql)
    result_str = ""
    for chunk in results:
        result_str += chunk.to_string(index=False) + "\n"
    return result_str