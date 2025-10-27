from typing import Dict, Any
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, Tool
from langchain.tools import BaseTool
from rich import print
import typer

load_dotenv()

# Initialize Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = GoogleGenerativeAI(model="gemini-pro", temperature=0)

# Initialize database
db = SQLDatabase.from_uri(f"sqlite:///{os.getenv('DATABASE_PATH')}")

class SQLQueryGenerator:
    def __init__(self):
        self.tools = [
            QuerySQLDataBaseTool(db=db),
        ]
        
        self.prompt = PromptTemplate.from_template(
            """You are an expert SQL query generator. Your task is to convert English questions into SQL queries 
            and execute them against a database. Use the provided tools to query the database.
            
            Question: {input}
            
            Let's approach this step by step:
            1. Analyze the question
            2. Generate appropriate SQL query
            3. Execute the query
            4. Present the results in a clear format
            
            {agent_scratchpad}"""
        )
        
    def run(self, question: str) -> Dict[str, Any]:
        agent = AgentExecutor.from_agent_and_tools(
            llm=llm,
            tools=self.tools,
            prompt=self.prompt,
            output_parser=ReActSingleInputOutputParser(),
            format_scratchpad=format_log_to_str,
            verbose=True,
        )
        
        return agent.invoke({"input": question})

app = typer.Typer()

@app.command()
def query(question: str):
    """Convert an English question to SQL and get the answer."""
    generator = SQLQueryGenerator()
    result = generator.run(question)
    print("[bold green]Result:[/bold green]")
    print(result)





if __name__ == "__main__":
    app()
