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
from langgraph.prebuilt import ToolNode


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
        improvements: any improvement suggestions
        iteration: iteration number
    """
    question: str
    generation: str
    documents: List[str]
    improvements: str
    iteration: int
    

class GraphContext(TypedDict):
    """
    Represents the context of our graph.

    Attributes:
        generate_model: model used in generate node
        evaluate_model: model used in evaluate node
    """
    generate_model: str
    evaluate_model: str



def setup_graph():
    workflow = StateGraph(GraphState)

    tool_node = 
    # Define the nodes
    workflow.add_node("text2sql", text2sql)
    workflow.add_node("generate", generate)
    workflow.add_node("evaluate", evaluate)

    # Build graph
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "evaluate")
    workflow.add_conditional_edges(
        "evaluate",
        decide_to_finish,
        {
            "continue": "generate",
            "finish": END,
        },
    )

    app = workflow.compile()
    
    inputs = {"question": "Give me a 10 sentence summary on fad diets"}

    for output in app.stream(inputs, context={
        "generate_model": "models/gemini-2.5-pro-preview-03-25", 
        "evaluate_model": "models/gemini-2.5-pro-preview-03-25"
        }):
        for okey, ovalue in output.items():
            print(f"Output from node '{okey}':")
            print("---")
            for ikey, ivalue in ovalue.items():
                if ikey == "documents":
                    continue
                pprint(f"{ikey}:{ivalue}")
            print("\n---\n")
        
    pprint(ovalue['generation'])


if __name__ == "__main__":
    main()