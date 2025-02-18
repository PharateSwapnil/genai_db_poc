"""Database Agent."""

import os
import json
import copy
import itertools

from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm

from langchain import hub
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase

import sqlite3
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import ToolNode
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from prompt_templates import SYSTEM_PROMPT_MESSAGE, INSIGHTS_PROMPT_TEMPLATE

# ---------------- CONSTANTS ----------------------
HF_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
# --------------------------------------
CHROMADB_PERSIST_PATH = "./chromadb/"
DATA_BASE_PATH = "/home/richhiey/Desktop/code/genai/data"
DB_PATH = os.path.join(DATA_BASE_PATH, "time_series", "time_series.sqlite")
KB_PATHS = [os.path.join(DATA_BASE_PATH, "time_series", "datapackage.json"),]
# --------------------------------------
MODEL_TO_USE_DICT = {
    "provider": "groq",
    "name": "llama3-8b-8192"
}
MODEL_TO_USE_DICT = {
    "provider": "ollama",
    "name": "llama3-groq-tool-use:latest"
}
MODEL_TO_USE_DICT = {
    "provider": "huggingface",
    "name": "HuggingFaceH4/zephyr-7b-gemma-v0.1"
}
MODEL_TO_USE_DICT = {
    "provider": "google",
    "name": "gemini-2.0-flash"
}
# --------------------------------------

# ----------------- Prompt templates ---------------------
system_prompt_template = PromptTemplate(
    input_variables=["dialect", "top_k"],
    template=SYSTEM_PROMPT_MESSAGE
)
insights_prompt_template = PromptTemplate(
    input_variables=["question", "answer", "context"],
    template=INSIGHTS_PROMPT_TEMPLATE
)
extract_final_answer_template = PromptTemplate(
    input_variables=["question"],
    template="Extract text data from the given question within the section Final Answer: and return that as a piece of text"
)
# --------------------------------------

def get_llm_model(model_dict=MODEL_TO_USE_DICT):
    if model_dict["provider"] == "huggingface":
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        llm = HuggingFaceEndpoint(
            repo_id=model_dict["name"],
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        )
        model = ChatHuggingFace(llm=llm) # model_dict["name"])
    if model_dict["provider"] == "groq":
        from langchain_groq import ChatGroq
        model = ChatGroq(model=model_dict["name"])
    if model_dict["provider"] == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = ChatGoogleGenerativeAI(model=model_dict["name"])
    else:
        from langchain_ollama import ChatOllama
        model = ChatOllama(model=model_dict["name"])

    return model

def get_engine_power_plant() -> Engine:
    """Engine for opsd data."""
    return create_engine(
        f"sqlite:///{DB_PATH}", poolclass=StaticPool
    )

def filter_metadata(metadata):
    filtered_metadata = {}
    # filtered_metadata["name"] = metadata["name"]
    filtered_metadata["title"] = metadata["title"]
    filtered_metadata["description"] = metadata["description"]
    # filtered_metadata["longDescription"] = metadata["longDescription"]
    filtered_metadata["keywords"] = metadata["keywords"]
    if metadata.get("geographicalScope"):
        filtered_metadata["geographicalScope"] = metadata["geographicalScope"]
    if metadata.get("temporalScope"):
        filtered_metadata["temporalScope"] = metadata["temporalScope"]
    resources =  metadata["resources"]
    filtered_resources = []
    for resource in resources:
        if resource.get("schema") and resource.get("profile") == "tabular-data-resource":
            filtered_resource = {}
            filtered_resource["schema"] = resource["schema"]
            if resource.get("profile"):
                filtered_resource["profile"] = resource["profile"]
            if resource.get("title"):
                filtered_resource["title"] = resource["title"]
            if resource.get("name"):
                filtered_resource["name"] = resource["name"]
            filtered_resources.append(filtered_resource)
    filtered_metadata["resources"] = filtered_resources
    return filtered_metadata

def read_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data

def split_json_custom(json_data):
    resources =  json_data["resources"]
    filtered_resources = []
    for resource in resources:
        if resource.get("schema"): # and resource.get("profile") == "tabular-data-resource":
            filtered_resource = {}
            if resource.get("title"):
                filtered_resource["table_description"] = resource["title"]
            if resource.get("name"):
                filtered_resource["table_name"] = resource["name"]
            resource_schema = resource["schema"]
            resource_schema_fields = resource_schema["fields"]
            filtered_resource["primary_key"] = resource_schema["primaryKey"]
            for field in resource_schema_fields:
                filt_resource_copy = copy.deepcopy(filtered_resource)
                field_resource = filt_resource_copy | field
                field_resource["column_name"] = field_resource.pop("name")
                filtered_resources.append(field_resource)
    return filtered_resources


# RAG utilities
def index_json_to_chromadb(kb_paths, db_persist_path, embedding_model):
    print("Indexing database json metadata to ChromaDB ...")
    documents = [filter_metadata(read_json(kb_path)) for kb_path in kb_paths]
    split_documents = [split_json_custom(json_data=doc) for doc in documents]
    splits = list(itertools.chain(*split_documents))
    splits = [Document(page_content=str(doc)) for doc in tqdm(splits)]
    chroma_vector_database = Chroma.from_documents(
        persist_directory=CHROMADB_PERSIST_PATH,
        documents=splits,
        embedding=embedding_model
    )
    return chroma_vector_database

def load_chromadb_from_path(db_save_path, embedding_model):
    vector_db = Chroma(persist_directory=db_save_path, embedding_function=embedding_model)
    return vector_db

def _create_retriever_tool(vector_db):
    retriever = vector_db.as_retriever(search_kwargs={"k":10})
    retriever_tool = create_retriever_tool(
        retriever=retriever,
        name="search_db_metadata",
        description="Tool to retrieve relevant text chunks from a vector database"
    )
    return retriever_tool

class DBAgentState(TypedDict):
    question: str
    context: str
    messages: str

def create_db_agent_graph(model_with_tools, tools, retrieval_tool, prompt_template):
    def insight_generator(state: DBAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        insights_prompt = insights_prompt_template.invoke({
            "question": state["question"],
            "answer": last_message.content,
            "context": state["context"],
        })
        final_response = model_with_tools.invoke(insights_prompt)
        return {"messages": [final_response]}

    def retrieve(state: DBAgentState):
        question = state["question"]
        retrieved_outputs = retrieval_tool.invoke(question)
        return {"context": retrieved_outputs} 

    def should_continue(state: DBAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        # print(last_message.tool_calls)
        if last_message.tool_calls:
            return "tools"
        return "insights"

    def call_model(state: DBAgentState):
        prompt = prompt_template.invoke({
            "question": state["question"],
            "context": state["context"],
        })
        response = model_with_tools.invoke(prompt)
        # print(response)
        return {"messages": [response]}

    tool_node = ToolNode(tools)
    workflow = StateGraph(DBAgentState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_node("insights", insight_generator)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", "insights"])
    workflow.add_edge("tools", "agent")
    workflow.add_edge("insights", END)
    app = workflow.compile()
    return app

def get_db_info_str():
    # Connect to the SQLite database
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    # Query to get all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Extract table names from the result
    table_names = [table[0] for table in tables]
    table_names_str = f"Tables: {table_names}"
    # print("Tables:", table_names)

    # Function to get column names for a given table
    def get_column_names(table_name):
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        return column_names

    column_info_str = ""
    # Get column names for each table
    for table in table_names:
        columns = get_column_names(table)
        column_info_str += f"Columns in {table}: {columns}"
    print(table_names)
    # print(column_info_str)
    return table_names, column_info_str

def create_db_agent():
    # --- Load LLM model ---
    model = get_llm_model()
    # --- Load database and Langchain DB toolkit  --- 
    engine = get_engine_power_plant()
    db = SQLDatabase(engine)
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    # --- Load embedding model ---
    embedding_model = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
    # --- Index knowledge base against DB metadata (if not done already) ---
    if not os.path.exists(CHROMADB_PERSIST_PATH):
        vector_db = index_json_to_chromadb(KB_PATHS, CHROMADB_PERSIST_PATH, embedding_model)
    else:
        vector_db = load_chromadb_from_path(CHROMADB_PERSIST_PATH, embedding_model)
    retriever_tool = _create_retriever_tool(vector_db)
    # --- Create an executor for a ReAct agent using the llm, prompt template and DB tools ---
    tools = toolkit.get_tools()
    model = model.bind_tools(tools)

    table_names, column_info =  get_db_info_str()
    partial_prompt_template = system_prompt_template.partial(
        dialect="sqlite", top_k=10, table_names_str=table_names, column_info_str=""
    )
    db_agent = create_db_agent_graph(model, tools, retriever_tool, partial_prompt_template)
    from IPython.display import Image, display
    import matplotlib.pyplot as plt
    try:
        img = Image(db_agent.get_graph().draw_mermaid_png(output_file_path="graph.png"))
    except Exception:
        # This requires some extra dependencies and is optional
        pass
    # ---
    return db_agent

db_agent = create_db_agent()

if __name__ == "__main__":
    messages = db_agent.invoke({
        "question": "Forecast the load in Austria for next week."
    })
    print(messages["messages"])



# ---------------------------------------------------------------------------------
# Questions for agent about conventional_power_plants_DE dataset
# ---------------------------------------------------------------------------------
# "question": "Hi",
# "question": "Can you list all the tables you have access to?",
# "question": "List down the number of power plants by country and state in Germany."
# "question": "List the amount of capacity installed for each type of technology in Germany from conventional_power_plants_DE?",
# "question": "List different types of technologies used across different power plants and the total capacity for each specific technology?"
# "question": "List the the total capacity of all renewable energy sources installed within Germany, France and Poland?"
# "question": "List down the total installed capacity (net and gross) for each energy source in Germany."
# "question": "List down the total installed capacity (net and gross) for each energy source in Germany."
# "question": "Find the plant with the highest efficiency for each technology type in the EU?"
# "question": "Track the total number of plants commissioned in the european union in each decade from 1960 to 2020?"
# "question": "Track the total number of power plants decommisioned in Germany from 2010 to 2020?"
# "question": "Identify power plants located in cities where the number of plants exceeds the average per city?"
# ---------------------------------------------------------------------------------
