"""
Langgraph agent for generating forecasts given a time-series database
"""

import os
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase

from forecast import generate_forecast, plot_forecasts
from db_agent_2 import get_llm_model, index_json_to_chromadb, load_chromadb_from_path, \
                        _create_retriever_tool, load_database, get_db_info_str, DBAgentState, \
                        MODEL_TO_USE_DICT, HF_EMBEDDING_MODEL, CHROMADB_PERSIST_PATH, DATA_BASE_PATH
from prompt_templates import FORECAST_PROMPT_TEMPLATE, \
                            SQL_QUERY_AGENT_PROMPT_MESSAGE, \
                            SQL_QUERY_PARSER_PROMPT_MESSAGE

TOP_K_VALS = 10000
DB_PATH = "/home/richhiey/Desktop/code/genai/data/time_series/time_series.sqlite"
KB_PATHS = [os.path.join(DATA_BASE_PATH, "time_series", "datapackage.json"),]
db_engine = load_database(DB_PATH)
db = SQLDatabase(db_engine)

class ForecastInformation(BaseModel):
    """
    Placeholder class to forecast related information extracted from the input message given by a user.
    In order to run a forecast, we need three specific variables:
    sql_query: The SQL query to retrieve historical data from the database
    start_date: The start date of the forecast period provided in the input message
    horizon: The forecast horizon in hours provided in the input message
    """
    sql_query: str = Field(description="The SQL query to retrieve historical data from the database")
    start_date: str = Field(description="The start date of the forecast period provided in the input message")
    horizon: int = Field(
        default=240, description="The time horizon in hours provided in the input message"
    ) # default horizon is set to 10 days
    column_names: List = Field(
        description="Names of the columns included in the SELECT statement of the SQL query, returned in the same order as seen in the SELECT statement of the SQL query"
    )
    

sql_query_agent_prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template=SQL_QUERY_AGENT_PROMPT_MESSAGE
)

def fetch_results_from_db(sql_query):
    return eval(db.run(sql_query, include_columns=True))

def create_forecast_agent_graph(model_with_tools, tools, retrieval_tool, prompt_template):
    def retrieve(state: DBAgentState):
        question = state["question"]
        retrieved_outputs = retrieval_tool.invoke(question)
        return {"context": retrieved_outputs}

    def should_continue(state: DBAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "db_tools"
        return "forecast"

    def call_model(state: DBAgentState):
        prompt = prompt_template.invoke({
            "question": state["question"],
            "context": state["context"],
        })
        response = model_with_tools.invoke(prompt)
        return {"messages": [response]}

    def forecast(state: DBAgentState):
        messages = state["messages"]
        last_message = messages[-1]
        sql_query_parser = PromptTemplate(
            input_variables=["message"],
            template=SQL_QUERY_PARSER_PROMPT_MESSAGE
        )
        parser = PydanticOutputParser(pydantic_object=ForecastInformation)
        print(parser.get_format_instructions())
        response = model_with_tools.invoke(
            sql_query_parser.invoke({
                "message": last_message,
                "format_instructions": parser.get_format_instructions()
            })
        )
        response = dict(parser.invoke(response))
        print(response)
        db_results = fetch_results_from_db(response["sql_query"])
        df_results = pd.DataFrame(db_results)
        forecast = generate_forecast(
            df_results, response["start_date"], response["horizon"], response["column_names"]
        )
        df_forecast = pd.DataFrame(forecast)
        forecast_start_dt = pd.to_datetime(response["start_date"], utc=True)
        forecast_end_dt = forecast_start_dt + pd.Timedelta(hours=response["horizon"])
        fig = plot_forecasts(
            forecast,
            forecast_start_dt.strftime("%Y-%m-%d"),
            forecast_end_dt.strftime("%Y-%m-%d")
        )
        print(df_forecast)
        return {"forecast_df": df_forecast, "plot": fig, "messages": [last_message]}

    tool_node = ToolNode(tools)
    workflow = StateGraph(DBAgentState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("data_agent", call_model)
    workflow.add_node("db_tools", tool_node)
    workflow.add_node("forecast", forecast)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "data_agent")
    workflow.add_conditional_edges("data_agent", should_continue, ["db_tools", "forecast"])
    workflow.add_edge("db_tools", "data_agent")
    workflow.add_edge("forecast", END)
    app = workflow.compile()
    return app

def plot_db_agent_graph(db_agent):
    from IPython.display import Image, display
    import matplotlib.pyplot as plt
    try:
        img = Image(db_agent.get_graph().draw_mermaid_png(output_file_path="graph.png"))
    except Exception:
        pass

def create_forecast_agent():
    model = get_llm_model(MODEL_TO_USE_DICT)
    # --- Load embedding model ---
    embedding_model = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
    # --- Index knowledge base against DB metadata (if not done already) ---
    if not os.path.exists(CHROMADB_PERSIST_PATH):
        vector_db = index_json_to_chromadb(KB_PATHS, CHROMADB_PERSIST_PATH, embedding_model)
    else:
        vector_db = load_chromadb_from_path(CHROMADB_PERSIST_PATH, embedding_model)
    retriever_tool = _create_retriever_tool(vector_db)
    # --- Load database and DB toolkit for langgraph ---
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    tools = toolkit.get_tools()
    model = model.bind_tools(tools)
    table_names, column_info = get_db_info_str(db_path=DB_PATH, time_series=True)
    print(column_info)
    partial_prompt_template = sql_query_agent_prompt_template.partial(
        dialect="sqlite", table_names_str=table_names, column_info_str=column_info
    )
    db_agent = create_forecast_agent_graph(model, tools, retriever_tool, partial_prompt_template)
    plot_db_agent_graph(db_agent)
    return db_agent

agent = create_forecast_agent()

if __name__ == "__main__":
    question = """
    Create a load forecast for Austria for the first week of December 2015.
    Retrieve data from one year before the start week and forecast load data for a horizon of the next 10 days after the start week.
    Choose the database containing datapoints at an interval of 1 hour.
    """
    answer = agent.invoke({"question": question})
    print(answer)
