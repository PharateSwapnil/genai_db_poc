import os
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from prompt_templates import FORECAST_PROMPT_TEMPLATE
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from forecast import generate_forecast_tool
from db_agent_2 import get_llm_model, index_json_to_chromadb, load_chromadb_from_path, \
        _create_retriever_tool, DBAgentState, get_db_info_str, \
        MODEL_TO_USE_DICT, HF_EMBEDDING_MODEL, CHROMADB_PERSIST_PATH, DATA_BASE_PATH

KB_PATHS = [os.path.join(DATA_BASE_PATH, "time_series", "datapackage.json"),]
forecast_prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template=FORECAST_PROMPT_TEMPLATE
)

import pandas as pd
DF_PATH = "/home/richhiey/Desktop/code/genai/data/time_series/time_series_60min_singleindex.csv"

def load_df(df_path=DF_PATH):
    return pd.read_csv(df_path)

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
        if last_message.tool_calls:
            return "tools"
        return "insights"

    def call_model(state: DBAgentState):
        prompt = forecast_prompt_template.invoke({
            "question": state["question"],
            "context": state["context"],
        })
        response = model_with_tools.invoke(prompt)
        return {"messages": [response]}

    def chronos_tool(state: DBAgentState):
        from forecast import generate_forecast
        # forecast = generate_forecast()

    tool_node = ToolNode(tools)
    workflow = StateGraph(DBAgentState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.add_node("forecast", chronos_tool)
    workflow.add_node("insights", insight_generator)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "agent")
    workflow.add_conditional_edges("agent", should_continue, ["forecast", "insights"])
    workflow.add_edge("forecast", "agent")
    workflow.add_edge("insights", END)
    app = workflow.compile()
    return app

if __name__ == "__main__":
    model = get_llm_model(MODEL_TO_USE_DICT)
    # --- Load embedding model ---
    embedding_model = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
    # --- Index knowledge base against DB metadata (if not done already) ---
    if not os.path.exists(CHROMADB_PERSIST_PATH):
        vector_db = index_json_to_chromadb(KB_PATHS, CHROMADB_PERSIST_PATH, embedding_model)
    else:
        vector_db = load_chromadb_from_path(CHROMADB_PERSIST_PATH, embedding_model)
    retriever_tool = _create_retriever_tool(vector_db)
    df = load_df()

    from langchain_experimental.agents import create_pandas_dataframe_agent
    question = "Forecast load data in Austria for the first week of December 2015"
    prompt = forecast_prompt_template.invoke({"question": question, "context": ""})
    model = model.bind_tools([generate_forecast_tool])
    agent_executor = create_pandas_dataframe_agent(model, df, verbose=True, allow_dangerous_code=True)
    outputs = agent_executor.run(prompt)
    print(outputs)
