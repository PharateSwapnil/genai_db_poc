"""Chainlit UI for assistant with rag."""

import chainlit as cl
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

# from askdata.chatbot import react_graph
from db_agent_2 import db_agent as react_graph

# @cl.on_message
# async def main(message: cl.Message):
#     # Your custom logic goes here...

#     # Send a response back to the user
#     await cl.Message(
#         content=f"Received: {message.content}",
#     ).send()

import pandas as pd
from chainlit import make_async

@cl.on_message
async def on_message(msg: cl.Message) -> cl.Message:
    """Message handeler.

    Args:
        msg (cl.Message): Input human message

    Returns:
        cl.Message: Final answer message
    """
    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="Processing your request ...")
    await final_answer.send()
    # --------------------------------
    def sync_fn():
        return react_graph.invoke({"question": msg.content})
    async_function = make_async(sync_fn)
    answer = await async_function()
    # --------------------------------
    if answer.get("plot"):
        fig = answer["plot"]
        df = answer["forecast_df"]
        elements = [
            cl.Plotly(name="chart", figure=fig, display="inline"),
            # cl.Dataframe(name="dataframe", dataframe=pd.DataFrame(df), display="inline")
        ]
        final_answer.elements = elements
    message = answer["messages"][-1]
    final_answer.content = message.content
    await final_answer.update()
    await final_answer.send()
    print("Done!")
    # --------------------------------
    # final_answer = cl.Message(content="")
    # for m, metadata in react_graph.stream({"question": msg.content}, stream_mode="messages", config=RunnableConfig(callbacks=[cb], **config)):
    #     if (
    #         m.content
    #         and not isinstance(m, HumanMessage)
    #         and metadata["langgraph_node"] ==  "forecast"
    #     ):
    #         print(m)
    #         print(metadata)
    #         await final_answer.stream_token(m.content)

    # await final_answer.send()
