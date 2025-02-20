"""Chainlit UI for assistant with rag."""

import chainlit as cl
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

# from askdata.chatbot import react_graph
from df_forecast_agent import agent as react_graph

# @cl.on_message
# async def main(message: cl.Message):
#     # Your custom logic goes here...

#     # Send a response back to the user
#     await cl.Message(
#         content=f"Received: {message.content}",
#     ).send()

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
    # --------------------------------
    # answer = react_graph.invoke({"question": msg.content})
    # # --------------------------------
    # fig = answer["plot"]
    # df = answer["forecast_df"]
    # message = answer["messages"][-1]
    # elements = [
    #     cl.Plotly(name="chart", figure=fig, display="inline"),
    #     # cl.DataFrame(name="dataframe", dataframe=df, display="inline")
    # ]
    # print(message)
    # print(elements)
    # print(final_answer.content)
    # await final_answer.send()
    # --------------------------------
    final_answer = cl.Message(content="")
    for m, metadata in react_graph.stream({"question": msg.content}, stream_mode="messages", config=RunnableConfig(callbacks=[cb], **config)):
        if (
            m.content
            and not isinstance(m, HumanMessage)
            and metadata["langgraph_node"] ==  "forecast"
        ):
            print(m)
            print(metadata)
            await final_answer.stream_token(m.content)

    await final_answer.send()
