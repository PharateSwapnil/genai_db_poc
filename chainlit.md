## Welcome to SQL Data Agent! 🤖

Hi there! This is a SQL Data Agent to assist you in fetching data and related insights from an attached relational database. The databases available to the agent are the conventional power plants and time-series datasets from OPSD available [here](https://data.open-power-system-data.org/conventional_power_plants/).

This [Time-series database]() contains different kinds of timeseries data relevant for power system modelling, namely electricity prices, electricity consumption (load) as well as wind and solar power generation and capacities. The data is aggregated either by country, control area or bidding zone. Geographical coverage includes the EU and some neighbouring countries. The [Conventional Power Plants database]() contains data on conventional power plants for Germany as well as other selected European countries. The data includes individual power plants with their technical characteristics. These include installed capacity, main energy source, type of technology, CHP capability, and geographical information. The geographical scope is primarily on Germany and its neighboring countries. 

The technologies used to set this agent up are [LangGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/), [Google Gemini](https://gemini.google.com/app?hl=en-IN) and [HuggingFace embedding models](https://huggingface.co/blog/getting-started-with-embeddings) for the SQL agent, and [Chainlit](https://docs.chainlit.io/get-started/overview) for UI integration.


This is currently a PoC, it can be sharp around the edges.

---
