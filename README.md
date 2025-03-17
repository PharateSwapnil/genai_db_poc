## Data Agent

PoC to create agents that can retrieve data from a source and generate insights based on it.
At the moment, it works by generating SQL queries to retrieve relevant data, executing them against a DB that it has access to, and generating responses based on the retrieved data.

### Setup

Virtual environment and installation of requirements can be done with the commands given below:
```
conda create -n genai_poc python=3.11
conda activate genai_poc
pip install -r requirements.txt
```
If you use `venv`, please change the commands related to conda accordingly.

#### Environment variables
Create a `.env` file with your personal API Keys for [HuggingFace Hub](https://huggingface.co/docs/hub/security-tokens) and [Gemini API](https://ai.google.dev/gemini-api/docs) and place it in the root folder of this repository.
The `.env` file will look something like this:
```
HF_TOKEN = "hf_xxxxxx"
HUGGINGFACEHUB_API_TOKEN = "hf_xxxxxx"
GOOGLE_API_KEY = "xx_xxxxxx_xxxxxx"
```
When using a different model provider (like OpenAI, Groq, Meta Llama, etc.), include the API Keys here and they will be loaded at runtime.
The model dictionary used to load the ChatModel can be changed [here](https://github.com/richhiey/genai_db_poc/blob/main/db_agent_2.py#L54). 
### Data sources

Cleaned up version of the datasets to be used with this PoC can be found here --> [Download](https://godigitaltcllp-my.sharepoint.com/:f:/g/personal/richhiey_thomas_godigitaltc_com/ElXtx2JOzX5Npovd87h2tisBBCMOsXa2LDYncqdR2PHkSQ?e=3xwW6T)

Paths to the dataset used can be modified here for `df_forecast_agent.py`:
- https://github.com/richhiey/genai_db_poc/blob/main/df_forecast_agent.py#L29

Paths to the dataset used can be modified here for `db_agent_2.py`:
- https://github.com/richhiey/genai_db_poc/blob/main/db_agent_2.py#L38

The databases used during development are the conventional power plants and time-series datasets from OPSD available [here](https://data.open-power-system-data.org/).

[Time-series database](https://data.open-power-system-data.org/time_series) contains different kinds of timeseries data relevant for power system modelling, namely electricity prices, electricity consumption (load) as well as wind and solar power generation and capacities. The data is aggregated either by country, control area or bidding zone. Geographical coverage includes the EU and some neighbouring countries.

[Conventional Power Plants database](https://data.open-power-system-data.org/conventional_power_plants) contains data on conventional power plants for Germany as well as other selected European countries. The data includes individual power plants with their technical characteristics. These include installed capacity, main energy source, type of technology, CHP capability, and geographical information. The geographical scope is primarily on Germany and its neighboring countries.

---

### Spin up the interface

In order to run the chainlit UI for this PoC, execute the command below:
```
chainlit run app.py
```
The first time it executes, might take around 30 seconds to collect the embedding model and index the metadata from `datapackage.json`.

#### Individual python scripts for debugging the agent files
SQL DB Agent designed to fetch relevant data given a prompt query in natural language
```
python3 db_agent_2.py
```

Data Agent designed to fetch relevant historical data given a prompt query and forecast future values using Amazon Chronos forecasting tool.
```
python3 df_forecast_agent.py
```
