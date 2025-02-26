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

### Data sources
The databases used during development are the conventional power plants and time-series datasets from OPSD available [here](https://data.open-power-system-data.org/).

```
# Time-series dataset for df_forecast_agent.py
wget https://data.open-power-system-data.org/time_series/2020-10-06/time_series.sqlite
wget https://data.open-power-system-data.org/time_series/2020-10-06/datapackage.json
```

[Time-series database](https://data.open-power-system-data.org/time_series) contains different kinds of timeseries data relevant for power system modelling, namely electricity prices, electricity consumption (load) as well as wind and solar power generation and capacities. The data is aggregated either by country, control area or bidding zone. Geographical coverage includes the EU and some neighbouring countries.

Paths to the dataset used can be modified here for `df_forecast_agent.py`:
- https://github.com/richhiey/genai_db_poc/blob/main/df_forecast_agent.py#L29

```
# Conventional power plants dataset used with db_agent_2.py
wget https://data.open-power-system-data.org/conventional_power_plants/2020-10-01/conventional_power_plants.sqlite
wget https://data.open-power-system-data.org/conventional_power_plants/2020-10-01/datapackage.json
```

[Conventional Power Plants database](https://data.open-power-system-data.org/conventional_power_plants) contains data on conventional power plants for Germany as well as other selected European countries. The data includes individual power plants with their technical characteristics. These include installed capacity, main energy source, type of technology, CHP capability, and geographical information. The geographical scope is primarily on Germany and its neighboring countries. 

Paths to the dataset used can be modified here for `db_agent_2.py`:
- https://github.com/richhiey/genai_db_poc/blob/main/db_agent_2.py#L38

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
