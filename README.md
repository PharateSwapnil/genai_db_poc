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

In order to run the chainlit UI for this PoC, execute the command below:
```
chainlit run app.py
```

### Data sources
The databases used during development are the conventional power plants and time-series datasets from OPSD available [here](https://data.open-power-system-data.org/).

This [Time-series database](https://data.open-power-system-data.org/time_series) contains different kinds of timeseries data relevant for power system modelling, namely electricity prices, electricity consumption (load) as well as wind and solar power generation and capacities. The data is aggregated either by country, control area or bidding zone. Geographical coverage includes the EU and some neighbouring countries. The [Conventional Power Plants database](https://data.open-power-system-data.org/conventional_power_plants) contains data on conventional power plants for Germany as well as other selected European countries. The data includes individual power plants with their technical characteristics. These include installed capacity, main energy source, type of technology, CHP capability, and geographical information. The geographical scope is primarily on Germany and its neighboring countries. 

Paths to the datasets used can be modified here:
- https://github.com/richhiey/genai_db_poc/blob/main/df_forecast_agent.py#L29
- https://github.com/richhiey/genai_db_poc/blob/main/db_agent_2.py#L38

---

SQL DB Agent designed to fetch relevant data given a prompt query in natural language
```
python3 db_agent_2.py
```

Data Agent designed to fetch relevant historical data given a prompt query and forecast future values using Amazon Chronos forecasting tool.
```
python3 df_forecast_agent.py
```
