## Welcome to SQL Data Agent! 🤖

Hi there! This is a SQL Data Agent to assist you in fetching data and related insights from an attached relational database.
The database available to the agent is the conventional power plants dataset from OPSD available [here](https://data.open-power-system-data.org/conventional_power_plants/).

The technologies used to set this agent up are [LangGraph](https://langchain-ai.github.io/langgraph/tutorials/introduction/), [Google Gemini](https://gemini.google.com/app?hl=en-IN), [HuggingFace embedding models](https://huggingface.co/blog/getting-started-with-embeddings) of the SQL agent and [Chainlit](https://docs.chainlit.io/get-started/overview) for UI integration.
This is currently a PoC! Hence, it maybe sharp around the edges.

## Example prompt ideas

- Hi!
- Can you list all the tables you have access to?
- List down the number of power plants by country and state in Germany.
- List the amount of capacity installed for each type of technology in Germany from conventional_power_plants_DE?
- List different types of technologies used across different power plants and the total capacity for each specific technology?"
- List the the total capacity of all renewable energy sources installed within Germany, France and Poland?"
- List down the total installed capacity (net and gross) for each energy source in Germany."
- List down the total installed capacity (net and gross) for each energy source in Germany."
- Find the plant with the highest efficiency for each technology type in the EU?"
- Track the total number of plants commissioned in the european union in each decade from 1960 to 2020?"
- Track the total number of power plants decommisioned in Germany from 2010 to 2020?"
- Identify power plants located in cities where the number of plants exceeds the average per city?"

## Useful Links 🔗

- **Documentation:** Get started with our comprehensive [Chainlit Documentation](https://docs.chainlit.io) 📚
- **Discord Community:** Join our friendly [Chainlit Discord](https://discord.gg/k73SQ3FyUh) to ask questions, share your projects, and connect with other developers! 💬

We can't wait to see what you create with Chainlit! Happy coding! 💻😊

## Welcome screen

To modify the welcome screen, edit the `chainlit.md` file at the root of your project. If you do not want a welcome screen, just leave this file empty.
