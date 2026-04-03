from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch

load_dotenv()

llm = ChatOpenRouter(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        temperature=0,
        max_retries=2
    )

tools = [TavilySearch()]
agent = create_agent(model=llm,tools=tools)

def main():
    result = agent.invoke({"messages":HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the Miami area on site:://linkedin.com and list their details?")})
    print(result)

if __name__ == "__main__":
    main()
