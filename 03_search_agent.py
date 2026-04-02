from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain.tools import tool
import os

load_dotenv()

@tool
def search(query:str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f'Searching for {query}')
    return ("Weather in Kochi is humid")    

llm = ChatOpenRouter(
        model="nvidia/nemotron-nano-12b-v2-vl:free",
        temperature=0,
        max_retries=2
    )

tools = [search]
agent = create_agent(model=llm,tools=tools)

def main():
    result = agent.invoke({"messages":HumanMessage(content="What is the weather in Kochi?")})
    print(result)

if __name__ == "__main__":
    main()
