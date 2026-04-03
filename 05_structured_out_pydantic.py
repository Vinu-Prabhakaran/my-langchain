from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field

load_dotenv()

llm = ChatOpenRouter(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        temperature=0,
        max_retries=2
    )

tools = [TavilySearch()]
agent = create_agent(model=llm,tools=tools)

class AgentResponse(BaseModel):
    """Agent response containing structured information with answer and sources."""
    answer: str = Field(description="The answer to the query")
    sources: list[str] = Field(description="List of source urls used to generate the answer")

agent = create_agent(model=llm,tools=tools,
    response_format=AgentResponse  
)

def main():
    result = agent.invoke({"messages":HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the Miami area on site:://linkedin.com and list their details?")})
    print(result["structured_response"])

if __name__ == "__main__":
    main()
