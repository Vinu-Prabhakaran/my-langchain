from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = 'openrouter:nvidia/nemotron-3-super-120b-a12b:free'
# MODEL = 'openrouter:nvidia/nemotron-nano-12b-v2-vl:free'
# MODEL = 'google_genai:gemini-2.5-flash'

load_dotenv()

# Tools that the agent can use to perform specific tasks. 
# In this example, we have two simple tools: one for looking up product prices and another for applying discounts.
@tool
def get_product_price(product_name: str) -> float:
    """Looks up the price of a product."""
    print(f"Looking up price for {product_name}...")
    mock_prices = {
        "laptop": 999.99,
        "smartphone": 499.00,
        "headphones": 199.00
    }
    return mock_prices.get(product_name.lower(), 0.0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Applies a discount to the price based on the discount tier and returns the discounted price."""
    print(f"Applying {discount_tier} tier discount to ${price}...")
    discount_mapping = {
        "bronze": 5,
        "silver": 10,
        "gold": 15
    }
    discount_percentage = discount_mapping.get(discount_tier.lower(), 0)
    return round(price * (1 - discount_percentage / 100), 2)

# Agent Loop: The agent will use the defined tools to perform a task iteratively 
# until it reaches a solution or hits the maximum number of iterations.
@traceable(name="Agent Loop with Tool Calling")
def run_agent(query: str):
    tools = [get_product_price, apply_discount]
    tool_dict = {t.name:t for t in tools}
    llm = init_chat_model(MODEL,temperature=0)
    llm_with_tools = llm.bind_tools(tools=tools)

    print(f'Query : {query}')
    print('#' * 50)

    messages = [SystemMessage(
        content="""You are a helpful assistant that can look up product prices and apply discounts to calculate final prices.
        Use the following tools to perform your tasks:
        1. get_product_price(product_name: str) -> float: Looks up the price of a product.
        2. apply_discount(price: float, discount_tier: str) -> float: Applies a discount to the price based on the discount tier and returns the discounted price.
        When you need to use a tool, call it with the appropriate arguments. If you have the final answer, return it directly without calling any more tools.
        --- STRICT INSTRUCTIONS ---
        1. Always use the tools for their intended purpose. Do not try to perform calculations or lookups without using the tools.
        2. If you need to get the price of a product, use get_product_price with the product name as the argument.
        3. If you need to apply a discount, use apply_discount only once you have the price and discount tier as arguments.
        4. Do not return the final answer until you have all the necessary information from the tools.
        5. Do not assume any information about product names, prices or discounts. Always use the tools to get accurate information.
        """)]

    messages.append(HumanMessage(content=query))

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"--- Iteration {i} ---")
        ai_response = llm_with_tools.invoke(messages)
        messages.append(ai_response)

        tool_calls = ai_response.tool_calls
        if not tool_calls:
            print(f"Final Answer: {ai_response.content}")
            return ai_response.content
        
        # Force only first tool call to be executed per iteration to ensure the agent is using tools correctly 
        # and not trying to do multiple steps in one go.
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")
        tool_function = tool_dict.get(tool_name)
        if tool_function:
            tool_result = tool_function.invoke(tool_args)
            print(f"Tool '{tool_name}' called with arguments {tool_args} returned: {tool_result}")
            messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call_id))
        else:
            print(f"Error: Tool '{tool_name}' not found.")
            raise ValueError("Error: Invalid tool call.")

    print(f'Max Iterations: {MAX_ITERATIONS} reached without a final answer.')

if __name__ == "__main__":
    run_agent("What is the price of a smartphone with silver discount?")


# Results:
# vinu at Vinus-MacBook-Pro in langchain-course on project/my-langchain [!?]
# $ /Users/vinu/Documents/Work/MyProjects/langchain-course/.venv/bin/python /Users/vinu/Documents/Work/MyProjects/langchain-course/06_agent_loop_lang
# chain_tool_calling.py
# Query : What is the price of a laptop with gold discount?
# ##################################################
# --- Iteration 1 ---
# Looking up price for laptop...
# Tool 'get_product_price' called with arguments {'product_name': 'laptop'} returned: 999.99
# --- Iteration 2 ---
# Applying gold% discount to $999.99...
# Tool 'apply_discount' called with arguments {'price': 999.99, 'discount_tier': 'gold'} returned: 849.99
# --- Iteration 3 ---
# Final Answer: [{'type': 'text', 'text': 'The price of a laptop with gold discount is 849.99.', 'extras': {'signature': 'CrQBAb4+9vu8cFv0Auh9Y0bdu5wpgd9yoqoIinHkO+sRFNkaJqmYAeSpedlWfxHzWHRgKnm4ZkMMzO3DLLgnkBazLCQi2uzNdzXwkcWi1EkCJ9Q6WbVYF7zsTMOYFyOuXRzLfoDvyXanX5RB7Omo5b4N/XtotlUAonydhpqUReIzRuE6y2wH3E7/GoiExrBele8M2xZXR6u8Kbdq/XR7GFZGC0RpSLoREKPirBQ0JwSno+2odU3G'}}]