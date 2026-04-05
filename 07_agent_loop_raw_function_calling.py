from dotenv import load_dotenv
import os,json
from openai import OpenAI
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = 'nvidia/nemotron-3-super-120b-a12b:free'
# MODEL = 'openrouter:nvidia/nemotron-nano-12b-v2-vl:free'
# MODEL = 'google_genai:gemini-2.5-flash'

load_dotenv()

# Initialize the client for OpenRouter
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)
# Tools that the agent can use to perform specific tasks. 
# In this example, we have two simple tools: one for looking up product prices and another for applying discounts.
@traceable(run_type='tool')
def get_product_price(product_name: str) -> float:
    """Looks up the price of a product."""
    print(f"Looking up price for {product_name}...")
    mock_prices = {
        "laptop": 999.99,
        "smartphone": 499.00,
        "headphones": 199.00
    }
    return mock_prices.get(product_name.lower(), 0.0)

@traceable(run_type='tool')
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

# Difference 2: Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

# --- Helper: traced Ollama call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.
@traceable(name="Ollama Chat", run_type="llm")
def openai_chat_traced(request):
    return openai_client.chat.completions.create(**request)

# Agent Loop: The agent will use the defined tools to perform a task iteratively 
# until it reaches a solution or hits the maximum number of iterations.
@traceable(name="Agent Loop with Tool Calling")
def run_agent(query: str):

    tool_dict = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
    }

    print(f'Query : {query}')
    print('#' * 50)

    messages = [
    {
        "role": "system",
        "content": """You are a helpful assistant that can look up product prices and apply discounts to calculate final prices.
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
            """
    },
    {
        "role": "user",
        "content": query,
    }
    ]
    # Difference 4: Creating the request payload for the OpenRouter API is more manual compared to using LangChain's agent abstraction. 
    # We need to ensure the messages and tools are formatted correctly according to the API specification.
    request = {
        "model": MODEL,
        "tools": tools_for_llm,
        "messages": messages
    }

    for i in range(1, MAX_ITERATIONS + 1):
        print(f"--- Iteration {i} ---")
        
        # Difference 5: We must manually parse the response from the OpenRouter API to check for tool calls and execute them.
        response = openai_chat_traced(request)
        ai_response = response.choices[0].message

        messages.append(ai_response)

        tool_calls = ai_response.tool_calls
        if not tool_calls:
            print(f"Final Answer: {ai_response.content}")
            return ai_response.content
        
        # Force only first tool call to be executed per iteration to ensure the agent is using tools correctly 
        # and not trying to do multiple steps in one go.
        tool_call = tool_calls[0]
        # Difference 6: The structure of the tool call in the raw OpenRouter response is different from how 
        # LangChain structures it.
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        tool_function = tool_dict.get(tool_name)
        tool_call_id = tool_call.id
        if tool_function:
            # Difference 7: We must manually execute the tool function and then append the result back to the messages 
            # for the next iteration.
            tool_result = tool_dict[tool_name](**tool_args)
            print(f"Tool '{tool_name}' called with arguments {tool_args} returned: {tool_result}")
            messages.append({
                "role": "tool",
                "content": str(tool_result),
                "tool_call_id": tool_call_id
            })
        else:
            print(f"Error: Tool '{tool_name}' not found.")
            raise ValueError("Error: Invalid tool call.")

    print(f'Max Iterations: {MAX_ITERATIONS} reached without a final answer.')

if __name__ == "__main__":
    run_agent("What is the price of a smartphone with silver discount?")


# Results:
# --- Iteration 1 ---
# Looking up price for smartphone...
# Tool 'get_product_price' called with arguments {'product_name': 'smartphone'} returned: 499.0
# --- Iteration 2 ---
# Applying silver tier discount to $499...
# Tool 'apply_discount' called with arguments {'price': 499, 'discount_tier': 'silver'} returned: 449.1
# --- Iteration 3 ---
# Final Answer: 449.1
