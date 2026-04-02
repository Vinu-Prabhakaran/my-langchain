from dotenv import load_dotenv
import os

load_dotenv()

def main():
    print("Hello from langchain-course!")
    print(f'{os.environ.get('OPENROUTER_API_KEY') = }')


if __name__ == "__main__":
    main()
