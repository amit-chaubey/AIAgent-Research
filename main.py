from dotenv import load_dotenv
import os
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic # Uncomment if using Anthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# Load environment variables from .env file
from langchain_core.output_parsers import PydanticOutputParser


load_dotenv()

class ResponseModel(BaseModel):
    content: str
    summary: str
    source: list[str]
    tools: list[str]


llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=1000,
    streaming=True,
)

parser = PydanticOutputParser(pydantic_object=ResponseModel)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("user", "Answer the following question: {question}"),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# from langchain_huggingface import ChatHuggingFace # Uncomment if using Hugging Face

# llm = ChatHuggingFace(
#     model="HuggingFaceH4/zephyr-7b-beta",
#     temperature=0.7,
#     max_tokens=1000,
#     streaming=True,
# ) # Uncomment if using Hugging Face


# llm = ChatAnthropic(
#     model="claude-3-5-sonnet-20241022",
#     temperature=0.7,
#     max_tokens=1000,
#     streaming=True,
# ) # Uncomment if using Anthropic

reponse = llm.invoke(
    "What is the capital of France?",
)
print(reponse.content)