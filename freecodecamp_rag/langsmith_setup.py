from langsmith import traceable
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable
from langsmith.run_trees import RunTree
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatMessagePromptTemplate
load_dotenv()
@traceable
def process_query(query):
    return query.lower()
print(process_query("Manish"))

@traceable(name="basic_config")
def basic_chain():
    """basic langsmith functions"""
    llm=ChatGoogleGenerativeAI(
        model="gemini-3.6-flash"
    )
    prompt=ChatPromptTemplate.from_template(" answer this {topic} one sentence")
    
    chain=prompt | llm | StrOutputParser()
    
    print("langsmith tracing started..")
    result=chain.invoke({"topic":"Rag"})
    print(f"Result: {result}")
    print("\nCheck LangSmith dashboard for trace details.")
@traceable(name="named_runs_demo", tags=["production", "summarization"])
def demo_named_runs():
    """Name your runs for easier identification."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template("Summarize: {text}")

    chain = prompt | llm | StrOutputParser()

    print("\nNamed Runs Demo:\n")

    result = chain.invoke(
        {"text": "LangSmith provides observability for LLM applications."}
    )

    print(f"Result: {result}")
    print("Run tagged with 'production', 'summarization'")


@traceable(name="trace_with_metadata_demo", tags=["metadata", "filtering"])
def demo_trace_with_metadata(user_id: str, request_type: str):
    """Add metadata to traces for filtering."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Metadata is automatically captured
    result = llm.invoke(f"Hello from user {user_id}")

    return result.content


if __name__ == "__main__":
    basic_chain()
    demo_named_runs()
    demo_trace_with_metadata(user_id="user_123", request_type="greeting")

