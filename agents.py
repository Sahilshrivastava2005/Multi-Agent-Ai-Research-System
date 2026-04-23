import os
from typing import Annotated, List, TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url

load_dotenv()

# --- LLM Setup ---
# Using gemini-3-flash-preview for state-of-the-art research performance
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)



# --- State Definition ---
class AgentState(TypedDict):
    topic: str
    search_results: str
    scraped_content: str
    report: str
    feedback: str
    messages: List[BaseMessage]

# --- Agent Helpers ---

def get_search_agent_response(topic: str):
    """Call the search agent logic directly."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful research assistant. Use the search tool to find information."),
        ("human", "Find recent, reliable and detailed information about: {topic}")
    ])
    # For simplicity in this pipeline, we call the tool and then the LLM
    results = web_search.invoke(topic)
    return results

def get_reader_agent_response(topic: str, search_results: str):
    """Call the reader agent logic directly."""
    # In a more complex setup, we'd let the LLM choose the URL.
    # Here we'll let the LLM extract the best URL from search results and then we scrape it.
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a specialized content extractor. Given search results, identify the single most relevant and authoritative URL to scrape for deep information."),
        ("human", "Search Results:\n{results}\n\nReturn ONLY the URL, nothing else.")
    ])
    chain = extraction_prompt | llm | StrOutputParser()
    url = chain.invoke({"results": search_results}).strip()
    
    if url.startswith("http"):
        content = scrape_url.invoke(url)
        return content
    return "No valid URL found to scrape."

# --- Writing & Critique Chains ---

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports in Markdown."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- # {topic}
- ## Introduction
- ## Key Findings (minimum 3 well-explained points)
- ## Analysis & Implications
- ## Conclusion
- ## Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

# Evaluation
**Score:** X/10

## Strengths
- ...

## Areas to Improve
- ...

## Verdict
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()