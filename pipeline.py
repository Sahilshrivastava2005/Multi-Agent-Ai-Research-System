from langgraph.graph import StateGraph, END
from agents import (
    AgentState, 
    get_search_agent_response, 
    get_reader_agent_response, 
    writer_chain, 
    critic_chain
)
from rich import print

# --- Node Definitions ---

def search_node(state: AgentState):
    print("\n" + " ="*50)
    print("Step 1: Search Agent is working...")
    print("="*50)
    topic = state["topic"]
    results = get_search_agent_response(topic)
    return {"search_results": results}

def reader_node(state: AgentState):
    print("\n" + " ="*50)
    print("Step 2: Reader Agent is scraping top resources...")
    print("="*50)
    topic = state["topic"]
    search_results = state["search_results"]
    content = get_reader_agent_response(topic, search_results)
    return {"scraped_content": content}

def writer_node(state: AgentState):
    print("\n" + " ="*50)
    print("Step 3: Writer is drafting the report...")
    print("="*50)
    topic = state["topic"]
    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )
    report = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })
    return {"report": report}

def critic_node(state: AgentState):
    print("\n" + " ="*50)
    print("Step 4: Critic is reviewing the report...")
    print("="*50)
    report = state["report"]
    feedback = critic_chain.invoke({"report": report})
    return {"feedback": feedback}

# --- Graph Construction ---

def build_research_graph():
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("search", search_node)
    workflow.add_node("reader", reader_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)

    # Set edges
    workflow.set_entry_point("search")
    workflow.add_edge("search", "reader")
    workflow.add_edge("reader", "writer")
    workflow.add_edge("writer", "critic")
    workflow.add_edge("critic", END)

    return workflow.compile()

# --- Execution Function ---

def run_research_pipeline(topic: str):
    graph = build_research_graph()
    initial_state = {
        "topic": topic,
        "search_results": "",
        "scraped_content": "",
        "report": "",
        "feedback": "",
        "messages": []
    }
    
    # We use stream to observe changes if needed, but for simplicity we'll just run it
    final_state = graph.invoke(initial_state)
    return final_state

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    user_topic = input("\nEnter a research topic: ")
    if user_topic:
        result = run_research_pipeline(user_topic)
        print("\n" + " ="*50)
        print("FINAL REPORT")
        print("="*50)
        print(result["report"])
        print("\n" + " ="*50)
        print("CRITIC FEEDBACK")
        print("="*50)
        print(result["feedback"])

