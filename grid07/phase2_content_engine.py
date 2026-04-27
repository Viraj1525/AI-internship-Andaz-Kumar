"""
Phase 2: Autonomous Content Engine (LangGraph)

This module implements an autonomous content generation system using LangGraph.
Each bot persona searches for relevant news and generates opinionated posts.
"""

import os
import json
from typing import TypedDict, Annotated
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


# =============================================================================
# MOCK TOOL: Web Search Simulation
# =============================================================================

@tool
def mock_searxng_search(query: str) -> str:
    """
    Mock web search returning hardcoded news based on keywords.

    Args:
        query: The search query string.

    Returns:
        Hardcoded news results based on keyword matching.
    """
    query_lower = query.lower()

    if "crypto" in query_lower or "bitcoin" in query_lower:
        return "Bitcoin hits new all-time high amid regulatory ETF approvals. Ethereum staking yields reach 8%."
    elif "ai" in query_lower or "openai" in query_lower or "model" in query_lower:
        return "OpenAI releases GPT-5 with reasoning capabilities. EU AI Act enforcement begins Q3."
    elif "market" in query_lower or "stock" in query_lower or "fed" in query_lower:
        return "Fed signals two rate cuts in 2025. S&P 500 hits record high. Tech earnings beat expectations."
    elif "climate" in query_lower or "environment" in query_lower or "nature" in query_lower:
        return "UN report: 2024 hottest year on record. Corporate carbon pledges called 'greenwashing' by auditors."
    elif "space" in query_lower or "elon" in query_lower or "tesla" in query_lower:
        return "SpaceX Starship completes orbital test. Tesla FSD miles double quarter-over-quarter."
    else:
        return "Tech sector leads market gains. AI investment reaches $500B globally in 2025."


# =============================================================================
# BOT PERSONAS
# =============================================================================

BOT_PERSONAS = {
    "bot_a": "I believe AI and crypto will solve all human problems. I am highly optimistic about technology, Elon Musk, and space exploration. I dismiss regulatory concerns.",
    "bot_b": "I believe late-stage capitalism and tech monopolies are destroying society. I am highly critical of AI, social media, and billionaires. I value privacy and nature.",
    "bot_c": "I strictly care about markets, interest rates, trading algorithms, and making money. I speak in finance jargon and view everything through the lens of ROI.",
}


# =============================================================================
# LANGGRAPH STATE
# =============================================================================

class GraphState(TypedDict):
    """
    State definition for the content generation graph.

    Tracks the bot identity, search results, and generated post content.
    """
    bot_id: str
    persona: str
    search_query: str
    search_results: str
    post_content: str
    topic: str


# =============================================================================
# LLM INITIALIZATION
# =============================================================================

def get_llm():
    """
    Initialize LLM based on environment configuration.

    Returns:
        Configured LangChain chat model.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        model = os.getenv("LLM_MODEL", "llama3-8b-8192")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        return ChatGroq(api_key=api_key, model=model)
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        return ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo")
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


# =============================================================================
# GRAPH NODES
# =============================================================================

def decide_search(state: GraphState) -> dict:
    """
    Node 1: Decide what topic to search for based on bot persona.

    The LLM analyzes the bot's persona and decides what topic
    the bot would want to post about, then formats a search query.

    Args:
        state: Current graph state.

    Returns:
        Updated state with search_query and topic.
    """
    persona = state["persona"]
    bot_id = state["bot_id"]

    # Prompt to decide search topic and query
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a social media bot deciding what to post about.
Based on your persona, decide what topic you want to post about today.
Then create a search query to find relevant news.

Persona: {persona}

Respond in JSON format:
{{"topic": "your chosen topic", "search_query": "your search query"}}"""),
    ])

    llm = get_llm()
    chain = prompt | llm

    try:
        response = chain.invoke({"persona": persona})
        # Parse the response - extract JSON from the response
        response_text = response.content.strip()

        # Try to extract JSON if it's wrapped in markdown
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)

        return {
            "search_query": result.get("search_query", "technology news"),
            "topic": result.get("topic", "technology")
        }
    except Exception as e:
        # Fallback based on bot_id
        fallback_queries = {
            "bot_a": ("AI technology breakthrough", "AI technology"),
            "bot_b": ("corporate environmental impact", "environment"),
            "bot_c": ("market analysis stocks", "market"),
        }
        query, topic = fallback_queries.get(bot_id, ("news", "general"))
        return {"search_query": query, "topic": topic}


def web_search(state: GraphState) -> dict:
    """
    Node 2: Execute web search with the decided query.

    Calls the mock_searxng_search tool with the search_query from state.

    Args:
        state: Current graph state.

    Returns:
        Updated state with search_results.
    """
    search_query = state["search_query"]

    try:
        results = mock_searxng_search.invoke(search_query)
        return {"search_results": results}
    except Exception as e:
        return {"search_results": "No search results available."}


def draft_post(state: GraphState) -> dict:
    """
    Node 3: Draft the final post using persona and search results.

    The LLM generates an opinionated post under 280 characters
    using the bot's persona and the search results as context.

    Args:
        state: Current graph state.

    Returns:
        Updated state with post_content in JSON format.
    """
    persona = state["persona"]
    bot_id = state["bot_id"]
    search_results = state["search_results"]
    topic = state["topic"]

    # Prompt to draft the post
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a social media bot generating a post.

PERSONA: {persona}
SEARCH RESULTS: {search_results}
TOPIC: {topic}

Generate an opinionated post under 280 characters that:
1. Reflects your persona's viewpoint
2. References the search results
3. Sounds like a real social media post

Return ONLY valid JSON:
{{"bot_id": "{bot_id}", "topic": "{topic}", "post_content": "your post here"}}"""),
    ])

    llm = get_llm()
    chain = prompt | llm

    try:
        response = chain.invoke({
            "persona": persona,
            "search_results": search_results,
            "topic": topic,
            "bot_id": bot_id
        })

        response_text = response.content.strip()

        # Extract JSON from response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        result = json.loads(response_text)

        return {
            "post_content": result.get("post_content", ""),
        }
    except Exception as e:
        return {"post_content": f"[Error generating post: {str(e)}]"}


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_graph() -> StateGraph:
    """
    Build the LangGraph state machine.

    Graph structure:
        [decide_search] → [web_search] → [draft_post] → END

    Returns:
        Compiled StateGraph ready for execution.
    """
    # Initialize the graph with state definition
    graph_builder = StateGraph(GraphState)

    # Add nodes
    graph_builder.add_node("decide_search", decide_search)
    graph_builder.add_node("web_search", web_search)
    graph_builder.add_node("draft_post", draft_post)

    # Add edges (define flow)
    graph_builder.add_edge("decide_search", "web_search")
    graph_builder.add_edge("web_search", "draft_post")
    graph_builder.add_edge("draft_post", END)

    # Set entry point
    graph_builder.set_entry_point("decide_search")

    return graph_builder.compile()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def run_content_engine() -> list:
    """
    Run the content generation engine for all 3 bots.

    Returns:
        List of JSON outputs from each bot.
    """
    print("=== PHASE 2: AUTONOMOUS CONTENT ENGINE ===\n")
    print("Graph structure: [decide_search] -> [web_search] -> [draft_post] -> END\n")
    print("-" * 70)

    # Build and compile the graph
    graph = build_graph()

    results = []

    for bot_id, persona in BOT_PERSONAS.items():
        print(f"\nGenerating content for {bot_id}...")
        print(f"  Persona: {persona[:60]}...")

        # Initial state
        initial_state = {
            "bot_id": bot_id,
            "persona": persona,
            "search_query": "",
            "search_results": "",
            "post_content": "",
            "topic": ""
        }

        try:
            # Run the graph
            final_state = graph.invoke(initial_state)

            # Build result JSON
            result = {
                "bot_id": final_state.get("bot_id", bot_id),
                "topic": final_state.get("topic", "unknown"),
                "post_content": final_state.get("post_content", "")
            }

            results.append(result)

            print(f"  Topic: {final_state.get('topic', 'N/A')}")
            print(f"  Post: {final_state.get('post_content', 'N/A')}")

        except Exception as e:
            print(f"  Error: {str(e)}")
            results.append({
                "bot_id": bot_id,
                "topic": "error",
                "post_content": f"[Error: {str(e)}]"
            })

    print("\n" + "-" * 70)
    print("\n=== PHASE 2 COMPLETE ===\n")

    return results


if __name__ == "__main__":
    results = run_content_engine()

    print("\n=== FINAL JSON OUTPUTS ===\n")
    for result in results:
        print(json.dumps(result, indent=2))
