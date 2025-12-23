"""
Travel Agent Example: LlamaIndex GradientAI with Bucket List Knowledge Base
============================================================================

This example demonstrates a travel-focused AI agent that uses:
- Function/tool calling with LlamaIndex
- Bucket List Travel Knowledge Base (bucketlisttravels.com)
- Multi-step agent reasoning for trip planning

Knowledge Base Source: https://www.bucketlisttravels.com/round-up/100-bucket-list-destinations

Environment Variables Required:
- GRADIENT_MODEL_ACCESS_KEY: Your model access key for inference
- DIGITALOCEAN_API_TOKEN: Your DigitalOcean API token
- DIGITALOCEAN_KB_ID: Your Knowledge Base ID (travel KB)
"""

import os
from typing import List, Optional
from dotenv import load_dotenv

from gradient_adk import entrypoint, trace_llm, trace_retriever, trace_tool
from gradient import Gradient
from llama_index.llms.digitalocean.gradientai import GradientAI
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import ChatMessage

load_dotenv()

# Initialize clients
gradient_client = Gradient(access_token=os.environ.get("DIGITALOCEAN_API_TOKEN"))

llm = GradientAI(
    model="openai-gpt-oss-120b",
    model_access_key=os.environ.get("GRADIENT_MODEL_ACCESS_KEY"),
)


# ============================================================================
# Travel-Specific Tools for Bucket List Destinations
# ============================================================================

def search_bucket_list_destinations(query: str, num_results: int = 5) -> str:
    """
    Search the bucket list travel knowledge base for destinations and travel information.
    Use this to find bucket list destinations, attractions, and travel experiences.
    
    Args:
        query: Search query about destinations, places, or travel experiences
        num_results: Number of results to return (default: 5)
    
    Returns:
        Information about matching bucket list destinations
    """
    kb_id = os.environ.get("DIGITALOCEAN_KB_ID")
    if not kb_id:
        return "Error: Knowledge Base not configured"
    
    response = gradient_client.retrieve.documents(
        knowledge_base_id=kb_id,
        num_results=num_results,
        query=query,
    )
    
    if response and response.results:
        results = []
        for i, result in enumerate(response.results, 1):
            if hasattr(result, 'text_content'):
                results.append(f"[Result {i}]: {result.text_content}")
        return "\n\n".join(results) if results else "No destinations found."
    
    return "No results found in travel knowledge base."


def find_destinations_by_region(region: str) -> str:
    """
    Find bucket list destinations in a specific region or continent.
    
    Args:
        region: The region to search (e.g., 'Europe', 'Asia', 'Africa', 
                'South America', 'North America', 'Australia', 'Caribbean')
    
    Returns:
        List of bucket list destinations in that region
    """
    kb_id = os.environ.get("DIGITALOCEAN_KB_ID")
    if not kb_id:
        return "Error: Knowledge Base not configured"
    
    query = f"bucket list destinations in {region} travel places to visit"
    
    response = gradient_client.retrieve.documents(
        knowledge_base_id=kb_id,
        num_results=8,
        query=query,
    )
    
    if response and response.results:
        results = []
        for result in response.results:
            if hasattr(result, 'text_content'):
                results.append(result.text_content)
        return f"Bucket list destinations in {region}:\n\n" + "\n\n---\n\n".join(results)
    
    return f"No destinations found for {region}."


def find_destinations_by_experience(experience_type: str) -> str:
    """
    Find bucket list destinations based on type of experience or activity.
    
    Args:
        experience_type: Type of experience (e.g., 'beaches', 'culture', 'history',
                        'adventure', 'nature', 'food', 'romantic', 'family-friendly',
                        'wildlife', 'mountains', 'islands', 'cities')
    
    Returns:
        Destinations matching the experience type
    """
    kb_id = os.environ.get("DIGITALOCEAN_KB_ID")
    if not kb_id:
        return "Error: Knowledge Base not configured"
    
    query = f"bucket list {experience_type} destinations travel experiences"
    
    response = gradient_client.retrieve.documents(
        knowledge_base_id=kb_id,
        num_results=6,
        query=query,
    )
    
    if response and response.results:
        results = []
        for result in response.results:
            if hasattr(result, 'text_content'):
                results.append(result.text_content)
        return f"Best {experience_type} destinations:\n\n" + "\n\n---\n\n".join(results)
    
    return f"No destinations found for {experience_type} experiences."


def get_destination_details(destination: str) -> str:
    """
    Get detailed information about a specific bucket list destination.
    
    Args:
        destination: Name of the destination (e.g., 'London', 'Paris', 'Tokyo', 
                    'Machu Picchu', 'Grand Canyon', 'Santorini')
    
    Returns:
        Detailed information about the destination including attractions,
        best time to visit, and why it's a bucket list destination
    """
    kb_id = os.environ.get("DIGITALOCEAN_KB_ID")
    if not kb_id:
        return "Error: Knowledge Base not configured"
    
    query = f"{destination} bucket list destination guide attractions things to do"
    
    response = gradient_client.retrieve.documents(
        knowledge_base_id=kb_id,
        num_results=3,
        query=query,
    )
    
    if response and response.results:
        results = []
        for result in response.results:
            if hasattr(result, 'text_content'):
                results.append(result.text_content)
        return f"Details about {destination}:\n\n" + "\n\n".join(results)
    
    return f"No detailed information found for {destination}."


def compare_destinations(destination1: str, destination2: str) -> str:
    """
    Compare two bucket list destinations to help with travel decision making.
    
    Args:
        destination1: First destination to compare
        destination2: Second destination to compare
    
    Returns:
        Information about both destinations for comparison
    """
    kb_id = os.environ.get("DIGITALOCEAN_KB_ID")
    if not kb_id:
        return "Error: Knowledge Base not configured"
    
    # Get info for first destination
    response1 = gradient_client.retrieve.documents(
        knowledge_base_id=kb_id,
        num_results=2,
        query=f"{destination1} bucket list destination attractions",
    )
    
    # Get info for second destination
    response2 = gradient_client.retrieve.documents(
        knowledge_base_id=kb_id,
        num_results=2,
        query=f"{destination2} bucket list destination attractions",
    )
    
    result = f"=== {destination1.upper()} ===\n"
    if response1 and response1.results:
        for r in response1.results:
            if hasattr(r, 'text_content'):
                result += r.text_content + "\n"
    else:
        result += "No information found.\n"
    
    result += f"\n=== {destination2.upper()} ===\n"
    if response2 and response2.results:
        for r in response2.results:
            if hasattr(r, 'text_content'):
                result += r.text_content + "\n"
    else:
        result += "No information found.\n"
    
    return result


def find_bucket_list_experiences(experience: str) -> str:
    """
    Find specific bucket list experiences and attractions (not just destinations).
    
    Args:
        experience: The experience to search for (e.g., 'Northern Lights', 
                   'Great Wall of China', 'safari', 'hot air balloon',
                   'Niagara Falls', 'Taj Mahal', 'Machu Picchu')
    
    Returns:
        Information about the bucket list experience
    """
    kb_id = os.environ.get("DIGITALOCEAN_KB_ID")
    if not kb_id:
        return "Error: Knowledge Base not configured"
    
    query = f"{experience} bucket list experience must see attraction"
    
    response = gradient_client.retrieve.documents(
        knowledge_base_id=kb_id,
        num_results=4,
        query=query,
    )
    
    if response and response.results:
        results = []
        for result in response.results:
            if hasattr(result, 'text_content'):
                results.append(result.text_content)
        return f"Bucket list experience - {experience}:\n\n" + "\n\n".join(results)
    
    return f"No information found for '{experience}'."


def check_destination_in_bucket_list(destination: str) -> str:
    """
    Check if a specific place is in the bucket list destinations database.
    
    Args:
        destination: The destination to check (e.g., 'London', 'Paris', 'Tokyo')
    
    Returns:
        Whether the destination is in the bucket list and key highlights
    """
    kb_id = os.environ.get("DIGITALOCEAN_KB_ID")
    if not kb_id:
        return "Error: Knowledge Base not configured"
    
    query = f"{destination} bucket list destination"
    
    response = gradient_client.retrieve.documents(
        knowledge_base_id=kb_id,
        num_results=3,
        query=query,
    )
    
    if response and response.results:
        for result in response.results:
            if hasattr(result, 'text_content'):
                content = result.text_content.lower()
                if destination.lower() in content:
                    return f"Yes, {destination} is a bucket list destination!\n\n{result.text_content}"
        return f"{destination} was not found as a featured bucket list destination, but here's related info:\n\n{response.results[0].text_content if response.results else 'No info available.'}"
    
    return f"Could not find information about {destination}."


# ============================================================================
# Create LlamaIndex FunctionTools
# ============================================================================

search_destinations_tool = FunctionTool.from_defaults(
    fn=search_bucket_list_destinations,
    name="search_bucket_list_destinations",
    description="Search the bucket list travel knowledge base for any travel-related query. Use this as your primary search tool for destinations, attractions, and travel information."
)

region_search_tool = FunctionTool.from_defaults(
    fn=find_destinations_by_region,
    name="find_destinations_by_region",
    description="Find bucket list destinations in a specific region or continent (Europe, Asia, Africa, South America, North America, Australia, Caribbean, Middle East)."
)

experience_search_tool = FunctionTool.from_defaults(
    fn=find_destinations_by_experience,
    name="find_destinations_by_experience",
    description="Find destinations based on experience type: beaches, culture, history, adventure, nature, food, romantic, family-friendly, wildlife, mountains, islands, or cities."
)

destination_details_tool = FunctionTool.from_defaults(
    fn=get_destination_details,
    name="get_destination_details",
    description="Get detailed information about a specific bucket list destination including attractions, highlights, and why it's worth visiting."
)

compare_tool = FunctionTool.from_defaults(
    fn=compare_destinations,
    name="compare_destinations",
    description="Compare two bucket list destinations side by side to help decide between them."
)

experience_tool = FunctionTool.from_defaults(
    fn=find_bucket_list_experiences,
    name="find_bucket_list_experiences",
    description="Find information about specific bucket list experiences like Northern Lights, Great Wall of China, safari, Niagara Falls, Taj Mahal, hot air balloon rides, etc."
)

check_destination_tool = FunctionTool.from_defaults(
    fn=check_destination_in_bucket_list,
    name="check_destination_in_bucket_list",
    description="Check if a specific destination is in the bucket list and get its key highlights."
)

# All available travel tools
TOOLS = [
    search_destinations_tool,
    region_search_tool,
    experience_search_tool,
    destination_details_tool,
    compare_tool,
    experience_tool,
    check_destination_tool,
]


# ============================================================================
# ADK-Traced Functions
# ============================================================================

@trace_tool("tool_execution")
async def execute_tools(tool_calls: list) -> dict:
    """Execute tool calls and return results."""
    results = {}
    for tool_call in tool_calls:
        tool_name = tool_call.tool_name
        tool_kwargs = tool_call.tool_kwargs
        
        for tool in TOOLS:
            if tool.metadata.name == tool_name:
                try:
                    result = tool.fn(**tool_kwargs)
                    results[tool_name] = result
                except Exception as e:
                    results[tool_name] = f"Error: {str(e)}"
                break
        else:
            results[tool_name] = f"Error: Unknown tool '{tool_name}'"
    
    return results


@trace_llm("agent_reasoning")
async def agent_step(messages: List[ChatMessage], tools: list) -> tuple:
    """Perform one step of agent reasoning."""
    response = await llm.achat_with_tools(
        tools=tools,
        chat_history=messages[:-1] if len(messages) > 1 else [],
        user_msg=messages[-1].content if messages else "",
    )
    
    tool_calls = llm.get_tool_calls_from_response(response, error_on_no_tool_call=False)
    return response, tool_calls


@entrypoint
async def main(data: dict, context: dict) -> dict:
    """
    Travel Agent entrypoint - Your AI-powered bucket list travel assistant.
    
    This agent can:
    1. Search bucket list destinations worldwide
    2. Find destinations by region (Europe, Asia, etc.)
    3. Find destinations by experience type (beaches, adventure, culture)
    4. Get detailed info about specific destinations
    5. Compare two destinations
    6. Find specific bucket list experiences (Northern Lights, safaris, etc.)
    7. Check if a place is in the bucket list
    
    Args:
        data: Input containing 'prompt' or 'query'
        context: ADK context
    
    Returns:
        Travel recommendations with reasoning trace
    """
    query = data.get("prompt") or data.get("query", "")
    
    if not query:
        return {"error": "No query provided. Ask me about bucket list destinations!"}
    
    # Initialize conversation with travel agent persona
    messages = [
        ChatMessage(
            role="system",
            content="""You are an expert travel agent specializing in bucket list destinations.
You have access to a comprehensive database of the world's top 100+ bucket list destinations 
from bucketlisttravels.com.

Your tools:
1. search_bucket_list_destinations - General search for any travel query
2. find_destinations_by_region - Find destinations in Europe, Asia, Africa, etc.
3. find_destinations_by_experience - Find beaches, adventure, culture, food destinations
4. get_destination_details - Deep dive into a specific destination
5. compare_destinations - Compare two places side by side
6. find_bucket_list_experiences - Find specific experiences (Northern Lights, safaris, etc.)
7. check_destination_in_bucket_list - Verify if a place is a bucket list destination

Always use the appropriate tool to find accurate information before answering.
Provide helpful, enthusiastic travel advice based on the knowledge base results.
Keep responses concise but informative."""
        ),
        ChatMessage(role="user", content=query)
    ]
    
    # Single tool call + response pattern (simpler, more reliable)
    reasoning_trace = []
    
    # Step 1: Let the LLM decide which tool to use
    response, tool_calls = await agent_step(messages, TOOLS)
    
    if tool_calls:
        # Execute the tool calls
        tool_results = await execute_tools(tool_calls)
        
        reasoning_trace.append({
            "step": 1,
            "action": "tool_calls",
            "tools": [{"name": tc.tool_name, "args": tc.tool_kwargs} for tc in tool_calls],
            "results_preview": {k: v[:300] + "..." if len(v) > 300 else v for k, v in tool_results.items()}
        })
        
        # Build context from tool results
        tool_result_str = "\n\n".join([f"### {name} results:\n{result}" for name, result in tool_results.items()])
        
        # Step 2: Generate final response with tool results as context
        final_messages = [
            ChatMessage(
                role="system",
                content="""You are an expert travel agent. Based on the search results provided, 
give a helpful, enthusiastic response about bucket list destinations. 
Be concise but informative. Don't use markdown formatting."""
            ),
            ChatMessage(
                role="user",
                content=f"""User question: {query}

Search results from knowledge base:
{tool_result_str}

Please provide a helpful travel recommendation based on these results."""
            )
        ]
        
        final_response = await llm.achat(final_messages)
        
        reasoning_trace.append({
            "step": 2,
            "action": "final_response",
            "content": final_response.message.content
        })
    else:
        # No tools called, use direct response
        reasoning_trace.append({
            "step": 1,
            "action": "final_response",
            "content": response.message.content
        })
    
    return {
        "response": reasoning_trace[-1]["content"],
        "reasoning_trace": reasoning_trace,
        "tools_available": [t.metadata.name for t in TOOLS],
        "model": "openai-gpt-oss-120b",
        "powered_by": "LlamaIndex + DigitalOcean Gradient AI",
        "knowledge_base": "Bucket List Travel Destinations"
    }


# ============================================================================
# Example Queries for Testing
# ============================================================================

EXAMPLE_QUERIES = [
    "Is London in the bucket list?",
    "What are the best bucket list destinations in Europe?",
    "I want a beach vacation. What destinations do you recommend?",
    "Tell me about Machu Picchu",
    "Compare Paris and Barcelona for a romantic trip",
    "What bucket list experiences involve wildlife?",
    "I have 2 weeks. Should I visit Japan or Italy?",
    "What are must-see natural wonders of the world?",
    "Find me adventure destinations with mountains",
    "What destinations are good for food lovers?",
]


if __name__ == "__main__":
    import asyncio
    
    async def run_travel_agent():
        print("=" * 70)
        print("🌍 BUCKET LIST TRAVEL AGENT")
        print("   Powered by LlamaIndex + DigitalOcean Gradient AI")
        print("=" * 70)
        
        # Test query
        test_query = "What are the top bucket list destinations in Asia?"
        
        print(f"\n📝 Query: {test_query}\n")
        print("-" * 70)
        
        result = await main({"prompt": test_query}, {})
        
        print(f"\n✈️ Response:\n{result['response']}")
        print(f"\n🔧 Tools Used: {[t['tools'] for t in result['reasoning_trace'] if t['action'] == 'tool_calls']}")
        
        print("\n" + "=" * 70)
        print("💡 Try these example queries:")
        for i, q in enumerate(EXAMPLE_QUERIES[:5], 1):
            print(f"   {i}. {q}")
        print("=" * 70)
    
    asyncio.run(run_travel_agent())
