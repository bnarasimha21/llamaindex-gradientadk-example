"""
Travel Agent (Simple RAG): LlamaIndex GradientAI with Bucket List Knowledge Base
=================================================================================

This example demonstrates a travel-focused AI agent using a simple RAG approach:
- Bucket List Travel Knowledge Base (bucketlisttravels.com)
- LlamaIndex GradientAI for response generation
- Gradient ADK for observability and tracing

This is the simpler version without tool calling - it directly searches the
knowledge base and generates responses using the retrieved context.

Knowledge Base Source: https://www.bucketlisttravels.com/round-up/100-bucket-list-destinations

It combines:
- gradient_adk: For agent deployment with observability/tracing
- llama-index-llms-digitalocean-gradientai: LlamaIndex LLM integration
- gradient SDK: For Knowledge Base retrieval

Environment Variables Required:
- GRADIENT_MODEL_ACCESS_KEY: Your model access key for inference
- DIGITALOCEAN_API_TOKEN: Your DigitalOcean API token
- DIGITALOCEAN_KB_ID: Your Knowledge Base ID (travel KB)
"""

import os
from typing import List, Optional

from gradient_adk import entrypoint, trace_llm, trace_retriever, trace_tool
from gradient import Gradient
from llama_index.llms.digitalocean.gradientai import GradientAI
from llama_index.core.llms import ChatMessage
from dotenv import load_dotenv

load_dotenv()

# Debug: Check environment variables
print("=" * 70)
print("🌍 BUCKET LIST TRAVEL AGENT (Simple RAG)")
print("   Powered by LlamaIndex + DigitalOcean Gradient AI")
print("=" * 70)
print("🔧 Environment Check:")
print(f"   GRADIENT_MODEL_ACCESS_KEY: {'✅ Set' if os.environ.get('GRADIENT_MODEL_ACCESS_KEY') else '❌ Missing'}")
print(f"   DIGITALOCEAN_API_TOKEN: {'✅ Set' if os.environ.get('DIGITALOCEAN_API_TOKEN') else '❌ Missing'}")
print(f"   DIGITALOCEAN_KB_ID: {os.environ.get('DIGITALOCEAN_KB_ID', '❌ Missing')}")
print("=" * 70)

# Initialize the Gradient client for Knowledge Base operations
gradient_client = Gradient(access_token=os.environ.get("DIGITALOCEAN_API_TOKEN"))

# Initialize the LlamaIndex GradientAI LLM
llm = GradientAI(
    model="openai-gpt-oss-120b",
    model_access_key=os.environ.get("GRADIENT_MODEL_ACCESS_KEY"),
)


@trace_retriever("bucket_list_search")
async def search_knowledge_base(query: str, num_results: int = 5) -> List[str]:
    """
    Search the Bucket List Travel Knowledge Base for destinations and travel info.
    
    This function is decorated with @trace_retriever to capture retrieval
    operations in the ADK observability dashboard.
    
    Args:
        query: The search query about travel destinations, experiences, or attractions
        num_results: Number of results to return (default: 5)
    
    Returns:
        List of relevant travel information from the bucket list knowledge base
    """
    kb_id = os.environ.get("DIGITALOCEAN_KB_ID")
    if not kb_id:
        print("❌ ERROR: DIGITALOCEAN_KB_ID environment variable not set")
        return ["Error: DIGITALOCEAN_KB_ID environment variable not set"]
    
    print(f"🔍 Searching Bucket List Travel KB: {kb_id}")
    print(f"📝 Travel Query: {query}")
    
    try:
        response = gradient_client.retrieve.documents(
            knowledge_base_id=kb_id,
            num_results=num_results,
            query=query,
        )
        
        print(f"📦 Response: {response}")
        
        if response and response.results:
            # Extract text content from results (attribute is 'text_content', not 'text')
            docs = [result.text_content for result in response.results if hasattr(result, 'text_content')]
            print(f"✅ Retrieved {len(docs)} travel destinations/info")
            return docs
        
        print("⚠️ No travel destinations found in Knowledge Base")
        return []
        
    except Exception as e:
        print(f"❌ Travel KB Search Error: {e}")
        return []


@trace_llm("travel_agent_generate")
async def generate_response_with_context(
    query: str,
    context: List[str],
    system_prompt: Optional[str] = None
) -> str:
    """
    Generate a travel recommendation using LlamaIndex GradientAI with retrieved context.
    
    This function is decorated with @trace_llm to capture LLM operations
    including token usage in the ADK observability dashboard.
    
    Args:
        query: The user's travel question or destination query
        context: List of relevant travel info from bucket list knowledge base
        system_prompt: Optional system prompt to customize behavior
    
    Returns:
        Generated travel recommendation string
    """
    system_prompt = """You are an expert travel agent specializing in bucket list destinations. 
You have access to a comprehensive database of the world's top 100+ bucket list destinations 
from bucketlisttravels.com. Provide helpful, enthusiastic travel advice based on the knowledge 
base results. Keep responses concise but informative. Don't use markdown formatting."""
    
    # Build context string from retrieved travel information
    context_str = "\n\n---\n\n".join(context) if context else "No travel destinations found."
    
    # Construct messages for chat
    messages = [
        ChatMessage(
            role="system",
            content=system_prompt
        ),
        ChatMessage(
            role="user",
            content=f"""Bucket List Travel Knowledge Base Results:
{context_str}

Travel Question: {query}

Please provide a helpful travel recommendation based on the destinations and information above."""
        )
    ]
    
    # Use LlamaIndex GradientAI for generation
    response = await llm.achat(messages)
    return response.message.content


@trace_tool("format_travel_response")
async def format_response(content: str, include_sources: bool = True) -> dict:
    """
    Format the final travel recommendation with metadata.
    
    This function is decorated with @trace_tool to capture tool execution
    in the ADK observability dashboard.
    
    Args:
        content: The generated travel recommendation content
        include_sources: Whether to include source attribution
    
    Returns:
        Formatted travel response dictionary
    """
    return {
        "response": content,
        "model": "openai-gpt-oss-120b",
        "powered_by": "LlamaIndex + DigitalOcean Gradient AI",
        "knowledge_base": "Bucket List Travel Destinations",
        "include_sources": include_sources
    }


@entrypoint
async def main(data: dict, context: dict) -> dict:
    """
    Travel Agent (Simple RAG) - Your AI-powered bucket list travel assistant.
    
    This is the entry point that Gradient ADK uses to invoke your agent.
    It orchestrates a simple RAG pipeline for travel recommendations:
    1. Extract travel query from input
    2. Search bucket list knowledge base for relevant destinations
    3. Generate travel recommendation using LlamaIndex GradientAI
    4. Format and return the travel response
    
    This agent can answer questions about:
    - Bucket list destinations worldwide
    - Travel experiences (beaches, adventure, culture, etc.)
    - Specific destination information
    - Travel recommendations by region
    
    Args:
        data: Input data containing 'prompt' or 'query' field
        context: ADK context (includes tracing, metadata, etc.)
    
    Returns:
        Dictionary with travel recommendations
    """
    # Extract query from input (support both 'prompt' and 'query' keys)
    query = data.get("prompt") or data.get("query", "")
    
    if not query:
        return {"error": "No query provided. Ask me about bucket list destinations!"}
    
    # Optional parameters
    num_results = data.get("num_results", 5)
    system_prompt = data.get("system_prompt", """You are an expert travel agent specializing in bucket list destinations. 
You have access to a comprehensive database of the world's top 100+ bucket list destinations 
from bucketlisttravels.com. Provide helpful, enthusiastic travel advice based on the knowledge 
base results. Keep responses concise but informative. Don't use markdown formatting.""")
    
    # Step 1: Search Bucket List Travel Knowledge Base
    retrieved_docs = await search_knowledge_base(query, num_results=num_results)
    
    # Step 2: Generate travel recommendation with context using LlamaIndex
    response_content = await generate_response_with_context(
        query=query,
        context=retrieved_docs,
        system_prompt=system_prompt
    )
    
    # Step 3: Format the travel response
    formatted_response = await format_response(
        content=response_content,
        include_sources=len(retrieved_docs) > 0
    )
    
    return formatted_response


# ============================================================================
# Example Queries for Testing
# ============================================================================

EXAMPLE_QUERIES = [
    "Is London in the bucket list?",
    "What are the best bucket list destinations in Europe?",
    "I want a beach vacation. What destinations do you recommend?",
    "Tell me about Machu Picchu",
    "What bucket list experiences involve wildlife?",
    "What are must-see natural wonders of the world?",
    "Find me adventure destinations with mountains",
    "What destinations are good for food lovers?",
    "Which places should I visit in India?",
    "What are the top bucket list destinations in Asia?",
]


# For local testing
if __name__ == "__main__":
    import asyncio
    
    async def run_travel_agent():
        print("\n" + "-" * 70)
        
        # Test query
        test_query = "What are the top bucket list destinations in Asia?"
        
        print(f"📝 Query: {test_query}\n")
        print("-" * 70)
        
        result = await main({"prompt": test_query, "num_results": 5}, {})
        
        print(f"\n✈️ Response:\n{result.get('response', result)}")
        print(f"\n🗺️ Knowledge Base: {result.get('knowledge_base', 'N/A')}")
        
        print("\n" + "=" * 70)
        print("💡 Try these example queries:")
        for i, q in enumerate(EXAMPLE_QUERIES[:5], 1):
            print(f"   {i}. {q}")
        print("=" * 70)
    
    asyncio.run(run_travel_agent())
