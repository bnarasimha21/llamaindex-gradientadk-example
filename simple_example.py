"""
Example: LlamaIndex with DigitalOcean Gradient AI Knowledge Base
================================================================

This example demonstrates how to use the LlamaIndex DigitalOcean GradientAI
integration with the Gradient Agent Development Kit (ADK) and Knowledge Base.

It combines:
- gradient_adk: For agent deployment with observability/tracing
- llama-index-llms-digitalocean-gradientai: LlamaIndex LLM integration
- gradient SDK: For Knowledge Base retrieval

Environment Variables Required:
- GRADIENT_MODEL_ACCESS_KEY: Your model access key for inference
- DIGITALOCEAN_API_TOKEN: Your DigitalOcean API token
- DIGITALOCEAN_KB_ID: Your Knowledge Base ID
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
print("=" * 50)
print("🔧 Environment Check:")
print(f"   GRADIENT_MODEL_ACCESS_KEY: {'✅ Set' if os.environ.get('GRADIENT_MODEL_ACCESS_KEY') else '❌ Missing'}")
print(f"   DIGITALOCEAN_API_TOKEN: {'✅ Set' if os.environ.get('DIGITALOCEAN_API_TOKEN') else '❌ Missing'}")
print(f"   DIGITALOCEAN_KB_ID: {os.environ.get('DIGITALOCEAN_KB_ID', '❌ Missing')}")
print("=" * 50)

# Initialize the Gradient client for Knowledge Base operations
gradient_client = Gradient(access_token=os.environ.get("DIGITALOCEAN_API_TOKEN"))

# Initialize the LlamaIndex GradientAI LLM
llm = GradientAI(
    model="openai-gpt-oss-120b",
    model_access_key=os.environ.get("GRADIENT_MODEL_ACCESS_KEY"),
)


@trace_retriever("knowledge_base_search")
async def search_knowledge_base(query: str, num_results: int = 5) -> List[str]:
    """
    Search the DigitalOcean Gradient AI Knowledge Base.
    
    This function is decorated with @trace_retriever to capture retrieval
    operations in the ADK observability dashboard.
    
    Args:
        query: The search query
        num_results: Number of results to return (default: 5)
    
    Returns:
        List of relevant document chunks from the knowledge base
    """
    kb_id = os.environ.get("DIGITALOCEAN_KB_ID")
    if not kb_id:
        print("❌ ERROR: DIGITALOCEAN_KB_ID environment variable not set")
        return ["Error: DIGITALOCEAN_KB_ID environment variable not set"]
    
    print(f"🔍 Searching KB: {kb_id}")
    print(f"📝 Query: {query}")
    
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
            print(f"✅ Retrieved {len(docs)} documents")
            return docs
        
        print("⚠️ No results returned from Knowledge Base")
        return []
        
    except Exception as e:
        print(f"❌ KB Search Error: {e}")
        return []


@trace_llm("llamaindex_generate")
async def generate_response_with_context(
    query: str,
    context: List[str],
    system_prompt: Optional[str] = None
) -> str:
    """
    Generate a response using LlamaIndex GradientAI with retrieved context.
    
    This function is decorated with @trace_llm to capture LLM operations
    including token usage in the ADK observability dashboard.
    
    Args:
        query: The user's question
        context: List of relevant context chunks from knowledge base
        system_prompt: Optional system prompt to customize behavior
    
    Returns:
        Generated response string
    """
    system_prompt = """You are a expert travel agent who can answer any question from users about their travel needs. Answer should be brief and concise. Dont use markdowns or any special characters."""
    
    # Build context string from retrieved documents
    context_str = "\n\n---\n\n".join(context) if context else "No relevant context found."
    
    # Construct messages for chat
    messages = [
        ChatMessage(
            role="system",
            content=system_prompt
        ),
        ChatMessage(
            role="user",
            content=f"""Context from Knowledge Base:
{context_str}

Question: {query}

Please provide a helpful answer based on the context above."""
        )
    ]
    
    # Use LlamaIndex GradientAI for generation
    response = await llm.achat(messages)
    return response.message.content


@trace_tool("format_response")
async def format_response(content: str, include_sources: bool = True) -> dict:
    """
    Format the final response with metadata.
    
    This function is decorated with @trace_tool to capture tool execution
    in the ADK observability dashboard.
    
    Args:
        content: The generated response content
        include_sources: Whether to include source attribution
    
    Returns:
        Formatted response dictionary
    """
    return {
        "response": content,
        "model": "openai-gpt-oss-120b",
        "powered_by": "DigitalOcean Gradient AI + LlamaIndex",
        "include_sources": include_sources
    }


@entrypoint
async def main(data: dict, context: dict) -> dict:
    """
    Main entrypoint for the Gradient ADK agent.
    
    This is the entry point that Gradient ADK uses to invoke your agent.
    It orchestrates the RAG pipeline:
    1. Extract query from input
    2. Search knowledge base for relevant context
    3. Generate response using LlamaIndex GradientAI
    4. Format and return the response
    
    Args:
        data: Input data containing 'prompt' or 'query' field
        context: ADK context (includes tracing, metadata, etc.)
    
    Returns:
        Dictionary with the generated response
    """
    # Extract query from input (support both 'prompt' and 'query' keys)
    query = data.get("prompt") or data.get("query", "")
    
    if not query:
        return {"error": "No query provided. Please include 'prompt' or 'query' in your request."}
    
    # Optional parameters
    num_results = data.get("num_results", 5)
    system_prompt = data.get("system_prompt", "You are a expert travel agent who can answer any question from users about their travel needs. Answer should be brief and concise. Dont use markdowns or any special characters.")
    
    # Step 1: Search Knowledge Base
    retrieved_docs = await search_knowledge_base(query, num_results=num_results)
    
    # Step 2: Generate response with context using LlamaIndex
    response_content = await generate_response_with_context(
        query=query,
        context=retrieved_docs,
        system_prompt=system_prompt
    )
    
    # Step 3: Format the response
    formatted_response = await format_response(
        content=response_content,
        include_sources=len(retrieved_docs) > 0
    )
    
    return formatted_response


# For local testing
if __name__ == "__main__":
    import asyncio
    
    async def test():
        # Test data
        test_input = {
            "prompt": "Which places are in India?",
            "num_results": 5
        }
        
        result = await main(test_input, {})
        print("Response:", result)
    
    asyncio.run(test())

