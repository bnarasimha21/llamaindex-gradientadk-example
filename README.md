# LlamaIndex + Gradient ADK Example

RAG examples using:
- [llama-index-llms-digitalocean-gradientai](https://pypi.org/project/llama-index-llms-digitalocean-gradientai/) - LlamaIndex LLM integration
- [Gradient ADK](https://docs.digitalocean.com/products/gradient/) - Agent Development Kit for deployment & observability
- Gradient AI Knowledge Base - Document retrieval for RAG

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file:

```env
GRADIENT_MODEL_ACCESS_KEY=your-model-access-key
DIGITALOCEAN_API_TOKEN=your-api-token
DIGITALOCEAN_KB_ID=your-knowledge-base-uuid
```

## Examples

### Simple Example

Basic RAG with Knowledge Base search and LLM response:

```bash
python simple_example.py
```

### Tool Calling Example

Travel agent with multiple tools (search destinations, compare places, find experiences):

```bash
python tool_calling_example.py
```

## Resources

- [LlamaIndex DigitalOcean GradientAI PyPI](https://pypi.org/project/llama-index-llms-digitalocean-gradientai/)
- [Gradient ADK Documentation](https://docs.digitalocean.com/products/gradient/)
