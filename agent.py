from agno.agent import Agent
from agno.playground import Playground, serve_playground_app
from agno.models.openai import OpenAIChat
from knowledge_base import knowledge_base
from dotenv import load_dotenv
import os

load_dotenv()

# Add debug logging before loading
print(f"DATABASE_URL set: {bool(os.getenv('DATABASE_URL'))}")
print(f"Loading knowledge base...")

try:
    # Load knowledge base (set to False for production)
    knowledge_base.load(recreate=True)
    print(f"Knowledge base loaded successfully")
    print(f"Document chunks: {len(knowledge_base.document_chunks) if hasattr(knowledge_base, 'document_chunks') else 'Unknown'}")
except Exception as e:
    print(f"Error loading knowledge base: {e}")
    raise

agent = Agent(
    name="Knowledge Base Agent",
    agent_id="kb-agent", 
    model=OpenAIChat(id="gpt-4o"),
    knowledge=knowledge_base,
    search_knowledge=True,
    markdown=True,
)

# Create playground app
playground = Playground(
    agents=[agent],
    name="Knowledge Base Agent",
    description="A knowledge base agent playground",
    app_id="kb-agent-playground",
)

app = playground.get_app()

# For production deployment
if __name__ == "__main__":
    # Use environment variables for production
    port = int(os.getenv("PORT", 7777))
    host = os.getenv("HOST", "0.0.0.0")
    
    serve_playground_app(
        "agent:app", 
        reload=False,  # Set to False for production
        port=port,
        host=host
    )
