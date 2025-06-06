from agno.agent import Agent
from agno.playground import Playground, serve_playground_app
from knowledge_base import knowledge_base
from dotenv import load_dotenv
import os

load_dotenv()

# Load knowledge base (set to False for production)
knowledge_base.load(recreate=False)

agent = Agent(
    name="Knowledge Base Agent",
    agent_id="kb-agent", 
    knowledge=knowledge_base,
    search_knowledge=True,
    markdown=True,
)

# Create playground app
app = Playground(agents=[agent]).get_app()

# For production deployment
if __name__ == "__main__":
    # Use environment variables for production
    port = int(os.getenv("PORT", 7777))
    host = os.getenv("HOST", "0.0.0.0")  # Important for AWS
    
    serve_playground_app(
        "agent:app", 
        reload=False,  # Set to False for production
        port=port,
        host=host
    )