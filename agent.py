from agno.agent import Agent
from agno.playground import Playground
from agno.app.fastapi.app import FastAPIApp
from knowledge_base import knowledge_base
from agno.models.openai import OpenAIChat
from dotenv import load_dotenv
import os

load_dotenv()

# Load knowledge base (set to False for production)
knowledge_base.load(recreate=False)

agent = Agent(
    name="Knowledge Base Agent",
    agent_id="kb-agent", 
    model=OpenAIChat(id="gpt-4o"),  # Add the model
    knowledge=knowledge_base,
    search_knowledge=True,
    markdown=True,
)

# Create both Playground and FastAPI apps
playground = Playground(agents=[agent])
fastapi_app = FastAPIApp(agent=agent)

# Get the main app (Playground)
app = playground.get_app()

# Mount the FastAPI routes to the playground app
app.mount("/api", fastapi_app.get_app())

# For production deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7777))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "agent:app", 
        reload=False,
        port=port,
        host=host
    )
