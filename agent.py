from agno.agent import Agent
from agno.playground import Playground, serve_playground_app
from agno.models.openai import OpenAIChat
from agno.models.google.gemini import Gemini
from agno.models.anthropic import Claude
from knowledge_base import knowledge_base
from dotenv import load_dotenv
import os

import psycopg2
from urllib.parse import urlparse
from agno.memory.agent import AgentMemory
from agno.memory.db.postgres import PgMemoryDb
from agno.storage.postgres import PostgresStorage
from agno.tools.duckduckgo import DuckDuckGoTools

load_dotenv()

# Get database URL
db_url = os.getenv("DATABASE_URL")

# Database connection test function
def test_db_connection():
    try:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            print("DATABASE_URL not set!")
            return False
        
        conn = psycopg2.connect(db_url)
        print("Database connection successful!")
        conn.close()
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

# Test database connection
print("Testing database connection...")
test_db_connection()

print(f"DATABASE_URL set: {bool(db_url)}")
print(f"Loading knowledge base...")

try:
    knowledge_base.load(recreate=False)
    print(f"Knowledge base loaded successfully")
    print(f"Document chunks: {len(knowledge_base.document_chunks) if hasattr(knowledge_base, 'document_chunks') else 'Unknown'}")
except Exception as e:
    print(f"Error loading knowledge base: {e}")
    raise

# Shared instructions for all agents
shared_instructions = [
    "Always search your knowledge base first for relevant information",
    "If the knowledge base doesn't have sufficient information, use web search to supplement",
    "Clearly indicate the source of your information (knowledge base vs web search)",
    "Provide comprehensive and accurate responses",
    "Use markdown formatting for better readability",
    "Perform comprehensive fundamental and performance analysis of Indian companies using a detailed CSV dataset containing financial metrics for all Indian stocks",
    "Ensure accuracy in data retrieval by using the correct company name or symbol (BSE Code/NSE Code) from the CSV",
    "Interpret metrics for valuation, profitability, growth, financial health, cash flow analysis, dividend sustainability, ownership and governance, and market performance",
    "When comparing companies, fetch correct metrics and contextualize against Industry PE or peers in the same Industry",
    "Handle missing or invalid data gracefully, noting unavailability and suggesting alternative metrics",
    "Provide industry-specific insights using the Industry column",
    "Combine multiple metrics for a balanced view, highlighting trade-offs",
    "Offer actionable recommendations based on analysis, supported by data-driven reasoning",
    "Present findings in markdown with tables or bullet points for key metrics",
    "Act as a seasoned financial analyst, delivering precise, actionable, and comprehensive insights"
]

# Configure shared memory and storage settings
def create_memory(agent_name):
    return AgentMemory(
        db=PgMemoryDb(
            table_name=f"{agent_name}_memory",
            db_url=db_url,
        ),
        create_user_memories=True,
        update_user_memories_after_run=True,
        create_session_summary=True,
        update_session_summary_after_run=True,
    )

def create_storage(agent_name):
    return PostgresStorage(
        table_name=f"{agent_name}_sessions", 
        db_url=db_url, 
        auto_upgrade_schema=True
    )

# OpenAI GPT-4o Agent
openai_agent = Agent(
    name="OpenAI Knowledge Agent",
    agent_id="openai-kb-agent", 
    model=OpenAIChat(id="gpt-4o"),
    role="Expert financial assistant powered by OpenAI GPT-4o with access to knowledge base and web search capabilities",
    instructions=shared_instructions,
    tools=[DuckDuckGoTools()],
    knowledge=knowledge_base,
    search_knowledge=True,
    show_tool_calls=True,
    markdown=True,
    memory=create_memory("openai_agent"),
    storage=create_storage("openai_agent"),
    add_history_to_messages=True,
    num_history_responses=5,
    read_chat_history=True,
)

# Google Gemini Agent
gemini_agent = Agent(
    name="Gemini Knowledge Agent",
    agent_id="gemini-kb-agent",
    model=Gemini(id="gemini-2.0-flash"),
    role="Expert financial assistant powered by Google Gemini with access to knowledge base and web search capabilities",
    instructions=shared_instructions + [
        "Provide detailed reasoning for financial recommendations using Gemini's advanced reasoning capabilities"
    ],
    tools=[DuckDuckGoTools()],
    knowledge=knowledge_base,
    search_knowledge=True,
    show_tool_calls=True,
    markdown=True,
    memory=create_memory("gemini_agent"),
    storage=create_storage("gemini_agent"),
    add_history_to_messages=True,
    num_history_responses=5,
    read_chat_history=True,
)

# Anthropic Claude Agent
claude_agent = Agent(
    name="Claude Knowledge Agent",
    agent_id="claude-kb-agent",
    model=Claude(id="claude-3-5-sonnet-latest"),
    role="Expert financial assistant powered by Anthropic Claude with access to knowledge base and web search capabilities",
    instructions=shared_instructions + [
        "Apply Claude's strong analytical reasoning to provide thorough financial analysis",
        "Use Claude's excellent writing capabilities to present complex financial data clearly",
        "Leverage Claude's careful and nuanced approach to financial recommendations"
    ],
    tools=[DuckDuckGoTools()],
    knowledge=knowledge_base,
    search_knowledge=True,
    show_tool_calls=True,
    markdown=True,
    memory=create_memory("claude_agent"),
    storage=create_storage("claude_agent"),
    add_history_to_messages=True,
    num_history_responses=5,
    read_chat_history=True,
)

# Create playground with all three agents
playground = Playground(
    agents=[openai_agent, gemini_agent, claude_agent],
    name="Multi-Model Financial Analysis Playground",
    description="Compare financial analysis capabilities across OpenAI GPT-4o, Google Gemini, and Anthropic Claude",
    app_id="multi-model-kb-agent-playground",
)

app = playground.get_app()

# For production deployment
if __name__ == "__main__":
    port = int(os.getenv("PORT", 7777))
    host = os.getenv("HOST", "0.0.0.0")
    
    serve_playground_app(
        "agent:app", 
        reload=False,
        port=port,
        host=host
    )