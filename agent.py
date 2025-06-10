from agno.agent import Agent
from agno.playground import Playground, serve_playground_app
from agno.models.openai import OpenAIChat
from agno.models.google.gemini import Gemini
from agno.models.anthropic import Claude
from agno.models.huggingface import HuggingFace
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
    instructions="hi",
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
    instructions=[
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
    instructions= [
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

# HuggingFace Deepseek Agent
huggingface_agent = Agent(
    name="Deepseek Knowledge Agent",
    agent_id="Deepseek-kb-agent",
    model=HuggingFace(
        id="deepseek-ai/DeepSeek-R1-0528",
        max_tokens=8000,
        stream_output=False
    ),
    role="Expert financial assistant powered by Deepseek with access to knowledge base and web search capabilities",
    instructions=[
        "Leverage Deepseek's capabilities for detailed financial analysis",
        "Provide clear and concise financial recommendations"
    ],
    tools=[DuckDuckGoTools()],
    knowledge=knowledge_base,
    search_knowledge=True,
    show_tool_calls=True,
    markdown=True,
    memory=create_memory("huggingface_agent"),
    storage=create_storage("huggingface_agent"),
    add_history_to_messages=True,
    num_history_responses=5,
    read_chat_history=True,
)

# Create playground with all agents
playground = Playground(
    agents=[openai_agent, gemini_agent, claude_agent, huggingface_agent],
    name="Multi-Model Financial Analysis Playground",
    description="Compare financial analysis capabilities across OpenAI GPT-4o, Google Gemini, Anthropic Claude, and Deepseek",
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