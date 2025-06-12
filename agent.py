from agno.agent import Agent
from agno.playground import Playground, serve_playground_app
from agno.models.openai import OpenAIChat
from agno.models.google.gemini import Gemini
from agno.models.anthropic import Claude
from agno.models.deepseek import DeepSeek
from knowledge_base import knowledge_base
from dotenv import load_dotenv
import os
import sys
from agno.tools.csv_toolkit import CsvTools
from pathlib import Path

from agno.memory.agent import AgentMemory
from agno.memory.db.postgres import PgMemoryDb
from agno.storage.postgres import PostgresStorage
from agno.tools.duckduckgo import DuckDuckGoTools

load_dotenv()

# Get database URL
db_url = os.getenv("DATABASE_URL")

# ✅ Load the knowledge base with proper error handling
try:
    print("🔄 Loading knowledge base...")
    knowledge_base.load(recreate=False)
    print("✅ Knowledge base loaded successfully")
except Exception as e:
    print(f"❌ Failed to load knowledge base: {e}")
    print("This might be due to:")
    print("1. Database connection issues")
    print("2. Missing pgvector extension")
    print("3. Insufficient database permissions")
    print("4. CSV file not found or corrupted")
    sys.exit(1)

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

# DeepSeek Agent
deepseek_4o_agent = Agent(
    name="DeepSeek OpenAI",
    agent_id="deepseek-kb-agent",
    model=OpenAIChat(id="gpt-4o"),
    reasoning_model=DeepSeek(id="deepseek-reasoner"),
    role="Expert financial assistant powered by DeepSeek with access to knowledge base and web search capabilities",
    instructions=[
        "Leverage DeepSeek's capabilities for detailed financial analysis",
        "Provide clear and concise financial recommendations"
    ],
    tools=[DuckDuckGoTools()],
    knowledge=knowledge_base,
    search_knowledge=True,
    show_tool_calls=True,
    markdown=True,
    memory=create_memory("deepseek_agent"),
    storage=create_storage("deepseek_agent"),
    add_history_to_messages=True,
    num_history_responses=5,
    read_chat_history=True,
    exponential_backoff=True,
    delay_between_retries=5,
)


# Gemini Reasoning Agent
pure_deepseek_agent = Agent(
    name="Deepseek Pure",
    agent_id="deepseek-gemini-agent",
    model=DeepSeek(id="deepseek-reasoner"),
    role="Expert financial assistant powered by Deepseek's reasoning capabilities with access to knowledge base and web search",
    instructions=[
        "Leverage Deepseek's advanced reasoning capabilities for detailed financial analysis",
        "Provide clear and concise financial recommendations with strong analytical backing"
    ],
    tools=[DuckDuckGoTools()],
    knowledge=knowledge_base,
    search_knowledge=True,
    show_tool_calls=True,
    markdown=True,
    memory=create_memory("gemini_reasoning_agent"),
    storage=create_storage("gemini_reasoning_agent"),
    add_history_to_messages=True,
    num_history_responses=5,
    read_chat_history=True,
)

financial_csv = Path("./query_results.csv")

new_csvQueryagent = Agent(
    model=DeepSeek(id="deepseek-reasoner"),
    tools=[CsvTools(csvs=["query_results.csv"])],
    markdown=True,
    instructions=[
        # Data accuracy and validation
        "CRITICAL: Only use data that is actually present in the CSV file",
        "Never make assumptions or use external knowledge about companies",
        "If data is not available in the CSV, clearly state 'Data not available'",
        
        # Financial analysis workflow
        "FINANCIAL ANALYSIS WORKFLOW:",
        "1. First, examine available columns using get_columns()",
        "2. Query the specific company data from the CSV",
        "3. Present key financial metrics clearly",
        "4. Provide analysis based only on available data",
        
        # Query best practices
        "CSV QUERY GUIDELINES:",
        "- Use exact column names wrapped in double quotes",
        "- Try exact company name match first",
        "- If no results, try partial matching with LIKE operator",
        "- Always verify data exists before analyzing",
        
        # Financial analysis focus
        "FINANCIAL ANALYSIS REQUIREMENTS:",
        "- Present key financial ratios and metrics",
        "- Compare performance indicators when multiple periods available",
        "- Highlight significant financial trends or patterns",
        "- Provide clear interpretation of financial health",
        "- Use proper financial terminology",
        
        # Data presentation
        "PRESENTATION STANDARDS:",
        "- Show actual values from CSV without modification",
        "- Use clear formatting for financial figures",
        "- Organize analysis in logical sections (Revenue, Profitability, etc.)",
        "- Provide context for financial metrics when possible",
        "- State data source and time period clearly"
    ],
    description="Financial analyst specializing in CSV-based financial data analysis"
)

# Create playground with all agents
playground = Playground(
    agents=[claude_agent, deepseek_4o_agent, pure_deepseek_agent, new_csvQueryagent],
    name="Multi-Model Financial Analysis Playground",
    description="Compare financial analysis capabilities across Anthropic Claude, DeepSeek, and DeepSeek Pure",
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