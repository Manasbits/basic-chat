from agno.agent import Agent
from agno.playground import Playground, serve_playground_app
from agno.models.openai import OpenAIChat
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

# Add the database connection test function here
def test_db_connection():
    try:
        db_url = os.getenv("DATABASE_URL")

        if not db_url:
            print("DATABASE_URL not set!")
            return False
        
        # Test connection
        conn = psycopg2.connect(db_url)
        print("Database connection successful!")
        conn.close()
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

# Call the test function before loading knowledge base
print("Testing database connection...")
test_db_connection()

# Add debug logging before loading
print(f"DATABASE_URL set: {bool(db_url)}")
print(f"DATABASE_URL set: {bool(os.getenv('DATABASE_URL'))}")
print(f"Loading knowledge base...")

try:
    # Load knowledge base (set to False for production)
    knowledge_base.load(recreate=False)
    print(f"Knowledge base loaded successfully")
    print(f"Document chunks: {len(knowledge_base.document_chunks) if hasattr(knowledge_base, 'document_chunks') else 'Unknown'}")
except Exception as e:
    print(f"Error loading knowledge base: {e}")
    raise

# Configure memory
memory = AgentMemory(
    db=PgMemoryDb(
        table_name="agent_memory",
        db_url=db_url,
    ),
    create_user_memories=True,
    update_user_memories_after_run=True,
    create_session_summary=True,
    update_session_summary_after_run=True,
)

# Configure storage
storage = PostgresStorage(
    table_name="agent_sessions", 
    db_url=db_url, 
    auto_upgrade_schema=True
)

agent = Agent(
    name="Knowledge Base Agent",
    agent_id="kb-agent", 
    model=OpenAIChat(id="gpt-4o"),
    role="Expert assistant with access to knowledge base and web search capabilities",
    instructions=[
        "Always search your knowledge base first for relevant information",
        "If the knowledge base doesn't have sufficient information, use web search to supplement",
        "Clearly indicate the source of your information (knowledge base vs web search)",
        "Provide comprehensive and accurate responses",
        "Use markdown formatting for better readability",
        "Perform comprehensive fundamental and performance analysis of Indian companies using a detailed CSV dataset containing financial metrics for all Indian stocks, including columns: Name, BSE Code, NSE Code, Industry, Current Price, Price to Earning, Market Capitalization, Earnings Yield, Div plus Earning Yield, CROIC, Return on Assets, PEG Ratio, NPM last year, Change in Promoter Holding 3Years, Sales Growth 3Years, EPS Growth 3Years, Debt to Equity, Dividend Yield, Dividend Payout Ratio, Price to Book Value, Pledged Percentage, EPS Growth 10Years, Return over 1year, Return over 10years, Return on Capital Employed, Price to Free Cash Flow, Price to Sales, FCF to EBIT, FCF to NW, FCF by OpCF, Sales, OPM, Profit after Tax, Sales Latest Quarter, Profit after Tax Latest Quarter, YOY Quarterly Sales Growth, YOY Quarterly Profit Growth, Return on Equity, EPS, Debt, Promoter Holding, Change in Promoter Holding, Industry PE, Sales Growth, Profit Growth, EVEBITDA, Enterprise Value, Current Ratio, Interest Coverage Ratio, Return over 3months, Return over 6months, Sales Growth 5Years, Profit Growth 3Years, Profit Growth 5Years, Average Return on Equity 5Years, Average Return on Equity 3Years, Return over 3years, Return over 5years, Sales Last Year",
        "Ensure accuracy in data retrieval by using the correct company name or symbol (BSE Code/NSE Code) from the CSV, cross-referencing Name, BSE Code, and NSE Code to avoid errors",
        "Interpret metrics as follows: Use P/E, PEG Ratio, Price to Book Value, Price to Sales, and EVEBITDA for valuation; NPM last year, ROA, ROE, ROCE, and OPM for profitability; Sales Growth (3Years, 5Years), EPS Growth (3Years, 10Years), Profit Growth (3Years, 5Years), and YOY Quarterly Sales/Profit Growth for growth; Debt to Equity, Current Ratio, and Interest Coverage Ratio for financial health; Price to Free Cash Flow, FCF to EBIT, FCF to NW, and FCF by OpCF for cash flow analysis; Dividend Yield, Dividend Payout Ratio, and Div plus Earning Yield for dividend sustainability; Promoter Holding, Change in Promoter Holding, and Pledged Percentage for ownership and governance; Return over 1year, 3years, 5years, 10years, 3months, and 6months for market performance; CROIC for capital reinvestment efficiency",
        "When comparing companies (e.g., PAT margin, ROE), fetch correct metrics (e.g., NPM for PAT margin) and contextualize against Industry PE or peers in the same Industry",
        "Handle missing or invalid data gracefully, noting unavailability and suggesting alternative metrics or qualitative insights",
        "Provide industry-specific insights using the Industry column, e.g., high Debt to Equity may be acceptable for capital-intensive sectors like Power but risky for IT",
        "Combine multiple metrics for a balanced view, highlighting trade-offs (e.g., low P/E may suggest undervaluation but high Debt to Equity signals risk)",
        "Offer actionable recommendations based on analysis, supported by data-driven reasoning, indicating if a company is a strong investment, value trap, or needs further investigation",
        "Respond to queries precisely, e.g., for PAT margin comparisons, use NPM last year and provide context with industry trends or historical performance",
        "Present findings in markdown with tables or bullet points for key metrics, including qualitative insights for trends or anomalies",
        "If generating visualizations, use Recharts in a standalone HTML file with Tailwind CSS, loading data via loadFileData('query-results - Copy.csv') and processing with Papa.parse",
        "Act as a seasoned financial analyst, delivering precise, actionable, and comprehensive insights while ensuring data accuracy and relevance"
    ],
    tools=[DuckDuckGoTools()],
    knowledge=knowledge_base,
    search_knowledge=True,
    show_tool_calls=True,
    markdown=True,
    # Memory configuration
    memory=memory,
    # Storage configuration
    storage=storage,
    # Chat history configuration
    add_history_to_messages=True,
    num_history_responses=5,
    read_chat_history=True,
)

# Create playground app
playground = Playground(
    agents=[agent],
    name="Knowledge Base Agent",
    description="A knowledge base agent playground with memory",
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
