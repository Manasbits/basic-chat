from pathlib import Path
from agno.agent import Agent
from agno.tools.csv_toolkit import CsvTools
from agno.models.deepseek import DeepSeek
from dotenv import load_dotenv
import os

load_dotenv()

# Path to your financial CSV data
financial_csv = Path("./query_results.csv")

agent = Agent(
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

# Example usage
print("=== Financial Analysis ===")
agent.print_response("Analyze the financials of Angel One", markdown=True)