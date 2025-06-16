from agno.workflow import Workflow, RunResponse, RunEvent
from typing import Iterator, Optional
import json
from pathlib import Path
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.deepseek import DeepSeek
from agno.tools.csv_toolkit import CsvTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.reasoning import ReasoningTools
from dotenv import load_dotenv
from agno.tools.tavily import TavilyTools
from agno.app.agui.app import AGUIApp

load_dotenv()

# Agent 1: Stock Data Model and Agent
class StockData(BaseModel):
    name: str = Field(..., description="Company Name as registered on stock exchanges")
    bse_code: str = Field(..., description="Ticker symbol on Bombay Stock Exchange")
    nse_code: str = Field(..., description="Ticker symbol on National Stock Exchange of India")
    industry: str = Field(..., description="Sector or industry classification of the company")
    current_price: Optional[float] = Field(..., description="Latest market price per share in INR")
    price_to_earning: Optional[float] = Field(..., description="Price-to-Earnings ratio")
    market_capitalization: Optional[float] = Field(..., description="Total market value of outstanding shares in INR")
    earnings_yield: Optional[float] = Field(..., description="Annual earnings per share divided by current share price")
    div_plus_earning_yield: Optional[float] = Field(..., description="Sum of dividend yield and earnings yield")
    croic: Optional[float] = Field(..., description="Cash Return on Invested Capital")
    return_on_assets: Optional[float] = Field(..., description="Net income divided by average total assets")
    peg_ratio: Optional[float] = Field(..., description="Price/Earnings to Growth ratio")
    npm_last_year: Optional[float] = Field(..., description="Net Profit Margin for the most recent fiscal year")
    change_in_promoter_holding_3years: Optional[float] = Field(..., description="Three-year change in percentage points of promoter shareholding")
    sales_growth_3years: Optional[float] = Field(..., description="Three-year compound annual growth rate of sales revenue")
    eps_growth_3years: Optional[float] = Field(..., description="Three-year compound annual growth rate of earnings per share")
    debt_to_equity: Optional[float] = Field(..., description="Total debt divided by shareholders equity")
    dividend_yield: Optional[float] = Field(..., description="Annual dividend per share divided by current price")
    dividend_payout_ratio: Optional[float] = Field(..., description="Percentage of net income distributed as dividends")
    price_to_book_value: Optional[float] = Field(..., description="Market price divided by book value per share")
    pledged_percentage: Optional[float] = Field(..., description="Percentage of promoter shares pledged as loan collateral")
    eps_growth_10years: Optional[float] = Field(..., description="Ten-year compound annual growth rate of earnings per share")
    return_over_1year: Optional[float] = Field(..., description="Total shareholder return including dividends over past 1 year")
    return_over_10years: Optional[float] = Field(..., description="Total shareholder return including dividends over past 10 years")

csv_data_agent = Agent(
    model=DeepSeek(id="deepseek-chat"),
    tools=[CsvTools(csvs=[Path("query_results.csv")])],
    response_model=StockData,
    description="You are a financial data extraction specialist.",
    instructions=[
        "First always get the list of files and check the columns in the file",
        "Follow this search strategy in sequence:",
        "1. Try to find the company by NSE code first (exact match)",
        "2. If not found, try to find by company name (exact match)",
        "3. If still not found, try to find similar company names using LIKE operator",
        "4. If multiple matches found, return the most relevant match",
        "Always wrap column names with double quotes if they contain spaces or special characters",
        "For NSE code search, use: SELECT * FROM query_results WHERE \"NSE Code (Ticker symbol on National Stock Exchange of India)\" = '[NSE_CODE]'",
        "For company name search, use: SELECT * FROM query_results WHERE \"Name (Company Name as registered on stock exchanges)\" = '[COMPANY_NAME]'",
        "For similar names search, use: SELECT * FROM query_results WHERE \"Name (Company Name as registered on stock exchanges)\" LIKE '%[PARTIAL_NAME]%'",
        "Return structured data with all available financial metrics",
        "If no match is found after all attempts, return null"
    ],
    show_tool_calls=True
)

# Agent 2: Meta Prompt Model and Agent
class MetaPrompt(BaseModel):
    industry_context: str = Field(..., description="Industry-specific context and trends")
    company_context: str = Field(..., description="Company-specific context and background")
    analysis_framework: str = Field(..., description="Tailored analysis framework for this industry/company")
    key_metrics_focus: str = Field(..., description="Most important metrics to focus on for this industry")
    risk_factors: str = Field(..., description="Industry and company-specific risk factors to consider")
    meta_prompt: str = Field(..., description="Complete meta prompt for stock analysis")

meta_prompt_agent = Agent(
    model=DeepSeek(id="deepseek-reasoner"),
    tools=[TavilyTools()],
    response_model=MetaPrompt,
    description="You are an expert financial analyst and prompt engineer who creates specialized analysis frameworks.",
    instructions=[
        "Research the specific industry and company using web search",
        "Use reasoning tools to think through industry-specific factors",
        "Create a comprehensive meta prompt that considers:",
        "- Industry-specific valuation methods and key metrics",
        "- Current industry trends and challenges", 
        "- Company's position within the industry",
        "- Regulatory environment and market conditions",
        "- Seasonal factors and business cycles",
        "Generate a detailed meta prompt for stock analysis tailored to this specific company and industry"
    ],
    show_tool_calls=True
)

# Agent 3: Stock Recommendation Model and Agent
class StockRecommendation(BaseModel):
    recommendation: str = Field(..., description="BUY, HOLD, or SELL recommendation")
    confidence_score: float = Field(..., description="Confidence score from 0-100")
    target_price: Optional[float] = Field(..., description="Target price if applicable")
    time_horizon: str = Field(..., description="Recommended investment time horizon")
    key_strengths: list[str] = Field(..., description="Key strengths supporting the recommendation")
    key_risks: list[str] = Field(..., description="Key risks and concerns")
    rationale: str = Field(..., description="Detailed rationale for the recommendation")
    alternative_scenarios: str = Field(..., description="Alternative scenarios and their implications")

analysis_agent = Agent(
    model=DeepSeek(id="deepseek-reasoner"),
    tools=[TavilyTools()],
    response_model=StockRecommendation,
    description="You are a senior equity research analyst with 15+ years of experience in Indian stock markets.",
    instructions=[
        "Use the provided meta prompt as your analysis framework",
        "Analyze the structured financial data comprehensively",
        "Search for recent news and developments about the company",
        "Use reasoning tools to work through your analysis step by step",
        "Consider multiple valuation approaches",
        "Evaluate both quantitative metrics and qualitative factors",
        "Provide a clear BUY/HOLD/SELL recommendation with detailed rationale",
        "Include confidence score and risk assessment"
    ],
    show_tool_calls=True,
    markdown=True
)

class StockAnalysisWorkflow(Workflow):
    description: str = "Complete stock analysis workflow with three specialized agents"
    
    def __init__(self):
        super().__init__()
        self.csv_agent = csv_data_agent
        self.meta_prompt_agent = meta_prompt_agent  
        self.analysis_agent = analysis_agent
    
    def run_workflow(self, message: str) -> Iterator[RunResponse]:
        """
        Run the complete stock analysis workflow
        
        Args:
            message: Company query (e.g., "RELIANCE" or "Reliance Industries")
        """
        company_query = message
        
        # Step 1: Extract structured data from CSV
        yield RunResponse(content="🔍 Extracting company data from CSV...")
        
        csv_response = self.csv_agent.run(f"Find and extract all data for company: {company_query}")
        if not csv_response.content:
            yield RunResponse(content="❌ Company not found in CSV data")
            return
            
        stock_data = csv_response.content
        yield RunResponse(content=f"✅ Extracted data for {stock_data.name}")
        
        # Step 2: Generate meta prompt based on industry and company
        yield RunResponse(content="🧠 Generating specialized analysis framework...")
        
        meta_prompt_input = f"Industry: {stock_data.industry}, Company: {stock_data.name}"
        meta_response = self.meta_prompt_agent.run(meta_prompt_input)
        meta_prompt = meta_response.content
        
        yield RunResponse(content="✅ Generated industry-specific analysis framework")
        
        # Step 3: Perform comprehensive stock analysis
        yield RunResponse(content="📊 Performing comprehensive stock analysis...")
        
        analysis_input = {
            "meta_prompt": meta_prompt.meta_prompt,
            "stock_data": stock_data.model_dump(),
            "company_name": stock_data.name,
            "industry": stock_data.industry
        }
        
        analysis_response = self.analysis_agent.run(json.dumps(analysis_input, indent=2))
        recommendation = analysis_response.content
        
        # Final output
        yield RunResponse(
            content=f"""
# Stock Analysis Report: {stock_data.name}

## 🎯 Recommendation: {recommendation.recommendation}
**Confidence Score:** {recommendation.confidence_score}/100
**Target Price:** ₹{recommendation.target_price if recommendation.target_price else 'N/A'}
**Time Horizon:** {recommendation.time_horizon}

## 📈 Key Strengths
{chr(10).join([f"• {strength}" for strength in recommendation.key_strengths])}

## ⚠️ Key Risks  
{chr(10).join([f"• {risk}" for risk in recommendation.key_risks])}

## 🔍 Analysis Rationale
{recommendation.rationale}

## 🎲 Alternative Scenarios
{recommendation.alternative_scenarios}

---
*Analysis completed using industry-specific framework for {stock_data.industry} sector*
            """
        )

# Create AGUI App
agui_app = AGUIApp(
    agent=StockAnalysisWorkflow(),
    name="Stock Analysis Expert",
    app_id="stock_analysis_expert",
    description="A comprehensive stock analysis system that provides detailed recommendations based on financial data, industry context, and market trends.",
)

app = agui_app.get_app()

if __name__ == "__main__":
    # For playground usage
    agui_app.serve(app="workflow:app", port=8000, reload=True)
    
    # For direct usage
    # workflow = StockAnalysisWorkflow()
    # for response in workflow.run_workflow("ANGELONE"):
    #     print(response.content)