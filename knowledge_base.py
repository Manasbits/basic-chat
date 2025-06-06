from agno.knowledge.csv import CSVKnowledgeBase
from agno.vectordb.pgvector import PgVector
import os

knowledge_base = CSVKnowledgeBase(
    path="query-results.csv",
    vector_db=PgVector(
        table_name="csv_documents",
        db_url=os.getenv("DATABASE_URL"),  # Use environment variable
    ),
)
