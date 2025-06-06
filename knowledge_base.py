from agno.knowledge.csv import CSVKnowledgeBase
from agno.vectordb.pgvector import PgVector

knowledge_base = CSVKnowledgeBase(
    path="query-results.csv",
    # Table name: ai.csv_documents
    vector_db=PgVector(
        table_name="csv_documents",
        db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
    ),
)

