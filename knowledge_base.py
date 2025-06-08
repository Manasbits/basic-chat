import os
from pathlib import Path

# Add these debug lines
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")
print(f"CSV file exists: {Path('query-results.csv').exists()}")

knowledge_base = CSVKnowledgeBase(
    path="query-results.csv",
    vector_db=PgVector(
        table_name="csv_documents",
        db_url=os.getenv("DATABASE_URL"),
    ),
)
