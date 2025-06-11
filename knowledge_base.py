import os
from pathlib import Path
from agno.knowledge.csv import CSVKnowledgeBase
from agno.vectordb.pgvector import PgVector
from dotenv import load_dotenv
import psycopg
from urllib.parse import urlparse

load_dotenv()

# Enhanced debugging
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")

csv_path = Path('query-results.csv')
print(f"CSV file exists: {csv_path.exists()}")
if csv_path.exists():
    print(f"CSV file size: {csv_path.stat().st_size} bytes")

# AWS RDS Database URL
db_url = "postgresql+psycopg://mainuser:Manas123456@pgvector-db.czegegwgkvr5.ap-south-1.rds.amazonaws.com:5432/postgres?sslmode=require"

print(f"Database URL configured: {db_url is not None}")

# Test database connection
def test_db_connection(db_url):
    try:
        # Convert URL for psycopg connection (remove +psycopg for direct connection)
        test_url = db_url.replace("postgresql+psycopg://", "postgresql://")
        
        # Connect using psycopg to test
        conn = psycopg.connect(test_url)
        
        cursor = conn.cursor()
        
        # Check if pgvector extension exists
        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        vector_ext = cursor.fetchone()
        print(f"PgVector extension installed: {vector_ext is not None}")
        
        # Check database permissions
        cursor.execute("SELECT current_user, current_database();")
        user_db = cursor.fetchone()
        print(f"Connected as user: {user_db[0]}, database: {user_db[1]}")
        
        cursor.close()
        conn.close()
        print("✅ Database connection successful")
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

# Test the connection
if not test_db_connection(db_url):
    raise ConnectionError("Cannot connect to database")

# Create knowledge base
knowledge_base = CSVKnowledgeBase(
    path=csv_path,
    vector_db=PgVector(
        table_name="csv_documents",
        db_url=db_url,
    ),
)

print("✅ Knowledge base created successfully")