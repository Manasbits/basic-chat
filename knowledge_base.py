import os
from pathlib import Path
from agno.knowledge.csv import CSVKnowledgeBase
from agno.vectordb.pgvector import PgVector
from dotenv import load_dotenv
import psycopg2
from urllib.parse import urlparse

load_dotenv()

# Enhanced debugging
print(f"Current working directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")

csv_path = Path('query-results.csv')
print(f"CSV file exists: {csv_path.exists()}")
if csv_path.exists():
    print(f"CSV file size: {csv_path.stat().st_size} bytes")

# Get and validate DATABASE_URL
db_url = os.getenv("DATABASE_URL")
print(f"Database URL configured: {db_url is not None}")

if not db_url:
    raise ValueError("DATABASE_URL environment variable is not set")

# Test database connection
def test_db_connection(db_url):
    try:
        # Parse the URL to get connection parameters
        parsed = urlparse(db_url)
        
        # Connect using psycopg2 directly to test
        conn = psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port or 5432,
            database=parsed.path[1:],  # Remove leading '/'
            user=parsed.username,
            password=parsed.password
        )
        
        cursor = conn.cursor()
        
        # Check if pgvector extension exists
        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        vector_ext = cursor.fetchone()
        print(f"PgVector extension installed: {vector_ext is not None}")
        
        # Check if we can create tables
        cursor.execute("SELECT has_table_privilege(current_user, 'information_schema.tables', 'CREATE');")
        can_create = cursor.fetchone()[0]
        print(f"Can create tables: {can_create}")
        
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