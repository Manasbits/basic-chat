import psycopg
import os

# Your RDS connection details
db_url = "postgresql+psycopg://mainuser:Manas123456@pgvector-db.czegegwgkvr5.ap-south-1.rds.amazonaws.com:5432/postgres?sslmode=require"

try:
    # Test basic connection
    conn = psycopg.connect(db_url.replace("postgresql+psycopg://", "postgresql://"))
    print("✅ AWS RDS connection successful")
    
    # Test if pgvector extension is available
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        result = cur.fetchone()
        if result:
            print("✅ PgVector extension is installed")
        else:
            print("❌ PgVector extension not found - you need to install it")
    
    conn.close()
    
except Exception as e:
    print(f"❌ AWS RDS connection failed: {e}")