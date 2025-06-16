FROM python:3.11-slim

# Create app directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your application files
COPY agent.py .
COPY workflow.py .
COPY knowledge_base.py .
COPY query_results.csv .

# Expose ports for both applications
EXPOSE 7777

# Set environment variables for production
ENV HOST=0.0.0.0
ENV PORT=7777

# Run both applications
CMD python workflow.py
