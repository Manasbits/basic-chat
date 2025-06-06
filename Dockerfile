FROM python:3.11-slim

# Create app directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your application files
COPY agent.py .
COPY knowledge_base.py .
COPY .env .

# Expose port
EXPOSE 7777

# Set environment variables for production
ENV HOST=0.0.0.0
ENV PORT=7777

# Run the application
CMD ["python", "agent.py"]