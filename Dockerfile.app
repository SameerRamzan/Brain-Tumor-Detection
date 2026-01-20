# Use Python 3.9 Slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app.py auth_ui.py styles.py styles_auth_ui.py create_admin.py ./
# Copy assets if any (like logo.png)
COPY logo.png .

# Expose port and run
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]