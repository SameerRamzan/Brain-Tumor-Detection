# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the new frontend requirements file into the container at /app
COPY requirements-frontend.txt .

# Install any needed packages specified in the requirements file
# --no-cache-dir: Disables the cache, which is not needed in a final image
# -r: Specifies the requirements file
RUN pip install --no-cache-dir -r requirements-frontend.txt

# Copy the rest of your application code from your local repo to the container
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define the command to run your app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]