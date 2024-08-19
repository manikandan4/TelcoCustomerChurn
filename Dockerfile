# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
    
# Expose port 8501 to the outside world
EXPOSE 8501

# Streamlit specific environment variables
ENV STREAMLIT_CONFIG_FILE="/app/.streamlit/config.toml"

# Run Streamlit
CMD ["streamlit", "run", "customer_churn_predict.py", "--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
