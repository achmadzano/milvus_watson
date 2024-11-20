# Dockerfile
# Use the official Python image as a base
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install the dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app code to the container
COPY . .

# Expose the port that Uvicorn will run on
EXPOSE 8080

# Run the FastAPI app with Uvicorn
CMD ["uvicorn", "testt_n-1_final:app", "--host", "150.238.38.132", "--port", "8080"]
