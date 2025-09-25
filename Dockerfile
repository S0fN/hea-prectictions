# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# --- This is the crucial step to fix the error ---
# Update the package list and install the missing libgomp1 library
RUN apt-get update && apt-get install -y libgomp1

# Copy the file that lists the Python dependencies
COPY requirements.txt ./

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Tell the container to run your Streamlit app and listen on the correct port
CMD streamlit run app.py --server.port $PORT --server.enableCORS false --server.enableXsrfProtection false
