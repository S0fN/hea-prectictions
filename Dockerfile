# Dockerfile

# 1. Use a standard, slim Python base image
FROM python:3.10-slim

# 2. Install your required system library
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy requirements file and install Python packages
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all your project files into the container
COPY . .

# 6. Expose the port Railway will use. This is good practice.
EXPOSE 8080

# 7. Define the command to run your app.
# This replaces the "Custom Start Command" from the UI.
# It uses the exact command you provided.
CMD [ "streamlit", "run", "app.py", "--server.port", "$PORT", "--server.enableCORS", "false", "--server.enableXsrfProtection", "false" ]
