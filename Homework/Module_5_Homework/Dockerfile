# Use the base image
FROM svizor/zoomcamp-model:3.11.5-slim

# Set the working directory inside the container
WORKDIR /app

# Copy Pipenv files to the working directory
COPY Pipfile Pipfile.lock /app/

# Install dependencies using pipenv
RUN pip install pipenv && pipenv install --system --deploy

# Copy the Flask application script to the working directory
COPY flask_web_service.py /app/

# Set the environment variable for Flask
ENV FLASK_APP=flask_web_service.py

# Expose the port that the application will run on
EXPOSE 9696

# Use ENTRYPOINT to run Gunicorn
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "flask_web_service:app"]
