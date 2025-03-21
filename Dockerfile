# Extend the official Airflow image (version must match the one in docker-compose.yaml)
FROM apache/airflow:2.10.5

# Switch to root to install packages
USER root

# Install Git (needed for pip install from GitHub)
RUN apt-get update && apt-get install -y git

# Switch to airflow user after installing system packages
USER airflow

# Copy your DAGs, data_processing folder, and plugins into Airflow
COPY data_processing /opt/airflow/dags/data_processing
COPY .env /opt/airflow/dags

# Copy and install Python dependencies
COPY requirements.txt /tmp/requirements.txt

# Install Python packages (docling, multipart, and your requirements)
RUN pip install --no-cache-dir -r /tmp/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir docling==2.16.0 --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir python-multipart --extra-index-url https://download.pytorch.org/whl/cpu