FROM python:3.10-slim

WORKDIR /app

USER root

RUN apt-get update && apt-get install -y git

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir docling==2.16.0 python-multipart --extra-index-url https://download.pytorch.org/whl/cpu

COPY api /app/api
COPY data_processing /app/data_processing

 
EXPOSE 8000

CMD ["uvicorn", "api.fastapi_backend:app", "--host", "0.0.0.0", "--port", "8000"]