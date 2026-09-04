FROM python:3.11-slim

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --default-timeout=300 --no-cache-dir -r requirements-docker.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501"]