FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e .
COPY . .
EXPOSE 8000 8501
# Default: API. For the dashboard:  docker run -p 8501:8501 <image> streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
