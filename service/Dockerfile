FROM python:3.9-slim

WORKDIR /app

# 1) Instala dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copia el código y el embeddings.json
COPY main.py ./
COPY ../embeddings.json ./

ENV PORT=8080
EXPOSE 8080

# 3) Arranca el servidor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
