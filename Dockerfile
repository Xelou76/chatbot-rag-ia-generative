FROM python:3.12-slim

WORKDIR /app

# Dépendances d'abord : cette couche reste en cache tant que requirements.txt ne change pas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code de l'application (le .dockerignore exclut .env, .venv et les bases Chroma)
COPY main.py app.py ./

EXPOSE 8501

# La clé API est passée au lancement, jamais copiée dans l'image :
#   docker run -p 8501:8501 --env-file .env -v tp-rag-chroma:/app/chroma_db_openai tp-rag
# Le volume garde la base Chroma entre deux lancements (sinon elle est reconstruite à chaque démarrage).
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
