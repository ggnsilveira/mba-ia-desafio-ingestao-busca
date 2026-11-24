# Ingestão e Busca Semântica com LangChain e Postgres

Pré-requisitos: Python 3, Docker e uma API Key do Google Gemini.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Preencha a `GOOGLE_API_KEY` no arquivo `.env` e execute:

```bash
docker compose up -d
python src/ingest.py
python src/chat.py
```
