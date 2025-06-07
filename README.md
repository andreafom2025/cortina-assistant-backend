# Cortina Virtual Assistant - Backend (FastAPI + RAG)

This is the backend for the Cortina Virtual Assistant, which uses Retrieval-Augmented Generation (RAG) with GPT-4 to provide multilingual tourist information about Cortina d'Ampezzo and the Dolomites.

## 🚀 Technologies
- FastAPI
- OpenAI GPT-4
- FAISS (for vector search)
- LangChain

## 📦 Requirements
- Python 3.10+
- OpenAI API Key
- FAISS index folder named `faiss_cortina_index`

## 🛠️ Setup Instructions

1. Clone this repo:
```bash
git clone https://github.com/your-username/cortina-assistant-backend.git
cd cortina-assistant-backend
```

2. Create a `.env` file:
```
OPENAI_API_KEY=your_openai_key
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run locally:
```bash
uvicorn main:app --reload --port 10000
```

5. Deploy on Render:
- Set environment variable `OPENAI_API_KEY`
- Use start command: `uvicorn main:app --host 0.0.0.0 --port 10000`

## 🧠 Query API
POST `/query`
```json
{
  "query": "How do I get to Tre Cime?",
  "language": "en"
}
```

GET `/health`
```json
{ "status": "ok" }
```
